from __future__ import annotations

ARC_TRAINER_PATCH_ID = "arc-scratch-v3-complete"
ARC_TRAINER_ARCH = "recursive-deltanet-v1-generalization-v2"

import copy
import random

import numpy as np
import torch

from sefer.controllers.arc_reasoner import ARCReasoner
from sefer.representation.arc_grid import arc_grid_loss
from sefer.evaluation.arc import evaluate_arc, evaluate_arc_direct


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_loss(model, H, target, target_shape):
    c, h, w = model.decode_grid(H)
    color_loss, stats = arc_grid_loss(
        c, h, w, target, target_shape,
        foreground_boost=model.cfg.foreground_boost,
        balance_mix=model.cfg.color_balance_mix,
    )
    return color_loss, stats["shape_loss"], stats


def _loo_auxiliary(model, batch):
    """Predict one demonstration from the other demonstrations when possible."""
    counts = batch.demo_mask.sum(dim=1)
    valid_rows = torch.nonzero(counts >= 2, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return None

    # Deterministic choice keeps the auxiliary stable; dataset sampling already
    # randomizes which task/examples occupy each batch.
    held = []
    for b in valid_rows.tolist():
        held.append(int(torch.nonzero(batch.demo_mask[b], as_tuple=False)[0, 0]))
    held = torch.tensor(held, device=batch.demo_mask.device, dtype=torch.long)

    dx = batch.demos_x[valid_rows].clone()
    dy = batch.demos_y[valid_rows].clone()
    dxs = batch.demos_x_shapes[valid_rows].clone()
    dys = batch.demos_y_shapes[valid_rows].clone()
    dm = batch.demo_mask[valid_rows].clone()

    rows = torch.arange(valid_rows.numel(), device=dm.device)
    q = dx[rows, held]
    qs = dxs[rows, held]
    y = dy[rows, held]
    ys = dys[rows, held]
    dm[rows, held] = False

    rule = model.encode_rule(dx, dy, dxs, dys, dm)
    Hq = model.encode_grid(q, qs)
    rule = model.condition_rule_on_query(rule, Hq)
    Hd = model.predict_goal(Hq, rule)
    c, s, _ = _decode_loss(model, Hd, y, ys)
    return c + model.cfg.shape_weight * s


def _set_requires_grad(module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


def _recursive_loo_auxiliary(model, batch):
    """Train recursion on a demonstration withheld from the task context.

    The base encoder/direct predictor remain frozen during this phase. The
    recursive cell must improve the held-out demonstration prediction over
    successive reasoning steps.
    """
    counts = batch.demo_mask.sum(dim=1)
    valid_rows = torch.nonzero(counts >= 2, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return None, None, {}

    held = []
    for b in valid_rows.tolist():
        held.append(int(torch.nonzero(batch.demo_mask[b], as_tuple=False)[0, 0]))
    held = torch.tensor(held, device=batch.demo_mask.device, dtype=torch.long)

    dx = batch.demos_x[valid_rows].clone()
    dy = batch.demos_y[valid_rows].clone()
    dxs = batch.demos_x_shapes[valid_rows].clone()
    dys = batch.demos_y_shapes[valid_rows].clone()
    dm = batch.demo_mask[valid_rows].clone()

    rows = torch.arange(valid_rows.numel(), device=dm.device)
    q = dx[rows, held]
    qs = dxs[rows, held]
    y = dy[rows, held]
    ys = dys[rows, held]
    dm[rows, held] = False

    rule = model.encode_rule(dx, dy, dxs, dys, dm)
    Hq = model.encode_grid(q, qs)
    rule = model.condition_rule_on_query(rule, Hq)
    Ht = model.encode_grid(y, ys).detach()

    _, info = model.recursive_reason(
        Hq, rule, steps=model.cfg.recursive_steps, return_trace=True
    )

    step_losses = []
    step_latents = []
    for state in info["states"]:
        c, s, _ = _decode_loss(model, state, y, ys)
        latent = (state - Ht).pow(2).mean()
        total = c + model.cfg.shape_weight * s + model.cfg.recursive_latent_weight * latent
        step_losses.append(total)
        step_latents.append(latent)

    final_loss = step_losses[-1]
    margin = float(model.cfg.recursive_step_improvement_margin)
    improve_terms = [
        torch.relu(next_loss - prev_loss + margin)
        for prev_loss, next_loss in zip(step_losses[:-1], step_losses[1:])
    ]
    improvement_loss = (
        torch.stack(improve_terms).mean()
        if improve_terms else torch.zeros((), device=final_loss.device)
    )

    stats = {
        "loo_initial": float(step_losses[0].detach().item()),
        "loo_final": float(step_losses[-1].detach().item()),
        "loo_latent_initial": float(step_latents[0].detach().item()),
        "loo_latent_final": float(step_latents[-1].detach().item()),
    }
    return final_loss, improvement_loss, stats


def train_arc_v1(cfg, train_data, val_data=None):
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    device = torch.device(cfg.device)
    model = ARCReasoner(cfg).to(device)

    print(f"ARC recursive-v1 device={device}")
    print(f"ARC recursive-v1 tasks={train_data.task_count} batch={cfg.batch_size}")
    print(f"ARC recursive-v1 params={sum(p.numel() for p in model.parameters()):,}")
    print(
        f"ARC recursive-v1 curriculum codec={cfg.codec_steps} "
        f"direct={cfg.direct_steps} recursive={cfg.program_steps} "
        f"reasonSteps={cfg.recursive_steps}"
    )

    # Phase A: grid codec.
    codec_params = list(model.grid_encoder.parameters()) + list(model.grid_decoder.parameters())
    codec_opt = torch.optim.AdamW(
        codec_params, lr=cfg.codec_lr, weight_decay=cfg.weight_decay
    )
    for step in range(1, cfg.codec_steps + 1):
        model.train()
        grid, shape = train_data.sample_grid_batch(cfg.batch_size, device)
        H = model.encode_grid(grid, shape)
        c, h, w = model.decode_grid(H)
        color_loss, stats = arc_grid_loss(
            c, h, w, grid, shape,
            foreground_boost=cfg.foreground_boost,
            balance_mix=cfg.color_balance_mix,
        )
        loss = color_loss + cfg.shape_weight * stats["shape_loss"]
        codec_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec_params, cfg.grad_clip)
        codec_opt.step()

        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.codec_steps:
            print(
                f"arc codec step={step:04d} loss={loss.item():.4f} "
                f"pixelAcc={stats['pixel_acc'].item():.3f} "
                f"fgAcc={stats['foreground_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f}"
            )

    # Phase B: direct task-conditioned predictor.
    direct_opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.direct_lr, weight_decay=cfg.weight_decay
    )
    best_direct = None
    best_direct_score = -1.0

    for step in range(1, cfg.direct_steps + 1):
        model.train()
        batch = train_data.sample_batch(cfg.batch_size, device)
        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        Hq = model.encode_grid(batch.query_x, batch.query_shape)
        rule = model.condition_rule_on_query(rule, Hq)
        Hd = model.predict_goal(Hq, rule)
        Ht = model.encode_grid(batch.target_y, batch.target_shape).detach()

        color_loss, shape_loss, stats = _decode_loss(
            model, Hd, batch.target_y, batch.target_shape
        )
        latent_loss = (Hd - Ht).pow(2).mean()
        loo = _loo_auxiliary(model, batch)
        loss = (
            color_loss
            + cfg.shape_weight * shape_loss
            + cfg.direct_latent_weight * latent_loss
        )
        if loo is not None:
            loss = loss + cfg.loo_demo_weight * loo

        direct_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        direct_opt.step()

        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.direct_steps:
            print(
                f"arc direct step={step:04d} loss={loss.item():.4f} "
                f"pixelAcc={stats['pixel_acc'].item():.3f} "
                f"fgAcc={stats['foreground_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f} "
                f"latent={latent_loss.item():.4f}"
            )

        if val_data is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            m = evaluate_arc_direct(model, val_data, limit=cfg.eval_tasks, device=device)
            score = m["exact"] + 0.10 * m["pixel_acc"]
            print(
                f"arc direct val step={step:04d} exact={m['exact']:.3f} "
                f"pixel={m['pixel_acc']:.3f} shape={m['shape_acc']:.3f}"
            )
            if score > best_direct_score:
                best_direct_score = score
                best_direct = copy.deepcopy(model.state_dict())

    if best_direct is not None:
        model.load_state_dict(best_direct, strict=True)
        print(f"ARC recursive-v1 restored best direct score={best_direct_score:.4f}")

    # Phase C: recursive DeltaNet reasoning.
    # Preserve the best direct/meta representation. The recurrent adapter must
    # improve it instead of rewriting the entire model.
    _set_requires_grad(model, False)
    _set_requires_grad(model.recursive_cell, True)

    recursive_param_groups = [
        {"params": list(model.recursive_cell.parameters()), "lr": cfg.recursive_lr},
    ]
    if getattr(cfg, "recursive_unfreeze_rule_encoder", False):
        _set_requires_grad(model.rule_encoder, True)
        _set_requires_grad(model.query_rule_refiner, True)
        recursive_param_groups.append({
            "params": list(model.rule_encoder.parameters())
                    + list(model.query_rule_refiner.parameters()),
            "lr": cfg.recursive_lr * cfg.recursive_rule_lr_scale,
        })

    recursive_opt = torch.optim.AdamW(
        recursive_param_groups, weight_decay=cfg.weight_decay
    )
    print(
        "ARC recursive-v2 frozen base: training recursive_cell"
        + (" + low-LR rule encoder" if cfg.recursive_unfreeze_rule_encoder else "")
    )
    best_recursive = None
    best_recursive_score = -1.0
    best_recursive_step = None

    for step in range(1, cfg.program_steps + 1):
        model.train()
        batch = train_data.sample_batch(cfg.batch_size, device)

        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        Hq = model.encode_grid(batch.query_x, batch.query_shape)
        rule = model.condition_rule_on_query(rule, Hq)
        Ht = model.encode_grid(batch.target_y, batch.target_shape).detach()

        Hr, info = model.recursive_reason(
            Hq, rule, steps=cfg.recursive_steps, return_trace=True
        )
        Hd = info["states"][0]

        color_loss, shape_loss, stats = _decode_loss(
            model, Hr, batch.target_y, batch.target_shape
        )
        final_latent = (Hr - Ht).pow(2).mean()
        direct_err = (Hd - Ht).pow(2).mean(dim=(1, 2))
        final_err = (Hr - Ht).pow(2).mean(dim=(1, 2))
        consistency = torch.relu(final_err - direct_err).mean()

        intermediate = torch.zeros((), device=device)
        if len(info["states"]) > 2:
            losses = [
                (s - Ht).pow(2).mean()
                for s in info["states"][1:-1]
            ]
            intermediate = torch.stack(losses).mean()

        fast_reg = info["fast_state"].pow(2).mean()
        halt_loss = torch.zeros((), device=device)
        if info["halt_logits"]:
            # Encourage confidence only when recursion beats the direct state.
            improved = (final_err.detach() < direct_err.detach()).float()
            halt_target = improved
            halt_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                info["halt_logits"][-1], halt_target
            )

        loo_final, loo_improve, loo_stats = _recursive_loo_auxiliary(model, batch)
        loss = (
            cfg.recursive_grid_weight * (color_loss + cfg.shape_weight * shape_loss)
            + cfg.recursive_latent_weight * final_latent
            + cfg.recursive_intermediate_weight * intermediate
            + cfg.recursive_consistency_weight * consistency
            + cfg.recursive_fast_reg_weight * fast_reg
            + cfg.halt_weight * halt_loss
        )
        if loo_final is not None:
            loss = (
                loss
                + cfg.recursive_loo_weight * loo_final
                + cfg.recursive_step_improvement_weight * loo_improve
            )

        recursive_opt.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        recursive_opt.step()

        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.program_steps:
            last_stats = info["stats"][-1]
            print(
                f"arc recursive step={step:04d} loss={loss.item():.4f} "
                f"pixelAcc={stats['pixel_acc'].item():.3f} "
                f"fgAcc={stats['foreground_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f} "
                f"latent={final_latent.item():.4f} "
                f"cons={consistency.item():.4f} "
                f"fastNorm={last_stats['fast_norm'].mean().item():.4f} "
                f"eta={last_stats['eta'].mean().item():.3f} "
                f"gate={last_stats['gate'].mean().item():.3f} "
                f"halt={last_stats['halt_prob'].mean().item():.3f} "
                f"loo0={loo_stats.get('loo_initial', float('nan')):.3f} "
                f"looN={loo_stats.get('loo_final', float('nan')):.3f} "
                f"looImprove={(loo_improve.item() if loo_improve is not None else float('nan')):.4f}"
            )

        if val_data is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            m = evaluate_arc(model, val_data, limit=cfg.eval_tasks, device=device)
            score = m["exact"] + 0.10 * m["pixel_acc"]
            print(
                f"arc recursive val step={step:04d} exact={m['exact']:.3f} "
                f"pixel={m['pixel_acc']:.3f} shape={m['shape_acc']:.3f} "
                f"directPixel={m['direct_pixel_acc']:.3f} "
                f"gain={m['recursive_gain']:+.3f}"
            )
            if score > best_recursive_score:
                best_recursive_score = score
                best_recursive_step = step
                best_recursive = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_recursive,
                        "cfg": vars(cfg),
                        "metrics": m,
                        "arch": ARC_TRAINER_ARCH,
                    },
                    cfg.arc_best_checkpoint_path,
                )

    if cfg.restore_best_at_end and best_recursive is not None:
        model.load_state_dict(best_recursive, strict=True)
        print(f"ARC recursive-v2 restored best checkpoint step={best_recursive_step}")

    _set_requires_grad(model, True)

    final_metrics = evaluate_arc(
        model, val_data or train_data, limit=cfg.eval_tasks, device=device
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": vars(cfg),
            "metrics": final_metrics,
            "arch": ARC_TRAINER_ARCH,
        },
        cfg.arc_checkpoint_path,
    )
    # Preserve the launcher-compatible pre-NEAT name, but it now points to the
    # recursive model and no NEAT phase is required.
    if cfg.arc_pre_neat_checkpoint_path != cfg.arc_checkpoint_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": vars(cfg),
                "metrics": final_metrics,
                "arch": ARC_TRAINER_ARCH,
            },
            cfg.arc_pre_neat_checkpoint_path,
        )

    print(
        f"ARC recursive-v1 final exact={final_metrics['exact']:.3f} "
        f"pixel={final_metrics['pixel_acc']:.3f} "
        f"shape={final_metrics['shape_acc']:.3f} "
        f"directPixel={final_metrics['direct_pixel_acc']:.3f} "
        f"gain={final_metrics['recursive_gain']:+.3f}"
    )
    return model, final_metrics


def load_arc_v1_checkpoint(cfg, path: str | None = None,
                           device: str | torch.device | None = None):
    device = torch.device(device or cfg.device)
    model = ARCReasoner(cfg).to(device)
    payload = torch.load(path or cfg.arc_checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload
