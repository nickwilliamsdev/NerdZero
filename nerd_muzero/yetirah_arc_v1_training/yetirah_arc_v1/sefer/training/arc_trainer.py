from __future__ import annotations

ARC_TRAINER_PATCH_ID = "arc-scratch-v3-complete"
ARC_TRAINER_ARCH = "recursive-deltanet-v1-staged-predictor-adapt-v12"

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from sefer.controllers.arc_reasoner import ARCReasoner
from sefer.representation.arc_grid import arc_grid_loss, shape_mask
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


def _per_sample_grid_objective(model, H, target, target_shape):
    """Per-example decoded grid objective used only to supervise stopping depth."""
    color_logits, h_logits, w_logits = model.decode_grid(H)
    mask = shape_mask(target_shape, model.cfg.max_grid_size)
    per_cell = F.cross_entropy(color_logits, target.long(), reduction="none")
    denom = mask.flatten(1).sum(dim=1).clamp_min(1)
    cell = (per_cell * mask.float()).flatten(1).sum(dim=1) / denom

    h_target = (target_shape[:, 0] - 1).clamp(0, model.cfg.max_grid_size - 1)
    w_target = (target_shape[:, 1] - 1).clamp(0, model.cfg.max_grid_size - 1)
    shape = (
        F.cross_entropy(h_logits, h_target.long(), reduction="none")
        + F.cross_entropy(w_logits, w_target.long(), reduction="none")
    )
    return cell + model.cfg.shape_weight * shape


def _adaptive_step_losses(model, rule, info, target, target_shape):
    """Soft/listwise + pairwise supervision for the state selector."""
    states = info["states"]
    if not states:
        z = torch.zeros((), device=target.device)
        return None, z, z, None, None, {}

    step_losses = torch.stack(
        [_per_sample_grid_objective(model, s, target, target_shape) for s in states],
        dim=1,
    )  # [B,K], lower is better
    target_losses = step_losses.detach()
    best_step_idx = target_losses.argmin(dim=1)

    rank_logits = model.score_recursive_states(rule, info)
    temp = max(float(model.cfg.state_rank_temperature), 1e-4)

    # Listwise target preserves relative quality instead of collapsing to argmin.
    target_probs = torch.softmax(-target_losses / temp, dim=1)
    log_probs = torch.log_softmax(rank_logits / temp, dim=1)
    listwise_loss = -(target_probs * log_probs).sum(dim=1).mean()

    # Pairwise preference weighted by how different the candidate losses are.
    pair_terms = []
    K = target_losses.shape[1]
    for i in range(K):
        for j in range(i + 1, K):
            diff = target_losses[:, j] - target_losses[:, i]
            sign = torch.sign(diff)
            valid = sign != 0
            if valid.any():
                score_diff = rank_logits[:, i] - rank_logits[:, j]
                weight = diff.abs().detach().clamp(max=1.0)
                pair = F.softplus(
                    -sign * score_diff + float(model.cfg.state_rank_pairwise_margin)
                )
                pair_terms.append((pair[valid] * weight[valid]).mean())

    if pair_terms:
        pairwise_loss = torch.stack(pair_terms).mean()
    else:
        pairwise_loss = torch.zeros((), device=target.device)

    # Expected regret directly penalizes probability mass on worse states.
    pred_probs = torch.softmax(rank_logits, dim=1)
    regret = target_losses - target_losses.min(dim=1, keepdim=True).values
    regret_loss = (pred_probs * regret).sum(dim=1).mean()

    rank_loss = (
        model.cfg.state_rank_listwise_weight * listwise_loss
        + model.cfg.state_rank_pairwise_weight * pairwise_loss
        + model.cfg.state_rank_regret_weight * regret_loss
    )

    per_step_loss = (
        step_losses[:, 1:].mean()
        if step_losses.shape[1] > 1 else step_losses.mean()
    )

    rank_stats = {
        "rank_listwise": float(listwise_loss.detach().item()),
        "rank_pairwise": float(pairwise_loss.detach().item()),
        "rank_regret": float(regret_loss.detach().item()),
        "rank_acc": float(
            (rank_logits.detach().argmax(dim=1) == best_step_idx).float().mean().item()
        ),
    }
    return step_losses, rank_loss, per_step_loss, best_step_idx, rank_logits, rank_stats



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
    """Train recursion on multiple randomly held-out demonstrations per task."""
    counts = batch.demo_mask.sum(dim=1)
    eligible_rows = torch.nonzero(counts >= 2, as_tuple=False).flatten()
    if eligible_rows.numel() == 0:
        return None, None, {}

    max_permutations = 2
    all_final_losses = []
    all_improve_losses = []
    all_initial_values = []
    all_final_values = []
    all_latent_initial = []
    all_latent_final = []

    for b in eligible_rows.tolist():
        valid_demo_ids = torch.nonzero(
            batch.demo_mask[b], as_tuple=False
        ).flatten()

        perm = valid_demo_ids[
            torch.randperm(valid_demo_ids.numel(), device=valid_demo_ids.device)
        ]
        held_ids = perm[:min(max_permutations, perm.numel())]

        for held_id_t in held_ids:
            held_id = int(held_id_t.item())

            dx = batch.demos_x[b:b + 1].clone()
            dy = batch.demos_y[b:b + 1].clone()
            dxs = batch.demos_x_shapes[b:b + 1].clone()
            dys = batch.demos_y_shapes[b:b + 1].clone()
            dm = batch.demo_mask[b:b + 1].clone()

            q = dx[:, held_id]
            qs = dxs[:, held_id]
            y = dy[:, held_id]
            ys = dys[:, held_id]
            dm[:, held_id] = False

            rule = model.encode_rule(dx, dy, dxs, dys, dm)
            Hq = model.encode_grid(q, qs)
            rule = model.condition_rule_on_query(rule, Hq)
            Ht = model.encode_grid(y, ys).detach()

            _, info = model.recursive_reason(
                Hq, rule, steps=model.cfg.recursive_steps, return_trace=True
            )

            step_losses = []
            step_latent = []
            for state in info["states"]:
                c, s, _ = _decode_loss(model, state, y, ys)
                latent = (state - Ht).pow(2).mean()
                decoded = c + model.cfg.shape_weight * s
                step_losses.append(
                    decoded + model.cfg.recursive_latent_weight * latent
                )
                step_latent.append(latent)

            final_loss = step_losses[-1]

            improve_terms = []
            margin = float(model.cfg.recursive_step_improvement_margin)
            for prev_loss, next_loss in zip(step_losses[:-1], step_losses[1:]):
                improve_terms.append(
                    torch.relu(next_loss - prev_loss + margin)
                )

            if improve_terms:
                improvement_loss = torch.stack(improve_terms).mean()
            else:
                improvement_loss = torch.zeros(
                    (), device=final_loss.device, dtype=final_loss.dtype
                )

            all_final_losses.append(final_loss)
            all_improve_losses.append(improvement_loss)
            all_initial_values.append(step_losses[0].detach())
            all_final_values.append(step_losses[-1].detach())
            all_latent_initial.append(step_latent[0].detach())
            all_latent_final.append(step_latent[-1].detach())

    if not all_final_losses:
        return None, None, {}

    final_loss = torch.stack(all_final_losses).mean()
    improvement_loss = torch.stack(all_improve_losses).mean()

    stats = {
        "loo_initial": float(torch.stack(all_initial_values).mean().item()),
        "loo_final": float(torch.stack(all_final_values).mean().item()),
        "loo_latent_initial": float(torch.stack(all_latent_initial).mean().item()),
        "loo_latent_final": float(torch.stack(all_latent_final).mean().item()),
        "loo_permutations": len(all_final_losses),
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
    # Stage 1: keep the direct/meta representation fully frozen while the
    # recursive cell learns a stable improvement operator.
    # Stage 2: after recursive_rule_unfreeze_step, adapt only the rule/query
    # pathway at a much smaller LR. The codec and direct predictor remain frozen.
    _set_requires_grad(model, False)
    _set_requires_grad(model.recursive_cell, True)

    recursive_param_groups = [
        {"params": list(model.recursive_cell.parameters()), "lr": cfg.recursive_lr},
    ]
    recursive_opt = torch.optim.AdamW(
        recursive_param_groups, weight_decay=cfg.weight_decay
    )

    rule_modules = [
        model.demo_slot_encoder,
        model.demo_slot_score,
        model.demo_pair_encoder,
        model.rule_encoder,
        model.query_slot_proj,
        model.query_rule_query,
        model.query_rule_refiner,
        model.query_rule_norm,
    ]
    rule_params = [
        p for module in rule_modules
        for p in module.parameters()
    ]
    rule_unfrozen = False

    predictor_modules = [
        model.direct_rule_to_slots,
        model.direct_norm,
    ]
    predictor_params = [
        p for module in predictor_modules
        for p in module.parameters()
    ]
    predictor_unfrozen = False

    print(
        "ARC recursive-v12 stage1: recursive_cell only; "
        f"rule path at step={cfg.recursive_rule_unfreeze_step} "
        f"(lrScale={cfg.recursive_rule_lr_scale:.3f}); "
        f"goal predictor at step={cfg.recursive_predictor_unfreeze_step} "
        f"(lrScale={cfg.recursive_predictor_lr_scale:.3f})"
    )
    best_recursive = None
    best_recursive_score = -1.0
    best_recursive_step = None
    recursive_bad_evals = 0
    recursive_best_gain = float("-inf")

    for step in range(1, cfg.program_steps + 1):
        model.train()

        if (
            getattr(cfg, "recursive_unfreeze_rule_encoder", False)
            and not rule_unfrozen
            and step >= cfg.recursive_rule_unfreeze_step
        ):
            for module in rule_modules:
                _set_requires_grad(module, True)

            recursive_opt.add_param_group({
                "params": rule_params,
                "lr": cfg.recursive_lr * cfg.recursive_rule_lr_scale,
                "weight_decay": cfg.weight_decay,
            })
            rule_unfrozen = True
            print(
                f"ARC recursive-v12 stage2 step={step}: "
                f"unfroze rule/query path lr="
                f"{cfg.recursive_lr * cfg.recursive_rule_lr_scale:.2e}"
            )

        if (
            not predictor_unfrozen
            and step >= cfg.recursive_predictor_unfreeze_step
        ):
            for module in predictor_modules:
                _set_requires_grad(module, True)

            recursive_opt.add_param_group({
                "params": predictor_params,
                "lr": cfg.recursive_lr * cfg.recursive_predictor_lr_scale,
                "weight_decay": cfg.weight_decay,
            })
            predictor_unfrozen = True
            print(
                f"ARC recursive-v12 stage3 step={step}: "
                f"unfroze direct goal predictor lr="
                f"{cfg.recursive_lr * cfg.recursive_predictor_lr_scale:.2e}"
            )

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
        step_grid_losses, rank_loss, per_step_grid_loss, best_step_idx, rank_logits, rank_stats = _adaptive_step_losses(
            model, rule, info, batch.target_y, batch.target_shape
        )

        loo_final, loo_improve, loo_stats = _recursive_loo_auxiliary(model, batch)
        loss = (
            cfg.recursive_grid_weight * (color_loss + cfg.shape_weight * shape_loss)
            + cfg.recursive_latent_weight * final_latent
            + cfg.recursive_intermediate_weight * intermediate
            + cfg.recursive_consistency_weight * consistency
            + cfg.recursive_fast_reg_weight * fast_reg
            + cfg.recursive_per_step_grid_weight * per_step_grid_loss
        )
        if loo_final is not None:
            loss = (
                loss
                + cfg.recursive_loo_weight * loo_final
                + cfg.recursive_step_improvement_weight * loo_improve
            )

        recursive_opt.zero_grad(set_to_none=True)
        loss.backward()
        recursive_params = [
            p for p in model.recursive_cell.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if recursive_params:
            torch.nn.utils.clip_grad_norm_(recursive_params, cfg.grad_clip)

        if rule_unfrozen:
            active_rule_params = [
                p for p in rule_params
                if p.requires_grad and p.grad is not None
            ]
            if active_rule_params:
                torch.nn.utils.clip_grad_norm_(
                    active_rule_params, cfg.recursive_rule_grad_clip
                )

        if predictor_unfrozen:
            active_predictor_params = [
                p for p in predictor_params
                if p.requires_grad and p.grad is not None
            ]
            if active_predictor_params:
                torch.nn.utils.clip_grad_norm_(
                    active_predictor_params,
                    cfg.recursive_predictor_grad_clip,
                )

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
                f"looImprove={(loo_improve.item() if loo_improve is not None else float('nan')):.4f} "
                f"looPerms={loo_stats.get('loo_permutations', 0)} "
                f"stage={'predictor-adapt' if predictor_unfrozen else ('rule-adapt' if rule_unfrozen else 'cell-only')} "
                f"bestStep={(best_step_idx.float().mean().item() + 1.0 if best_step_idx is not None else float('nan')):.2f} "
                f"rankLoss={rank_loss.item():.4f} "f"rankAcc={rank_stats.get('rank_acc', float('nan')):.3f} "f"list={rank_stats.get('rank_listwise', float('nan')):.3f} "f"pair={rank_stats.get('rank_pairwise', float('nan')):.3f} "f"regret={rank_stats.get('rank_regret', float('nan')):.3f}"
            )

        if val_data is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            m = evaluate_arc(model, val_data, limit=cfg.eval_tasks, device=device)
            score = (
                m["exact"]
                + 0.10 * m["adaptive_pixel_acc"]
                + cfg.recursive_gain_score_weight * m["recursive_gain"]
            )
            print(
                f"arc recursive val step={step:04d} exact={m['exact']:.3f} "
                f"ensemblePixel={m['adaptive_pixel_acc']:.3f} "
                f"fixedPixel={m['fixed_pixel_acc']:.3f} "
                f"oraclePixel={m['oracle_pixel_acc']:.3f} "
                f"directPixel={m['direct_pixel_acc']:.3f} "
                f"gain={m['recursive_gain']:+.3f} "
                f"hardDepth={m['avg_chosen_step']:.2f} "f"ensDepth={m.get('avg_ensemble_depth', float('nan')):.2f} score={score:.4f}"
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

            gain = float(m["recursive_gain"])
            if gain > recursive_best_gain + cfg.recursive_early_stop_min_delta:
                recursive_best_gain = gain
                recursive_bad_evals = 0
            else:
                recursive_bad_evals += 1

            if (
                step >= cfg.recursive_min_steps_before_stop
                and recursive_bad_evals >= cfg.recursive_early_stop_patience
            ):
                print(
                    f"ARC recursive-v3 early stop step={step} "
                    f"bestGain={recursive_best_gain:+.3f} "
                    f"badEvals={recursive_bad_evals}"
                )
                break

    if cfg.restore_best_at_end and best_recursive is not None:
        model.load_state_dict(best_recursive, strict=True)
        print(f"ARC recursive-v12 restored best checkpoint step={best_recursive_step}")

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
        f"ARC staged-predictor-v12 final exact={final_metrics['exact']:.3f} "
        f"ensemblePixel={final_metrics['adaptive_pixel_acc']:.3f} "
        f"fixedPixel={final_metrics['fixed_pixel_acc']:.3f} "
        f"oraclePixel={final_metrics['oracle_pixel_acc']:.3f} "
        f"directPixel={final_metrics['direct_pixel_acc']:.3f} "
        f"gain={final_metrics['recursive_gain']:+.3f} "
        f"hardDepth={final_metrics['avg_chosen_step']:.2f} "f"ensDepth={final_metrics.get('avg_ensemble_depth', float('nan')):.2f}"
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
