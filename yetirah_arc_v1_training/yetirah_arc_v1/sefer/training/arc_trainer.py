from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from yetirah_arc_v1_training.yetirah_arc_v1.sefer.controllers.arc_reasoner import ARCReasoner
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.evolution.evolve_transport import load_evolved_transport_cppn
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.representation.arc_grid import arc_grid_loss
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.evaluation.arc import evaluate_arc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_arc_from_v30(model: ARCReasoner, cfg) -> Dict[str, bool]:
    """Install the v30 NEAT CPPN + frozen algebra when available.

    ARC has its own encoder/controller/decoder, so only `core.*` parameters are
    imported from the synthetic checkpoint. The original v30 files are never
    modified.
    """
    loaded = {"neat": False, "algebra": False}
    if not cfg.initialize_from_v30:
        model.ensure_operator_bank()
        return loaded

    if os.path.exists(cfg.neat_winner_path):
        load_evolved_transport_cppn(model, cfg.neat_winner_path)
        loaded["neat"] = True
    else:
        print(f"ARC-v1: NEAT winner not found at {cfg.neat_winner_path}; using bootstrap core")

    if os.path.exists(cfg.algebra_checkpoint_path):
        payload = torch.load(cfg.algebra_checkpoint_path, map_location=model.core.coords.device)
        state = payload.get("model_state_dict", payload)
        core_state = {k[len("core."):]: v for k, v in state.items() if k.startswith("core.")}
        missing, unexpected = model.core.load_state_dict(core_state, strict=False)
        # Missing non-persistent caches/evolved graph are expected.
        if unexpected:
            print(f"ARC-v1 core load unexpected keys: {unexpected[:5]}")
        loaded["algebra"] = len(core_state) > 0
    else:
        print(f"ARC-v1: algebra checkpoint not found at {cfg.algebra_checkpoint_path}; using current core weights")

    model.ensure_operator_bank()
    print(
        f"ARC-v1 initialization: neat={loaded['neat']} algebra={loaded['algebra']} "
        f"adaptive_rank={cfg.operator_residual_rank}"
    )
    return loaded


def _decode_loss(model, H, target, target_shape):
    c, h, w = model.decode_grid(H)
    color_loss, stats = arc_grid_loss(c, h, w, target, target_shape)
    return color_loss, stats["shape_loss"], stats


def _set_requires_grad(module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


def train_arc_v1(cfg, train_data, val_data=None):
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    device = torch.device(cfg.device)
    model = ARCReasoner(cfg).to(device)
    initialize_arc_from_v30(model, cfg)

    print(f"ARC-v1 device={device}")
    print(f"ARC-v1 tasks={train_data.task_count} batch={cfg.batch_size}")
    print(f"ARC-v1 params={sum(p.numel() for p in model.parameters()):,}")
    print(
        f"ARC-v1 curriculum codec={cfg.codec_steps} direct={cfg.direct_steps} "
        f"program={cfg.program_steps}"
    )

    # ------------------------------------------------------------------
    # Phase A: grid <-> substrate codec.
    # ------------------------------------------------------------------
    codec_params = list(model.grid_encoder.parameters()) + list(model.grid_decoder.parameters())
    codec_opt = torch.optim.AdamW(codec_params, lr=cfg.codec_lr, weight_decay=cfg.weight_decay)
    for step in range(1, cfg.codec_steps + 1):
        model.train()
        grid, shape = train_data.sample_grid_batch(cfg.batch_size, device)
        H = model.encode_grid(grid, shape)
        color_logits, h_logits, w_logits = model.decode_grid(H)
        color_loss, stats = arc_grid_loss(color_logits, h_logits, w_logits, grid, shape)
        loss = color_loss + cfg.shape_weight * stats["shape_loss"]
        codec_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec_params, cfg.grad_clip)
        codec_opt.step()
        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.codec_steps:
            print(
                f"arc codec step={step:04d} loss={loss.item():.4f} "
                f"cellCE={color_loss.item():.4f} pixelAcc={stats['pixel_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f}"
            )

    # ------------------------------------------------------------------
    # Phase B: demonstrations -> rule; task-conditioned direct baseline.
    # This gives the rule embedding useful semantics before operator discovery.
    # ------------------------------------------------------------------
    direct_params = (
        list(model.grid_encoder.parameters())
        + list(model.grid_decoder.parameters())
        + list(model.demo_pair_encoder.parameters())
        + list(model.rule_encoder.parameters())
        + list(model.direct_rule_to_slots.parameters())
        + list(model.direct_norm.parameters())
    )
    direct_opt = torch.optim.AdamW(direct_params, lr=cfg.direct_lr, weight_decay=cfg.weight_decay)
    for step in range(1, cfg.direct_steps + 1):
        model.train()
        batch = train_data.sample_batch(cfg.batch_size, device)
        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        Hq = model.encode_grid(batch.query_x, batch.query_shape)
        Hd = model.direct_transform(Hq, rule)
        color_loss, shape_loss, stats = _decode_loss(model, Hd, batch.target_y, batch.target_shape)
        # Keep the codec honest while it becomes task-sensitive.
        H_identity = model.encode_grid(batch.query_x, batch.query_shape)
        id_color, id_shape, _ = _decode_loss(model, H_identity, batch.query_x, batch.query_shape)
        loss = (
            color_loss + cfg.shape_weight * shape_loss
            + 0.20 * id_color + 0.05 * id_shape
        )
        direct_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(direct_params, cfg.grad_clip)
        direct_opt.step()
        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.direct_steps:
            print(
                f"arc direct step={step:04d} loss={loss.item():.4f} "
                f"cellCE={color_loss.item():.4f} pixelAcc={stats['pixel_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f}"
            )

    # ------------------------------------------------------------------
    # Phase C: freeze the grid coordinate system and learn a short operator
    # program.  The true target is used only as a latent/output teacher.
    # ------------------------------------------------------------------
    _set_requires_grad(model.grid_encoder, False)
    _set_requires_grad(model.grid_decoder, False)
    for p in model.core.parameters():
        p.requires_grad_(False)

    base_params = (
        list(model.demo_pair_encoder.parameters())
        + list(model.rule_encoder.parameters())
        + list(model.direct_rule_to_slots.parameters())
        + list(model.direct_norm.parameters())
        + list(model.arc_program_controller.parameters())
        + list(model.arc_program_value.parameters())
    )
    op_params = list(model.operator_bank.parameters())
    program_opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.program_lr},
            {"params": op_params, "lr": cfg.operator_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    best_score = -1.0
    for step in range(1, cfg.program_steps + 1):
        model.train()
        batch = train_data.sample_batch(cfg.batch_size, device)
        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        with torch.no_grad():
            Hq = model.encode_grid(batch.query_x, batch.query_shape)
            Ht = model.encode_grid(batch.target_y, batch.target_shape)

        progress = (step - 1) / max(cfg.program_steps - 1, 1)
        temp = cfg.gumbel_temp_start * (1.0 - progress) + cfg.gumbel_temp_end * progress
        Hp, action_w, value_preds = model.rollout_program(
            Hq, rule, max_steps=cfg.max_program_steps,
            temperature=temp, hard=True, greedy=False,
        )
        latent_loss = (Hp - Ht).pow(2).mean()
        color_loss, shape_loss, stats = _decode_loss(model, Hp, batch.target_y, batch.target_shape)

        # Keep the demonstration representation useful with a cheap direct path.
        Hd = model.direct_transform(Hq, rule)
        direct_color, direct_shape, _ = _decode_loss(model, Hd, batch.target_y, batch.target_shape)
        direct_aux = direct_color + cfg.shape_weight * direct_shape

        # Short-program prior. STOP is absorbing in rollout_program.
        non_stop = action_w[:, :, : cfg.operator_count].sum(dim=-1)
        length_loss = non_stop.mean()

        # Prevent all ARC tasks from collapsing onto one latent operator.
        usage = action_w[:, :, : cfg.operator_count].mean(dim=(0, 1))
        usage = usage / usage.sum().clamp_min(1e-8)
        usage_balance_loss = (usage * torch.log(usage.clamp_min(1e-8))).sum() / math.log(cfg.operator_count)

        per_sample_latent = (Hp.detach() - Ht).pow(2).mean(dim=(1, 2))
        value_target = torch.exp(-4.0 * per_sample_latent).clamp(0.0, 1.0)
        value_loss = torch.stack([
            F.mse_loss(v, value_target) for v in value_preds
        ]).mean()

        op_reg = model.operator_bank.regularization()
        loss = (
            cfg.latent_weight * latent_loss
            + cfg.grid_weight * color_loss
            + cfg.shape_weight * shape_loss
            + cfg.direct_aux_weight * direct_aux
            + cfg.length_weight * length_loss
            + cfg.operator_reg_weight * op_reg
            + cfg.usage_balance_weight * usage_balance_loss
            + cfg.value_weight * value_loss
        )

        program_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params + op_params, cfg.grad_clip)
        program_opt.step()

        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.program_steps:
            with torch.no_grad():
                chosen = action_w.argmax(dim=-1)
                stop_frac = (chosen == cfg.operator_count).float().mean()
                op_unique = int(torch.unique(chosen[chosen < cfg.operator_count]).numel()) if (chosen < cfg.operator_count).any() else 0
            print(
                f"arc program step={step:04d} loss={loss.item():.4f} "
                f"latent={latent_loss.item():.4f} cellCE={color_loss.item():.4f} "
                f"pixelAcc={stats['pixel_acc'].item():.3f} shapeAcc={stats['shape_acc'].item():.3f} "
                f"stop={stop_frac.item():.3f} usedOps={op_unique} temp={temp:.3f}"
            )

        if val_data is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            metrics = evaluate_arc(model, val_data, limit=cfg.eval_tasks, device=device)
            score = metrics["exact"] + 0.10 * metrics["pixel_acc"]
            print(
                f"arc val step={step:04d} exact={metrics['exact']:.3f} "
                f"pixel={metrics['pixel_acc']:.3f} shape={metrics['shape_acc']:.3f}"
            )
            if score > best_score:
                best_score = score
                torch.save(
                    {"model_state_dict": model.state_dict(), "cfg": vars(cfg), "step": step, "metrics": metrics},
                    cfg.arc_best_checkpoint_path,
                )
                print(f"saved ARC-v1 best checkpoint: {cfg.arc_best_checkpoint_path}")

    final_metrics = evaluate_arc(model, val_data or train_data, limit=cfg.eval_tasks, device=device)
    torch.save(
        {"model_state_dict": model.state_dict(), "cfg": vars(cfg), "metrics": final_metrics},
        cfg.arc_checkpoint_path,
    )
    print(f"saved ARC-v1 checkpoint: {cfg.arc_checkpoint_path}")
    print(
        f"ARC-v1 final exact={final_metrics['exact']:.3f} pixel={final_metrics['pixel_acc']:.3f} "
        f"shape={final_metrics['shape_acc']:.3f}"
    )
    return model, final_metrics


def load_arc_v1_checkpoint(cfg, path: str | None = None, device: str | torch.device | None = None):
    """Rebuild ARCReasoner, recreate its adaptive operator bank, then load a checkpoint."""
    device = torch.device(device or cfg.device)
    model = ARCReasoner(cfg).to(device)
    initialize_arc_from_v30(model, cfg)
    payload = torch.load(path or cfg.arc_checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload
