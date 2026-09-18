from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sefer.controllers.arc_reasoner import ARCReasoner
from sefer.evolution.evolve_transport import load_evolved_transport_cppn
from sefer.representation.arc_grid import arc_grid_loss
from sefer.evaluation.arc import evaluate_arc, evaluate_arc_direct


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _candidate_checkpoint_paths(requested: str, search_depth: int = 3):
    """Yield sensible locations for a v30 checkpoint.

    The previous ARC-v1 run silently fell back to a bootstrap core because the
    checkpoints were not copied into the ARC working directory.  This resolver
    searches the explicit path first, then the ARC project root, then nearby
    sibling projects such as yetirah_v30_modular / yetirah_v30_optimized.
    """
    p = Path(requested).expanduser()
    if p.is_absolute():
        yield p
        return

    basename = p.name
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    roots = [Path.cwd(), project_root]

    cur = project_root.parent
    for _ in range(max(int(search_depth), 0) + 1):
        roots.append(cur)
        cur = cur.parent

    seen = set()
    sibling_names = (
        "yetirah_v30_modular",
        "yetirah_v30_optimized",
        "yetirah_v30",
        "v30",
    )
    for root in roots:
        for cand in [root / p, root / basename]:
            key = str(cand.resolve(strict=False))
            if key not in seen:
                seen.add(key)
                yield cand
        for sibling in sibling_names:
            cand = root / sibling / basename
            key = str(cand.resolve(strict=False))
            if key not in seen:
                seen.add(key)
                yield cand

        # Limited-depth wildcard search catches custom sibling folder names
        # without recursively traversing an entire home directory or .venv.
        for pattern in (f"*/{basename}", f"*/*/{basename}"):
            try:
                for cand in root.glob(pattern):
                    key = str(cand.resolve(strict=False))
                    if key not in seen:
                        seen.add(key)
                        yield cand
            except OSError:
                pass


def _resolve_checkpoint(requested: str, search_depth: int = 3) -> Path | None:
    for cand in _candidate_checkpoint_paths(requested, search_depth=search_depth):
        if cand.is_file():
            return cand.resolve()
    return None


def initialize_arc_from_v30(model: ARCReasoner, cfg) -> Dict[str, bool]:
    """Install the v30 NEAT CPPN + frozen algebra prior.

    ARC has its own encoder/controller/decoder, so only ``core.*`` parameters are
    imported from the synthetic checkpoint.  If v30 initialization is requested,
    the default is now fail-fast rather than silently spending hours training a
    bootstrap operator system.
    """
    loaded = {"neat": False, "algebra": False}
    if not cfg.initialize_from_v30:
        print("ARC-v1: v30 initialization disabled; using bootstrap core by request")
        model.ensure_operator_bank()
        return loaded

    neat_path = _resolve_checkpoint(cfg.neat_winner_path, cfg.checkpoint_search_depth)
    algebra_path = _resolve_checkpoint(cfg.algebra_checkpoint_path, cfg.checkpoint_search_depth)

    if neat_path is None or algebra_path is None:
        missing = []
        if neat_path is None:
            missing.append(cfg.neat_winner_path)
        if algebra_path is None:
            missing.append(cfg.algebra_checkpoint_path)
        msg = (
            "ARC-v1 could not locate required v30 initialization file(s): "
            + ", ".join(missing)
            + ". Pass --neat-winner/--algebra-checkpoint with explicit paths, "
              "place the files in this project or a nearby v30 sibling folder, "
              "or use --no-v30-init intentionally."
        )
        if cfg.require_v30_init:
            raise FileNotFoundError(msg)
        print(msg)

    if neat_path is not None:
        print(f"ARC-v1 loading evolved transport CPPN: {neat_path}")
        load_evolved_transport_cppn(model, str(neat_path))
        cfg.neat_winner_path = str(neat_path)
        loaded["neat"] = True

    if algebra_path is not None:
        print(f"ARC-v1 loading frozen algebra: {algebra_path}")
        payload = torch.load(algebra_path, map_location=model.core.coords.device)
        state = payload.get("model_state_dict", payload)
        core_state = {k[len("core."):]: v for k, v in state.items() if k.startswith("core.")}
        _, unexpected = model.core.load_state_dict(core_state, strict=False)
        if unexpected:
            print(f"ARC-v1 core load unexpected keys: {unexpected[:5]}")
        loaded["algebra"] = len(core_state) > 0
        cfg.algebra_checkpoint_path = str(algebra_path)

    model.ensure_operator_bank()
    print(
        f"ARC-v1 initialization: neat={loaded['neat']} algebra={loaded['algebra']} "
        f"fast_rank={cfg.fast_operator_rank} static_rank={cfg.fast_operator_static_rank}"
    )
    return loaded


def _decode_loss(model, H, target, target_shape):
    c, h, w = model.decode_grid(H)
    color_loss, stats = arc_grid_loss(
        c, h, w, target, target_shape,
        foreground_boost=model.cfg.foreground_boost,
        balance_mix=model.cfg.color_balance_mix,
    )
    return color_loss, stats["shape_loss"], stats


def _set_requires_grad(module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


def _program_stage(cfg, step: int) -> Tuple[int, int, int]:
    """Return (stage_index, active_operator_count, max_depth)."""
    progress = (step - 1) / max(cfg.program_steps - 1, 1)
    b0, b1, b2 = cfg.program_stage_fractions
    if progress < b0:
        idx = 0
    elif progress < b1:
        idx = 1
    elif progress < b2:
        idx = 2
    else:
        idx = 3
    active = min(int(cfg.program_stage_active_ops[idx]), int(cfg.operator_count))
    depth = min(int(cfg.program_stage_depths[idx]), int(cfg.max_program_steps))
    return idx, max(active, 1), max(depth, 1)


@torch.no_grad()
def _evaluate_operator_discovery(model, data, cfg, device):
    if data is None:
        return None
    model.eval()
    vals = {"best_lat": 0.0, "base_lat": 0.0, "improve": 0.0, "pixel": 0.0}
    batches = max(int(cfg.operator_discovery_val_batches), 1)
    bs = min(int(cfg.batch_size), max(int(cfg.eval_tasks), 1))
    for _ in range(batches):
        batch = data.sample_batch(bs, device)
        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        Hq = model.encode_grid(batch.query_x, batch.query_shape)
        Ht = model.encode_grid(batch.target_y, batch.target_shape)
        all_states = model.operator_bank.apply_all(Hq, rule)
        per_op_err = (all_states - Ht[:, None]).pow(2).mean(dim=(2, 3))
        best_err, best_idx = per_op_err.min(dim=1)
        base_err = (Hq - Ht).pow(2).mean(dim=(1, 2))
        b = torch.arange(Hq.shape[0], device=device)
        best_state = all_states[b, best_idx]
        _, _, stats = _decode_loss(model, best_state, batch.target_y, batch.target_shape)
        vals["best_lat"] += float(best_err.mean())
        vals["base_lat"] += float(base_err.mean())
        vals["improve"] += float((best_err < base_err).float().mean())
        vals["pixel"] += float(stats["pixel_acc"])
    for k in vals:
        vals[k] /= batches
    vals["score"] = vals["pixel"] + 0.25 * vals["improve"] - 0.10 * vals["best_lat"]
    return vals


@torch.no_grad()
def _rank_discovered_operators(model, data, cfg, device):
    model.eval()
    usage = torch.zeros(cfg.operator_count, device=device)
    batches = max(int(cfg.operator_discovery_val_batches), 1)
    bs = min(int(cfg.batch_size), max(int(cfg.eval_tasks), 1))
    for _ in range(batches):
        batch = data.sample_batch(bs, device)
        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        Hq = model.encode_grid(batch.query_x, batch.query_shape)
        Ht = model.encode_grid(batch.target_y, batch.target_shape)
        all_states = model.operator_bank.apply_all(Hq, rule)
        err = (all_states - Ht[:, None]).pow(2).mean(dim=(2, 3))
        best = err.argmin(dim=1)
        usage += torch.bincount(best, minlength=cfg.operator_count).float()
    return torch.argsort(usage, descending=True)


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
        f"opDiscovery={cfg.operator_discovery_steps} program={cfg.program_steps}"
    )
    print(
        "ARC-v1 program stages="
        + ", ".join(
            f"ops{ops}/d{depth}"
            for ops, depth in zip(cfg.program_stage_active_ops, cfg.program_stage_depths)
        )
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
        color_loss, stats = arc_grid_loss(
            color_logits, h_logits, w_logits, grid, shape,
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
                f"cellCE={color_loss.item():.4f} pixelAcc={stats['pixel_acc'].item():.3f} "
                f"fgAcc={stats['foreground_acc'].item():.3f} bgAcc={stats['background_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f}"
            )

    # ------------------------------------------------------------------
    # Phase B: demonstrations -> rule -> predicted latent goal.
    # ------------------------------------------------------------------
    direct_params = (
        list(model.grid_encoder.parameters())
        + list(model.grid_decoder.parameters())
        + list(model.demo_slot_encoder.parameters())
        + list(model.demo_slot_score.parameters())
        + list(model.demo_pair_encoder.parameters())
        + list(model.rule_encoder.parameters())
        + list(model.direct_rule_to_slots.parameters())
        + list(model.direct_norm.parameters())
    )
    direct_opt = torch.optim.AdamW(direct_params, lr=cfg.direct_lr, weight_decay=cfg.weight_decay)
    best_direct_score = -1.0
    best_direct_state = None
    for step in range(1, cfg.direct_steps + 1):
        model.train()
        batch = train_data.sample_batch(cfg.batch_size, device)
        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        Hq = model.encode_grid(batch.query_x, batch.query_shape)
        Hg = model.predict_goal(Hq, rule)
        color_loss, shape_loss, stats = _decode_loss(model, Hg, batch.target_y, batch.target_shape)

        # Keep the codec useful while the demo-conditioned goal predictor forms.
        H_identity = model.encode_grid(batch.query_x, batch.query_shape)
        id_color, id_shape, _ = _decode_loss(model, H_identity, batch.query_x, batch.query_shape)
        loss = color_loss + cfg.shape_weight * shape_loss + 0.20 * id_color + 0.05 * id_shape
        direct_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(direct_params, cfg.grad_clip)
        direct_opt.step()
        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.direct_steps:
            print(
                f"arc direct step={step:04d} loss={loss.item():.4f} "
                f"cellCE={color_loss.item():.4f} pixelAcc={stats['pixel_acc'].item():.3f} "
                f"fgAcc={stats['foreground_acc'].item():.3f} bgAcc={stats['background_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f}"
            )
        if val_data is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            dm = evaluate_arc_direct(model, val_data, limit=cfg.eval_tasks, device=device)
            dscore = dm["exact"] + 0.10 * dm["pixel_acc"]
            print(
                f"arc direct val step={step:04d} exact={dm['exact']:.3f} "
                f"pixel={dm['pixel_acc']:.3f} shape={dm['shape_acc']:.3f}"
            )
            if dscore > best_direct_score:
                best_direct_score = dscore
                best_direct_state = copy.deepcopy(model.state_dict())

    if best_direct_state is not None:
        model.load_state_dict(best_direct_state, strict=True)
        print(f"ARC-v1 restored best direct model score={best_direct_score:.4f}")

    # ------------------------------------------------------------------
    # Phase C: ARC operator discovery.  Keep the representation fixed and let
    # the low-rank residual operators specialize on real ARC transitions before
    # asking the controller to compose them.
    # ------------------------------------------------------------------
    _set_requires_grad(model.grid_encoder, False)
    _set_requires_grad(model.grid_decoder, False)
    _set_requires_grad(model.demo_slot_encoder, False)
    _set_requires_grad(model.demo_slot_score, False)
    _set_requires_grad(model.demo_pair_encoder, False)
    _set_requires_grad(model.rule_encoder, False)
    _set_requires_grad(model.direct_rule_to_slots, False)
    _set_requires_grad(model.direct_norm, False)
    for p in model.core.parameters():
        p.requires_grad_(False)

    op_discovery_params = list(model.operator_bank.parameters())
    op_discovery_opt = torch.optim.AdamW(
        op_discovery_params, lr=cfg.operator_discovery_lr, weight_decay=cfg.weight_decay
    )
    discovery_usage_accum = torch.zeros(cfg.operator_count, device=device)
    best_operator_state = None
    best_operator_score = -float("inf")
    best_operator_step = None
    for step in range(1, cfg.operator_discovery_steps + 1):
        model.train()
        batch = train_data.sample_batch(cfg.batch_size, device)
        with torch.no_grad():
            rule = model.encode_rule(
                batch.demos_x, batch.demos_y,
                batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
            )
            Hq = model.encode_grid(batch.query_x, batch.query_shape)
            Ht = model.encode_grid(batch.target_y, batch.target_shape)
        all_states = model.operator_bank.apply_all(Hq, rule)
        per_op_err = (all_states - Ht[:, None]).pow(2).mean(dim=(2, 3))
        progress = (step - 1) / max(cfg.operator_discovery_steps - 1, 1)
        tau = (
            cfg.operator_discovery_temp_start * (1.0 - progress)
            + cfg.operator_discovery_temp_end * progress
        )
        assignment = torch.softmax(-per_op_err / max(tau, 1e-4), dim=1)
        discovery_usage_accum += assignment.detach().mean(dim=0)
        soft_latent = (assignment * per_op_err).sum(dim=1).mean()
        Hmix = (all_states * assignment[:, :, None, None]).sum(dim=1)
        mix_color, mix_shape, mix_stats = _decode_loss(
            model, Hmix, batch.target_y, batch.target_shape
        )
        usage = assignment.mean(dim=0)
        usage = usage / usage.sum().clamp_min(1e-8)
        usage_loss = (usage * torch.log(usage.clamp_min(1e-8))).sum() / math.log(cfg.operator_count)
        reg = model.operator_bank.regularization()
        trust = model.operator_bank.trust_region(rule)
        trust_penalty = (
            cfg.operator_discovery_trust_weight * trust["transport_kl"]
            + cfg.fast_gate_penalty_weight * trust["gate_mean"]
        )
        discovery_loss = (
            soft_latent
            + cfg.operator_discovery_grid_weight * (mix_color + cfg.shape_weight * mix_shape)
            + cfg.operator_discovery_usage_weight * usage_loss
            + cfg.operator_discovery_reg_weight * reg
            + trust_penalty
        )
        op_discovery_opt.zero_grad(set_to_none=True)
        discovery_loss.backward()
        torch.nn.utils.clip_grad_norm_(op_discovery_params, cfg.grad_clip)
        op_discovery_opt.step()
        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.operator_discovery_steps:
            with torch.no_grad():
                best_err = per_op_err.min(dim=1).values.mean()
                current_err = (Hq - Ht).pow(2).mean(dim=(1, 2)).mean()
                improve_frac = (per_op_err.min(dim=1).values < (Hq - Ht).pow(2).mean(dim=(1, 2))).float().mean()
                hard_use = int(torch.unique(per_op_err.argmin(dim=1)).numel())
            fast_diag = model.operator_bank.diagnostics(rule)
            print(
                f"arc opdiscover step={step:04d} loss={discovery_loss.item():.4f} "
                f"bestLat={best_err.item():.4f} baseLat={current_err.item():.4f} "
                f"improve={improve_frac.item():.3f} mixPix={mix_stats['pixel_acc'].item():.3f} "
                f"fgAcc={mix_stats['foreground_acc'].item():.3f} usedOps={hard_use}/{cfg.operator_count} "
                f"fastGate={fast_diag['fast_gate'].item():.3f} "
                f"fastDelta={fast_diag['fast_delta_rms'].item():.4f} "
                f"fastKL={fast_diag['transport_kl'].item():.4f} tau={tau:.3f}"
            )

        if val_data is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            opm = _evaluate_operator_discovery(model, val_data, cfg, device)
            print(
                f"arc opdiscover val step={step:04d} bestLat={opm['best_lat']:.4f} "
                f"baseLat={opm['base_lat']:.4f} improve={opm['improve']:.3f} "
                f"pixel={opm['pixel']:.3f}"
            )
            if opm["score"] > best_operator_score:
                best_operator_score = opm["score"]
                best_operator_state = copy.deepcopy(model.operator_bank.state_dict())
                best_operator_step = step

    if best_operator_state is not None:
        model.operator_bank.load_state_dict(best_operator_state, strict=True)
        print(f"ARC-v1 restored best operator-discovery checkpoint from step={best_operator_step}")

    operator_order = _rank_discovered_operators(
        model, val_data if val_data is not None else train_data, cfg, device
    )
    print(
        "ARC-v1 discovered operator priority="
        + str(operator_order.detach().cpu().tolist())
    )

    # ------------------------------------------------------------------
    # Phase D: freeze task inference and learn progressively deeper programs
    # over the discovered ARC operator vocabulary.
    # ------------------------------------------------------------------
    # Task inference is intentionally frozen here.  The previous run let the
    # predicted goal drift while the controller was learning, which made the
    # planning target non-stationary.
    base_params = (
        list(model.arc_program_controller.parameters())
        + list(model.arc_program_value.parameters())
        + list(model.program_blend.parameters())
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
    best_program_state = None
    best_program_step = None
    last_stage = None
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

        # Crucially, this goal is available at inference: demos + query only.
        Hgoal = model.predict_goal(Hq, rule)
        goal_consistency_loss = (Hgoal - Ht).pow(2).mean()

        progress = (step - 1) / max(cfg.program_steps - 1, 1)
        temp = cfg.gumbel_temp_start * (1.0 - progress) + cfg.gumbel_temp_end * progress
        stage_idx, active_ops, stage_depth = _program_stage(cfg, step)
        if stage_idx != last_stage:
            print(
                f"ARC-v1 entering program stage {stage_idx + 1}: "
                f"activeOps={active_ops} maxDepth={stage_depth} temp={temp:.3f}"
            )
            last_stage = stage_idx

        active_indices = operator_order[:active_ops]
        Hp, action_w, value_preds, trace = model.rollout_program(
            Hq,
            rule,
            goal_H=Hgoal,
            max_steps=stage_depth,
            active_operator_count=active_ops,
            active_operator_indices=active_indices,
            temperature=temp,
            hard=True,
            greedy=False,
            return_trace=True,
        )
        latent_loss = (Hp - Ht).pow(2).mean()
        color_loss, shape_loss, stats = _decode_loss(model, Hp, batch.target_y, batch.target_shape)

        # Preserve/improve the goal predictor instead of discarding it when
        # program training starts.
        direct_color, direct_shape, direct_stats = _decode_loss(
            model, Hgoal, batch.target_y, batch.target_shape
        )
        direct_aux = direct_color + cfg.shape_weight * direct_shape

        # Training-only target-aware policy teacher. At each visited state,
        # choose the active operator that most improves true target-latent error;
        # choose STOP only when no operator clears a small improvement margin.
        # This directly addresses the v1.1 deep-stage STOP collapse without
        # leaking the target at inference (the teacher is absent there).
        oracle_losses = []
        oracle_stop_fracs = []
        for state_t, logits_t, halted_t in zip(
            trace["states_before"], trace["logits"], trace["halted_before"]
        ):
            with torch.no_grad():
                op_states = model.operator_bank.apply_all(state_t.detach(), rule.detach())[:, active_indices]
                current_err = (state_t.detach() - Ht).pow(2).mean(dim=(1, 2))
                op_err = (op_states - Ht[:, None]).pow(2).mean(dim=(2, 3))
                best_err, best_local_idx = op_err.min(dim=1)
                best_idx = active_indices[best_local_idx]
                improvement = current_err - best_err
                teacher = torch.where(
                    improvement > cfg.oracle_improvement_margin,
                    best_idx,
                    torch.full_like(best_idx, cfg.operator_count),
                )
                was_halted = halted_t.squeeze(-1).squeeze(-1) > 0.5
                teacher = torch.where(
                    was_halted,
                    torch.full_like(teacher, cfg.operator_count),
                    teacher,
                )
                oracle_stop_fracs.append((teacher == cfg.operator_count).float().mean())
            oracle_losses.append(F.cross_entropy(logits_t, teacher))
        oracle_policy_loss = torch.stack(oracle_losses).mean()
        oracle_stop_frac = torch.stack(oracle_stop_fracs).mean()

        non_stop = action_w[:, :, : cfg.operator_count].sum(dim=-1)
        length_loss = non_stop.mean()

        # Balance only the currently available vocabulary. This encourages
        # specialization without forcing probability onto masked operators.
        usage = action_w[:, :, active_indices].mean(dim=(0, 1))
        usage = usage / usage.sum().clamp_min(1e-8)
        if active_ops > 1:
            usage_balance_loss = (
                usage * torch.log(usage.clamp_min(1e-8))
            ).sum() / math.log(active_ops)
        else:
            usage_balance_loss = torch.zeros((), device=device)

        per_sample_latent = (Hp.detach() - Ht).pow(2).mean(dim=(1, 2))
        value_target = torch.exp(-4.0 * per_sample_latent).clamp(0.0, 1.0)
        value_loss = torch.stack([F.mse_loss(v, value_target) for v in value_preds]).mean()

        op_reg = model.operator_bank.regularization()
        fast_trust = model.operator_bank.trust_region(rule)
        fast_trust_loss = (
            cfg.fast_transport_kl_weight * fast_trust["transport_kl"]
            + cfg.fast_gate_penalty_weight * fast_trust["gate_mean"]
        )
        loss = (
            cfg.latent_weight * latent_loss
            + cfg.grid_weight * color_loss
            + cfg.shape_weight * shape_loss
            + cfg.length_weight * length_loss
            + cfg.operator_reg_weight * op_reg
            + cfg.usage_balance_weight * usage_balance_loss
            + cfg.value_weight * value_loss
            + cfg.oracle_policy_weight * oracle_policy_loss
            + fast_trust_loss
        )

        program_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params + op_params, cfg.grad_clip)
        program_opt.step()

        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.program_steps:
            with torch.no_grad():
                chosen = action_w.argmax(dim=-1)
                stop_frac = (chosen == cfg.operator_count).float().mean()
                op_unique = (
                    int(torch.unique(chosen[chosen < cfg.operator_count]).numel())
                    if (chosen < cfg.operator_count).any() else 0
                )
                program_len = (chosen != cfg.operator_count).float().sum(dim=1).mean()
            fast_diag = model.operator_bank.diagnostics(rule)
            blend_mean = trace["blend_gate"].mean()
            print(
                f"arc program step={step:04d} loss={loss.item():.4f} "
                f"latent={latent_loss.item():.4f} goalLat={goal_consistency_loss.item():.4f} "
                f"cellCE={color_loss.item():.4f} pixelAcc={stats['pixel_acc'].item():.3f} "
                f"fgAcc={stats['foreground_acc'].item():.3f} "
                f"directPix={direct_stats['pixel_acc'].item():.3f} "
                f"shapeAcc={stats['shape_acc'].item():.3f} stop={stop_frac.item():.3f} "
                f"oracleCE={oracle_policy_loss.item():.3f} oracleStop={oracle_stop_frac.item():.3f} "
                f"progLen={program_len.item():.2f} usedOps={op_unique}/{active_ops} "
                f"blend={blend_mean.item():.3f} fastGate={fast_diag['fast_gate'].item():.3f} "
                f"fastKL={fast_diag['transport_kl'].item():.4f} "
                f"depth={stage_depth} temp={temp:.3f}"
            )

        if val_data is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            metrics = evaluate_arc(model, val_data, limit=cfg.eval_tasks, device=device)
            score = metrics["exact"] + 0.10 * metrics["pixel_acc"]
            print(
                f"arc val step={step:04d} exact={metrics['exact']:.3f} "
                f"pixel={metrics['pixel_acc']:.3f} shape={metrics['shape_acc']:.3f} "
                f"directPixel={metrics['direct_pixel_acc']:.3f} "
                f"directShape={metrics['direct_shape_acc']:.3f}"
            )
            if score > best_score:
                best_score = score
                best_program_state = copy.deepcopy(model.state_dict())
                best_program_step = step
                torch.save(
                    {"model_state_dict": model.state_dict(), "cfg": vars(cfg), "step": step, "metrics": metrics},
                    cfg.arc_best_checkpoint_path,
                )
                print(f"saved ARC-v1 best checkpoint: {cfg.arc_best_checkpoint_path}")

    if val_data is not None and cfg.restore_best_at_end and best_program_state is not None:
        model.load_state_dict(best_program_state, strict=True)
        print(f"ARC-v1 restored best program checkpoint from step={best_program_step}")

    final_metrics = evaluate_arc(model, val_data or train_data, limit=cfg.eval_tasks, device=device)
    torch.save(
        {"model_state_dict": model.state_dict(), "cfg": vars(cfg), "metrics": final_metrics},
        cfg.arc_checkpoint_path,
    )
    print(f"saved ARC-v1 checkpoint: {cfg.arc_checkpoint_path}")
    print(
        f"ARC-v1 final exact={final_metrics['exact']:.3f} pixel={final_metrics['pixel_acc']:.3f} "
        f"shape={final_metrics['shape_acc']:.3f} directPixel={final_metrics['direct_pixel_acc']:.3f}"
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
