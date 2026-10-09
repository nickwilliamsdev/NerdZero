from __future__ import annotations

# Kept for compatibility with the current launcher.
ARC_EVAL_PATCH_ID = "v1.7-demo-consistency-search"
ARC_EVAL_ARCH = "recursive-deltanet-v1-demo-controller-v9"

from typing import Any, Dict
import numpy as np
import torch

from sefer.tasks.arc_dataset import _pair_arrays, pad_grid


def _episode_tensors(demos, query_x, max_demos, max_size, device):
    dx = np.zeros((1, max_demos, max_size, max_size), dtype=np.int64)
    dy = np.zeros_like(dx)
    dxs = np.ones((1, max_demos, 2), dtype=np.int64)
    dys = np.ones((1, max_demos, 2), dtype=np.int64)
    dm = np.zeros((1, max_demos), dtype=np.bool_)
    for d, pair in enumerate(list(demos)[:max_demos]):
        x, y = _pair_arrays(pair)
        if y is None:
            continue
        dx[0, d], dxs[0, d] = pad_grid(x, max_size)
        dy[0, d], dys[0, d] = pad_grid(y, max_size)
        dm[0, d] = True
    q, qs = pad_grid(query_x, max_size)
    return (
        torch.as_tensor(dx, device=device, dtype=torch.long),
        torch.as_tensor(dy, device=device, dtype=torch.long),
        torch.as_tensor(dxs, device=device, dtype=torch.long),
        torch.as_tensor(dys, device=device, dtype=torch.long),
        torch.as_tensor(dm, device=device, dtype=torch.bool),
        torch.as_tensor(q[None], device=device, dtype=torch.long),
        torch.as_tensor([qs], device=device, dtype=torch.long),
    )


def _decode_grid(model, H):
    color_logits, h_logits, w_logits = model.decode_grid(H)
    h = int(h_logits.argmax(dim=-1).item()) + 1
    w = int(w_logits.argmax(dim=-1).item()) + 1
    return color_logits.argmax(dim=1)[0, :h, :w].cpu().numpy().astype(np.int64)


def _grid_scores(pred: np.ndarray, target: np.ndarray):
    target = np.asarray(target, dtype=np.int64)
    shape_ok = pred.shape == target.shape
    h = min(pred.shape[0], target.shape[0])
    w = min(pred.shape[1], target.shape[1])
    overlap = float(np.mean(pred[:h, :w] == target[:h, :w])) if h and w else 0.0
    target_cells = max(int(target.size), 1)
    pixel_score = overlap * ((h * w) / target_cells)
    exact = int(shape_ok and np.array_equal(pred, target))
    return exact, pixel_score, float(shape_ok)


@torch.no_grad()
def _select_depth_from_demos(model, demos, device):
    """Choose depth 0..T from leave-one-demo-out reconstruction only."""
    T = int(model.cfg.recursive_steps)
    if len(demos) < 2:
        return T, [0.0 for _ in range(T + 1)]

    scores = [0.0 for _ in range(T + 1)]
    used = 0

    for held_idx in range(len(demos)):
        context = [d for i, d in enumerate(demos) if i != held_idx]
        held_in, held_out = demos[held_idx]

        dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
            context, held_in,
            model.cfg.max_demos, model.cfg.max_grid_size, device
        )
        rule = model.encode_rule(dx, dy, dxs, dys, dm)
        Hq = model.encode_grid(q, qs)
        rule = model.condition_rule_on_query(rule, Hq)
        _, info = model.recursive_reason(
            Hq, rule, steps=T, return_trace=True
        )

        for depth, state in enumerate(info["states"]):
            pred = _decode_grid(model, state)
            exact, pixel, shape = _grid_scores(pred, held_out)
            scores[depth] += (
                pixel
                + model.cfg.demo_depth_exact_weight * exact
                + model.cfg.demo_depth_shape_weight * shape
            )
        used += 1

    scores = [s / max(used, 1) for s in scores]
    adjusted = [
        s - model.cfg.demo_depth_prefer_shallower * d
        for d, s in enumerate(scores)
    ]
    chosen = max(range(len(adjusted)), key=lambda i: adjusted[i])
    return chosen, scores


def _demo_scores_to_cost_tensor(scores, device, dtype):
    return -torch.tensor(
        scores, device=device, dtype=dtype
    ).unsqueeze(0)


@torch.no_grad()
def predict_arc(model, demos, query_x, device=None,
                return_direct: bool = False, return_details: bool = False):
    device = device or next(model.parameters()).device
    cfg = model.cfg
    dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
        demos, query_x, cfg.max_demos, cfg.max_grid_size, device
    )

    rule = model.encode_rule(dx, dy, dxs, dys, dm)
    Hq = model.encode_grid(q, qs)
    rule = model.condition_rule_on_query(rule, Hq)
    _, info = model.recursive_reason(
        Hq, rule, steps=cfg.recursive_steps, return_trace=True
    )

    calibrated_depth, demo_depth_scores = _select_depth_from_demos(
        model, demos, device
    )
    demo_profile = _demo_scores_to_cost_tensor(
        demo_depth_scores, device, rule.dtype
    )
    adaptive_state, chosen, controller_logits = model.select_demo_conditioned_depth(
        rule, info, demo_profile
    )
    chosen_depth = int(chosen[0].item())
    adaptive_grid = _decode_grid(model, adaptive_state)
    direct_grid = _decode_grid(model, info["states"][0])
    fixed_grid = _decode_grid(model, info["states"][-1])

    details = {
        "mode": "adaptive_recursive_deltanet",
        "chosen_step": int(chosen[0].item()),
        "demo_depth_scores": demo_depth_scores,
        "calibrated_depth": int(calibrated_depth),
        "controller_logits": controller_logits[0].detach().cpu().tolist(),
        "halt_probs": [
            float(torch.sigmoid(x).item()) for x in info["halt_logits"]
        ],
        "fixed_grid": fixed_grid,
        "direct_grid": direct_grid,
    }

    if return_details:
        return adaptive_grid, [], direct_grid, details
    if return_direct:
        return adaptive_grid, [], direct_grid
    return adaptive_grid, []


@torch.no_grad()
def evaluate_arc_direct(model, dataset, limit: int = 64, device=None) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    exact = pixel = shape = 0.0
    n = 0
    for _, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
            demos, qx, model.cfg.max_demos, model.cfg.max_grid_size, device
        )
        rule = model.encode_rule(dx, dy, dxs, dys, dm)
        Hq = model.encode_grid(q, qs)
        rule = model.condition_rule_on_query(rule, Hq)
        Hd = model.predict_goal(Hq, rule)
        pred = _decode_grid(model, Hd)
        e, p, s = _grid_scores(pred, target)
        exact += e
        pixel += p
        shape += s
        n += 1

    if was_training:
        model.train()
    d = max(n, 1)
    return {
        "exact": exact / d,
        "pixel_acc": pixel / d,
        "shape_acc": shape / d,
        "count": n,
    }


@torch.no_grad()
def evaluate_arc(model, dataset, limit: int = 64, device=None) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    adaptive_exact = adaptive_pixel = adaptive_shape = 0.0
    fixed_exact = fixed_pixel = fixed_shape = 0.0
    direct_exact = direct_pixel = direct_shape = 0.0
    oracle_exact = oracle_pixel = oracle_shape = 0.0
    step_pixel_sum = [0.0 for _ in range(model.cfg.recursive_steps + 1)]
    chosen_step_sum = 0.0
    calibrated_depth_sum = 0.0
    oracle_step_sum = 0.0
    n = 0

    for _, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
            demos, qx, model.cfg.max_demos, model.cfg.max_grid_size, device
        )
        rule = model.encode_rule(dx, dy, dxs, dys, dm)
        Hq = model.encode_grid(q, qs)
        rule = model.condition_rule_on_query(rule, Hq)
        _, info = model.recursive_reason(
            Hq, rule, steps=model.cfg.recursive_steps, return_trace=True
        )

        calibrated_depth, demo_depth_scores = _select_depth_from_demos(
            model, demos, device
        )
        demo_profile = _demo_scores_to_cost_tensor(
            demo_depth_scores, device, rule.dtype
        )
        adaptive_state, chosen, controller_logits = model.select_demo_conditioned_depth(
            rule, info, demo_profile
        )
        chosen_depth = int(chosen[0].item())
        adaptive_pred = _decode_grid(model, adaptive_state)
        fixed_pred = _decode_grid(model, info["states"][-1])
        direct_pred = _decode_grid(model, info["states"][0])

        ae, ap, ash = _grid_scores(adaptive_pred, target)
        fe, fp, fsh = _grid_scores(fixed_pred, target)
        de, dp, dsh = _grid_scores(direct_pred, target)

        adaptive_exact += ae
        adaptive_pixel += ap
        adaptive_shape += ash
        fixed_exact += fe
        fixed_pixel += fp
        fixed_shape += fsh
        direct_exact += de
        direct_pixel += dp
        direct_shape += dsh
        chosen_step_sum += float(chosen[0].item())
        calibrated_depth_sum += float(calibrated_depth)

        # Oracle is diagnostic only: choose the target-best decoded state,
        # including step 0/direct. It is never used to make a real prediction.
        state_scores = []
        state_triplets = []
        for i, state in enumerate(info["states"]):
            triplet = _grid_scores(_decode_grid(model, state), target)
            state_triplets.append(triplet)
            step_pixel_sum[i] += triplet[1]
            state_scores.append((triplet[0], triplet[1], triplet[2]))

        oracle_idx = max(
            range(len(state_scores)),
            key=lambda i: (state_scores[i][0], state_scores[i][1], state_scores[i][2]),
        )
        oe, op, osh = state_triplets[oracle_idx]
        oracle_exact += oe
        oracle_pixel += op
        oracle_shape += osh
        oracle_step_sum += oracle_idx
        n += 1

    if was_training:
        model.train()

    d = max(n, 1)
    step_pixels = [x / d for x in step_pixel_sum]
    adaptive_pixel_acc = adaptive_pixel / d
    direct_pixel_acc = direct_pixel / d

    return {
        # Active inference path.
        "exact": adaptive_exact / d,
        "pixel_acc": adaptive_pixel_acc,
        "shape_acc": adaptive_shape / d,

        "adaptive_exact": adaptive_exact / d,
        "adaptive_pixel_acc": adaptive_pixel_acc,
        "adaptive_shape_acc": adaptive_shape / d,

        "fixed_exact": fixed_exact / d,
        "fixed_pixel_acc": fixed_pixel / d,
        "fixed_shape_acc": fixed_shape / d,

        "direct_exact": direct_exact / d,
        "direct_pixel_acc": direct_pixel_acc,
        "direct_shape_acc": direct_shape / d,

        "oracle_exact": oracle_exact / d,
        "oracle_pixel_acc": oracle_pixel / d,
        "oracle_shape_acc": oracle_shape / d,

        "recursive_gain": adaptive_pixel_acc - direct_pixel_acc,
        "step_pixel_acc": step_pixels,
        "avg_chosen_step": chosen_step_sum / d,
        "avg_calibrated_depth": calibrated_depth_sum / d,
        "avg_oracle_step": oracle_step_sum / d,
        "count": n,
    }

