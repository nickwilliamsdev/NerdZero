from __future__ import annotations

# Kept for compatibility with the current launcher.
ARC_EVAL_PATCH_ID = "v1.7-demo-consistency-search"
ARC_EVAL_ARCH = "recursive-deltanet-v1"

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
    Hr, info = model.recursive_reason(
        Hq, rule, steps=cfg.recursive_steps, return_trace=True
    )

    recursive_grid = _decode_grid(model, Hr)
    direct_grid = _decode_grid(model, info["direct_state"])
    details = {
        "mode": "recursive_deltanet",
        "steps": cfg.recursive_steps,
        "halt_probs": [
            float(torch.sigmoid(x).item()) for x in info["halt_logits"]
        ],
        "fast_norm": (
            float(info["stats"][-1]["fast_norm"].mean().item())
            if info["stats"] else 0.0
        ),
        "direct_grid": direct_grid,
    }

    if return_details:
        return recursive_grid, [], direct_grid, details
    if return_direct:
        return recursive_grid, [], direct_grid
    return recursive_grid, []


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

    exact = pixel = shape = 0.0
    direct_exact = direct_pixel = direct_shape = 0.0
    step_pixel_sum = [0.0 for _ in range(model.cfg.recursive_steps + 1)]
    n = 0

    for _, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
            demos, qx, model.cfg.max_demos, model.cfg.max_grid_size, device
        )
        rule = model.encode_rule(dx, dy, dxs, dys, dm)
        Hq = model.encode_grid(q, qs)
        rule = model.condition_rule_on_query(rule, Hq)
        Hr, info = model.recursive_reason(
            Hq, rule, steps=model.cfg.recursive_steps, return_trace=True
        )

        pred = _decode_grid(model, Hr)
        direct_pred = _decode_grid(model, info["states"][0])
        e, p, s = _grid_scores(pred, target)
        de, dp, ds = _grid_scores(direct_pred, target)

        exact += e
        pixel += p
        shape += s
        direct_exact += de
        direct_pixel += dp
        direct_shape += ds

        for i, state in enumerate(info["states"]):
            _, sp, _ = _grid_scores(_decode_grid(model, state), target)
            step_pixel_sum[i] += sp
        n += 1

    if was_training:
        model.train()

    d = max(n, 1)
    step_pixels = [x / d for x in step_pixel_sum]
    result = {
        "exact": exact / d,
        "pixel_acc": pixel / d,
        "shape_acc": shape / d,
        "direct_exact": direct_exact / d,
        "direct_pixel_acc": direct_pixel / d,
        "direct_shape_acc": direct_shape / d,
        "recursive_gain": (pixel - direct_pixel) / d,
        "step_pixel_acc": step_pixels,
        "count": n,
    }
    return result
