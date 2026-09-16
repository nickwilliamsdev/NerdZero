from __future__ import annotations

from collections import Counter
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
def predict_arc(model, demos, query_x, device=None, return_direct: bool = False):
    """Predict with the program path using a demo-conditioned latent goal.

    ``direct_grid`` is the decoded predicted goal before operator execution. It
    is useful diagnostically because it tells us whether errors come from task
    inference or from program execution.
    """
    device = device or next(model.parameters()).device
    cfg = model.cfg
    dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
        demos, query_x, cfg.max_demos, cfg.max_grid_size, device
    )
    rule = model.encode_rule(dx, dy, dxs, dys, dm)
    Hq = model.encode_grid(q, qs)
    Hgoal = model.predict_goal(Hq, rule)
    Hp, actions = model.greedy_actions(
        Hq,
        rule,
        goal_H=Hgoal,
        max_steps=cfg.max_program_steps,
        active_operator_count=cfg.operator_count,
    )
    grid = _decode_grid(model, Hp)
    if not return_direct:
        return grid, actions[0].cpu().tolist()
    direct_grid = _decode_grid(model, Hgoal)
    return grid, actions[0].cpu().tolist(), direct_grid


@torch.no_grad()
def evaluate_arc(model, dataset, limit: int = 64, device=None) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    exact = pixel_sum = shape_sum = 0.0
    direct_exact = direct_pixel_sum = direct_shape_sum = 0.0
    n = 0
    action_counter = Counter()
    program_length_sum = 0.0

    for task_id, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        pred, actions, direct_pred = predict_arc(
            model, demos, qx, device=device, return_direct=True
        )
        e, p, s = _grid_scores(pred, target)
        de, dp, ds = _grid_scores(direct_pred, target)
        exact += e
        pixel_sum += p
        shape_sum += s
        direct_exact += de
        direct_pixel_sum += dp
        direct_shape_sum += ds
        action_counter.update(actions)
        program_length_sum += sum(int(a != model.cfg.operator_count) for a in actions)
        n += 1

    if was_training:
        model.train()
    denom = max(n, 1)
    return {
        "count": n,
        "exact": exact / denom,
        "pixel_acc": pixel_sum / denom,
        "shape_acc": shape_sum / denom,
        "direct_exact": direct_exact / denom,
        "direct_pixel_acc": direct_pixel_sum / denom,
        "direct_shape_acc": direct_shape_sum / denom,
        "action_usage": dict(action_counter),
        "mean_program_length": program_length_sum / denom,
    }
