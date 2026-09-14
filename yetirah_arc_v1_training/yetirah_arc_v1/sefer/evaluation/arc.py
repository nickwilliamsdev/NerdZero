from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np
import torch

from yetirah_arc_v1_training.yetirah_arc_v1.sefer.tasks.arc_dataset import _pair_arrays, pad_grid


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


@torch.no_grad()
def predict_arc(model, demos, query_x, device=None):
    device = device or next(model.parameters()).device
    cfg = model.cfg
    dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
        demos, query_x, cfg.max_demos, cfg.max_grid_size, device
    )
    rule = model.encode_rule(dx, dy, dxs, dys, dm)
    Hq = model.encode_grid(q, qs)
    Hp, actions = model.greedy_actions(Hq, rule, max_steps=cfg.max_program_steps)
    color_logits, h_logits, w_logits = model.decode_grid(Hp)
    h = int(h_logits.argmax(dim=-1).item()) + 1
    w = int(w_logits.argmax(dim=-1).item()) + 1
    grid = color_logits.argmax(dim=1)[0, :h, :w].cpu().numpy().astype(np.int64)
    return grid, actions[0].cpu().tolist()


@torch.no_grad()
def evaluate_arc(model, dataset, limit: int = 64, device=None) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    exact = 0
    pixel_sum = 0.0
    shape_sum = 0.0
    n = 0
    action_counter = Counter()
    for task_id, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        pred, actions = predict_arc(model, demos, qx, device=device)
        target = np.asarray(target, dtype=np.int64)
        shape_ok = pred.shape == target.shape
        shape_sum += float(shape_ok)
        h = min(pred.shape[0], target.shape[0])
        w = min(pred.shape[1], target.shape[1])
        overlap = float(np.mean(pred[:h, :w] == target[:h, :w])) if h and w else 0.0
        # Penalize wrong output shape rather than giving full overlap credit.
        target_cells = max(int(target.size), 1)
        overlap_cells = h * w
        pixel_score = overlap * (overlap_cells / target_cells)
        pixel_sum += pixel_score
        exact += int(shape_ok and np.array_equal(pred, target))
        action_counter.update(actions)
        n += 1
    if was_training:
        model.train()
    denom = max(n, 1)
    return {
        "count": n,
        "exact": exact / denom,
        "pixel_acc": pixel_sum / denom,
        "shape_acc": shape_sum / denom,
        "action_usage": dict(action_counter),
    }
