from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sefer.tasks.arc_dataset import _pair_arrays, pad_grid

ARC_EVAL_PATCH_ID = "v1.7-demo-consistency-search"


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


def _candidate_grid_loss(model, states, targets, target_shapes):
    """Return per-candidate decoded demo loss for states [C,D,N,F]."""
    C, D, N, Fdim = states.shape
    flat = states.reshape(C * D, N, Fdim)
    color_logits, h_logits, w_logits = model.decode_grid(flat)
    S = targets.shape[-1]
    target_rep = targets[None].expand(C, D, S, S).reshape(C * D, S, S)
    shape_rep = target_shapes[None].expand(C, D, 2).reshape(C * D, 2)

    cell = F.cross_entropy(color_logits, target_rep, reduction="none")
    rr = torch.arange(S, device=states.device)[None, :, None]
    cc = torch.arange(S, device=states.device)[None, None, :]
    valid = (rr < shape_rep[:, 0, None, None]) & (cc < shape_rep[:, 1, None, None])
    per_ex = (cell * valid.float()).sum(dim=(1, 2)) / valid.float().sum(dim=(1, 2)).clamp_min(1.0)
    hce = F.cross_entropy(h_logits, shape_rep[:, 0] - 1, reduction="none")
    wce = F.cross_entropy(w_logits, shape_rep[:, 1] - 1, reduction="none")
    return per_ex.reshape(C, D).mean(dim=1), (hce + wce).reshape(C, D).mean(dim=1)


def _candidate_demo_pixel_fit(model, state, targets, target_shapes):
    """ARC-style mean pixel fit over demonstrations for one candidate [D,N,F]."""
    D = state.shape[0]
    color_logits, h_logits, w_logits = model.decode_grid(state)
    pred = color_logits.argmax(dim=1)
    pred_h = h_logits.argmax(dim=-1) + 1
    pred_w = w_logits.argmax(dim=-1) + 1
    fits = []
    for d in range(D):
        th = int(target_shapes[d, 0])
        tw = int(target_shapes[d, 1])
        ph = int(pred_h[d])
        pw = int(pred_w[d])
        h = min(th, ph)
        w = min(tw, pw)
        if h <= 0 or w <= 0:
            fits.append(0.0)
            continue
        correct = (pred[d, :h, :w] == targets[d, :h, :w]).float().sum()
        fits.append(float(correct / max(th * tw, 1)))
    return float(sum(fits) / max(len(fits), 1))


@torch.no_grad()
def search_demo_consistent_program(model, dx, dy, dxs, dys, dm, base_rule):
    """Find one operator sequence that best explains all known demonstrations.

    Search is target-free with respect to the unseen query. Known demo outputs
    are used exactly as ARC permits: they score candidate shared programs. The
    same discrete sequence is then transferred unchanged to the test query.
    """
    cfg = model.cfg
    valid = dm[0].nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return [], 0.0, {"root_ops": []}

    x = dx[0, valid]
    y = dy[0, valid]
    xs = dxs[0, valid]
    ys = dys[0, valid]
    Hx = model.encode_grid(x, xs)
    Hy = model.encode_grid(y, ys)
    D = Hx.shape[0]
    rules = base_rule.expand(D, -1)
    rules = model.condition_rule_on_query(rules, Hx)

    # ARC-adaptive action set: select operators that already explain demos best
    # at one step, then search compositions only within this task-local subset.
    root_all = model.operator_bank.apply_all(Hx, rules)  # [D,K,N,F]
    root_err = (root_all - Hy[:, None]).pow(2).mean(dim=(2, 3)).mean(dim=0)
    top_ops = min(int(cfg.demo_search_top_ops), int(cfg.operator_count))
    active = torch.topk(root_err, k=max(top_ops, 1), largest=False).indices
    A = int(active.numel())

    beam_width = max(int(cfg.demo_search_beam_width), 1)
    depth = max(int(cfg.demo_search_depth), 1)
    frontier = Hx[None]  # [Bbeam,D,N,F]
    seqs: List[Tuple[int, ...]] = [tuple()]
    archive_states: List[torch.Tensor] = [frontier[0].clone()]
    archive_seqs: List[Tuple[int, ...]] = [tuple()]
    archive_lat: List[float] = [float((Hx - Hy).pow(2).mean())]

    for level in range(depth):
        K = frontier.shape[0]
        flat_state = frontier.reshape(K * D, model.n_slots, model.node_dim)
        flat_rule = rules[None].expand(K, D, rules.shape[-1]).reshape(K * D, -1)
        all_ops = model.operator_bank.apply_all(flat_state, flat_rule)[:, active]
        cand = all_ops.reshape(K, D, A, model.n_slots, model.node_dim)
        cand = cand.permute(0, 2, 1, 3, 4).reshape(K * A, D, model.n_slots, model.node_dim)
        lat_score = (cand - Hy[None]).pow(2).mean(dim=(1, 2, 3))
        lat_score = lat_score + float(cfg.demo_search_length_weight) * float(level + 1)

        keep = min(beam_width, cand.shape[0])
        vals, idx = torch.topk(lat_score, k=keep, largest=False)
        frontier = cand[idx]
        new_seqs: List[Tuple[int, ...]] = []
        for raw_idx in idx.tolist():
            parent = raw_idx // A
            local = raw_idx % A
            new_seqs.append(seqs[parent] + (int(active[local]),))
        seqs = new_seqs
        for j in range(keep):
            archive_states.append(frontier[j].clone())
            archive_seqs.append(seqs[j])
            archive_lat.append(float(vals[j]))

    # Re-rank a small latent shortlist in decoded grid space. This protects the
    # search from latent-distance/pathologies that do not correspond to ARC cell
    # correctness or output shape.
    order = np.argsort(np.asarray(archive_lat))[: max(int(cfg.demo_search_finalists), 1)]
    states = torch.stack([archive_states[i] for i in order], dim=0)
    color_loss, shape_loss = _candidate_grid_loss(model, states, y, ys)
    lat = torch.as_tensor([archive_lat[i] for i in order], device=states.device, dtype=states.dtype)
    lengths = torch.as_tensor(
        [len(archive_seqs[i]) for i in order], device=states.device, dtype=states.dtype
    )
    final_score = (
        lat
        + float(cfg.demo_search_grid_weight) * color_loss
        + float(cfg.demo_search_shape_weight) * shape_loss
        + float(cfg.demo_search_length_weight) * lengths
    )
    best_local = int(final_score.argmin())
    best_archive_idx = int(order[best_local])
    best_seq = list(archive_seqs[best_archive_idx])
    demo_fit = _candidate_demo_pixel_fit(model, archive_states[best_archive_idx], y, ys)
    return best_seq, demo_fit, {"root_ops": active.cpu().tolist(), "score": float(final_score[best_local])}


@torch.no_grad()
def _predict_greedy(model, dx, dy, dxs, dys, dm, q, qs):
    rule = model.encode_rule(dx, dy, dxs, dys, dm)
    Hq = model.encode_grid(q, qs)
    rule = model.condition_rule_on_query(rule, Hq)
    Hgoal = model.predict_goal(Hq, rule)
    Hp, actions = model.greedy_actions(
        Hq,
        rule,
        goal_H=Hgoal,
        max_steps=model.cfg.max_program_steps,
        active_operator_count=model.cfg.operator_count,
    )
    return Hp, actions, Hgoal, rule, Hq


@torch.no_grad()
def predict_arc(model, demos, query_x, device=None, return_direct: bool = False, return_details: bool = False):
    """ARC prediction with optional demo-consistency program search.

    The search sees only demonstrations and the query input. It never sees the
    query target. If disabled, behavior falls back to the learned greedy policy.
    """
    device = device or next(model.parameters()).device
    cfg = model.cfg
    dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
        demos, query_x, cfg.max_demos, cfg.max_grid_size, device
    )
    greedy_H, greedy_actions, Hgoal, rule, Hq = _predict_greedy(model, dx, dy, dxs, dys, dm, q, qs)
    greedy_grid = _decode_grid(model, greedy_H)
    direct_grid = _decode_grid(model, Hgoal)

    details = {"mode": "greedy", "demo_fit": 0.0, "greedy_grid": greedy_grid}
    if getattr(cfg, "demo_search_enabled", False):
        base_rule = model.encode_rule(dx, dy, dxs, dys, dm)
        seq, demo_fit, search_info = search_demo_consistent_program(
            model, dx, dy, dxs, dys, dm, base_rule
        )
        if len(seq) == 0:
            final_H = Hgoal
        else:
            final_H, blend = model.apply_action_sequence(Hq, rule, seq, goal_H=Hgoal, blend=True)
            details["blend"] = float(blend.mean()) if blend is not None else 0.0
        grid = _decode_grid(model, final_H)
        actions = list(seq)[: cfg.max_program_steps]
        actions += [cfg.operator_count] * max(cfg.max_program_steps - len(actions), 0)
        details.update(search_info)
        details.update({"mode": "demo_search", "demo_fit": demo_fit, "sequence": list(seq)})
    else:
        grid = greedy_grid
        actions = greedy_actions[0].cpu().tolist()

    if return_details:
        return grid, actions, direct_grid, details
    if not return_direct:
        return grid, actions
    return grid, actions, direct_grid


@torch.no_grad()
def evaluate_arc_direct(model, dataset, limit: int = 64, device=None) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    exact = pixel_sum = shape_sum = 0.0
    n = 0
    for task_id, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
            demos, qx, model.cfg.max_demos, model.cfg.max_grid_size, device
        )
        rule = model.encode_rule(dx, dy, dxs, dys, dm)
        Hq = model.encode_grid(q, qs)
        rule = model.condition_rule_on_query(rule, Hq)
        Hgoal = model.predict_goal(Hq, rule)
        pred = _decode_grid(model, Hgoal)
        e, p, sh = _grid_scores(pred, target)
        exact += e
        pixel_sum += p
        shape_sum += sh
        n += 1
    if was_training:
        model.train()
    denom = max(n, 1)
    return {
        "count": n,
        "exact": exact / denom,
        "pixel_acc": pixel_sum / denom,
        "shape_acc": shape_sum / denom,
    }


@torch.no_grad()
def evaluate_arc(model, dataset, limit: int = 64, device=None) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    exact = pixel_sum = shape_sum = 0.0
    direct_exact = direct_pixel_sum = direct_shape_sum = 0.0
    greedy_exact = greedy_pixel_sum = greedy_shape_sum = 0.0
    n = 0
    action_counter = Counter()
    program_length_sum = 0.0
    demo_fit_sum = 0.0

    for task_id, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        pred, actions, direct_pred, details = predict_arc(
            model, demos, qx, device=device, return_direct=True, return_details=True
        )
        e, p, s = _grid_scores(pred, target)
        de, dp, ds = _grid_scores(direct_pred, target)
        ge, gp, gs = _grid_scores(details["greedy_grid"], target)
        exact += e
        pixel_sum += p
        shape_sum += s
        direct_exact += de
        direct_pixel_sum += dp
        direct_shape_sum += ds
        greedy_exact += ge
        greedy_pixel_sum += gp
        greedy_shape_sum += gs
        demo_fit_sum += float(details.get("demo_fit", 0.0))
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
        "greedy_exact": greedy_exact / denom,
        "greedy_pixel_acc": greedy_pixel_sum / denom,
        "greedy_shape_acc": greedy_shape_sum / denom,
        "search_demo_fit": demo_fit_sum / denom,
        "action_usage": dict(action_counter),
        "mean_program_length": program_length_sum / denom,
    }
