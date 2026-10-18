from __future__ import annotations

# Kept for compatibility with the current launcher.
ARC_EVAL_PATCH_ID = "v1.7-demo-consistency-search"
ARC_EVAL_ARCH = "recursive-deltanet-v1-multihypothesis-v14"

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


def _depth_ensemble_weights(model, demo_depth_scores, device, dtype):
    """Turn task-local demo scores into sparse depth weights."""
    scores = torch.tensor(
        demo_depth_scores, device=device, dtype=dtype
    )
    K = scores.numel()

    # Slight late-depth prior counters the systematic shallow bias observed
    # in hard demo calibration, while demo evidence remains dominant.
    depth = torch.arange(K, device=device, dtype=dtype)
    if K > 1:
        depth = depth / float(K - 1)
    adjusted = scores + model.cfg.depth_ensemble_late_bias * depth

    top_k = max(1, min(int(model.cfg.depth_ensemble_top_k), K))
    keep = torch.topk(adjusted, k=top_k).indices
    masked = torch.full_like(adjusted, float("-inf"))
    masked[keep] = adjusted[keep]

    temp = max(float(model.cfg.depth_ensemble_temperature), 1e-4)
    weights = torch.softmax(masked / temp, dim=0)

    # Preserve some of the robust fixed-final prediction in every task.
    floor = float(model.cfg.depth_ensemble_fixed_floor)
    floor = max(0.0, min(floor, 0.95))
    weights = weights * (1.0 - floor)
    weights[-1] = weights[-1] + floor
    weights = weights / weights.sum().clamp_min(1e-8)
    return weights


def _decode_weighted_states(model, states, weights):
    """Decode a weighted ensemble in logit space."""
    color_mix = None
    h_mix = None
    w_mix = None

    for weight, state in zip(weights, states):
        color_logits, h_logits, w_logits = model.decode_grid(state)
        if color_mix is None:
            color_mix = weight * color_logits
            h_mix = weight * h_logits
            w_mix = weight * w_logits
        else:
            color_mix = color_mix + weight * color_logits
            h_mix = h_mix + weight * h_logits
            w_mix = w_mix + weight * w_logits

    h = int(h_mix.argmax(dim=-1).item()) + 1
    w = int(w_mix.argmax(dim=-1).item()) + 1
    return (
        color_mix.argmax(dim=1)[0, :h, :w]
        .cpu().numpy().astype(np.int64)
    )


@torch.no_grad()
def _score_hypotheses_on_demos(model, demos, device):
    """Return demo-consistency scores [K,T+1] for every hypothesis/depth."""
    K = model.cfg.hypothesis_count
    T = model.cfg.recursive_steps
    scores = torch.zeros(K, T + 1, device=device)
    used = 0

    if len(demos) < 2:
        scores[:, -1] = 1.0
        return scores

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
        _, multi = model.multi_hypothesis_reason(
            Hq, rule, steps=T, return_trace=True
        )

        for k, info in enumerate(multi["hypotheses"]):
            for depth, state in enumerate(info["states"]):
                pred = _decode_grid(model, state)
                exact, pixel, shape = _grid_scores(pred, held_out)
                scores[k, depth] += (
                    pixel
                    + model.cfg.demo_depth_exact_weight * exact
                    + model.cfg.demo_depth_shape_weight * shape
                )
        used += 1

    return scores / max(used, 1)


def _select_hypothesis_and_weights(model, demo_scores):
    """Select branch by LOO consistency, then ensemble its supported depths."""
    K, D = demo_scores.shape
    branch_scores = []
    branch_weights = []

    for k in range(K):
        weights = _depth_ensemble_weights(
            model,
            demo_scores[k].detach().cpu().tolist(),
            demo_scores.device,
            demo_scores.dtype,
        )
        branch_weights.append(weights)
        branch_scores.append((weights * demo_scores[k]).sum())

    branch_scores = torch.stack(branch_scores)
    chosen = int(branch_scores.argmax().item())
    return chosen, branch_weights[chosen], branch_scores


def _decoded_candidate_diversity(preds):
    """Average pairwise disagreement over overlapping decoded cells."""
    vals = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            a, b = preds[i], preds[j]
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            if h and w:
                vals.append(float(np.mean(a[:h, :w] != b[:h, :w])))
    return float(np.mean(vals)) if vals else 0.0


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

    _, multi = model.multi_hypothesis_reason(
        Hq, rule, steps=cfg.recursive_steps, return_trace=True
    )
    demo_scores = _score_hypotheses_on_demos(model, demos, device)
    chosen_hyp, depth_weights, branch_scores = _select_hypothesis_and_weights(
        model, demo_scores
    )

    chosen_info = multi["hypotheses"][chosen_hyp]
    pred = _decode_weighted_states(
        model, chosen_info["states"], depth_weights
    )
    direct_grid = _decode_grid(model, chosen_info["states"][0])

    details = {
        "mode": "multi_hypothesis_recursive_deltanet",
        "chosen_hypothesis": chosen_hyp,
        "branch_scores": branch_scores.detach().cpu().tolist(),
        "depth_weights": depth_weights.detach().cpu().tolist(),
        "demo_scores": demo_scores.detach().cpu().tolist(),
        "direct_grid": direct_grid,
    }

    if return_details:
        return pred, [], direct_grid, details
    if return_direct:
        return pred, [], direct_grid
    return pred, []


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

    selected_exact = selected_pixel = selected_shape = 0.0
    direct_exact = direct_pixel = direct_shape = 0.0
    oracle_candidate_exact = oracle_candidate_pixel = oracle_candidate_shape = 0.0
    oracle_state_exact = oracle_state_pixel = oracle_state_shape = 0.0
    selected_hyp_sum = 0.0
    oracle_hyp_sum = 0.0
    diversity_sum = 0.0
    n = 0

    for _, demos, qx, target in dataset.evaluation_episodes(limit=limit):
        dx, dy, dxs, dys, dm, q, qs = _episode_tensors(
            demos, qx, model.cfg.max_demos,
            model.cfg.max_grid_size, device
        )
        rule = model.encode_rule(dx, dy, dxs, dys, dm)
        Hq = model.encode_grid(q, qs)
        rule = model.condition_rule_on_query(rule, Hq)

        _, multi = model.multi_hypothesis_reason(
            Hq, rule, steps=model.cfg.recursive_steps, return_trace=True
        )
        demo_scores = _score_hypotheses_on_demos(model, demos, device)
        chosen_hyp, chosen_weights, branch_scores = _select_hypothesis_and_weights(
            model, demo_scores
        )

        candidate_preds = []
        candidate_triplets = []
        for k, info in enumerate(multi["hypotheses"]):
            weights = _depth_ensemble_weights(
                model,
                demo_scores[k].detach().cpu().tolist(),
                device,
                rule.dtype,
            )
            p = _decode_weighted_states(model, info["states"], weights)
            candidate_preds.append(p)
            candidate_triplets.append(_grid_scores(p, target))

        selected_pred = candidate_preds[chosen_hyp]
        se, sp, ss = candidate_triplets[chosen_hyp]
        selected_exact += se
        selected_pixel += sp
        selected_shape += ss
        selected_hyp_sum += chosen_hyp

        # Direct baseline uses the unperturbed rule/direct predictor.
        Hd = model.predict_goal(Hq, rule)
        direct_pred = _decode_grid(model, Hd)
        de, dp, ds = _grid_scores(direct_pred, target)
        direct_exact += de
        direct_pixel += dp
        direct_shape += ds

        # Oracle candidate: target chooses among branch-level predictions only.
        oracle_hyp = max(
            range(len(candidate_triplets)),
            key=lambda k: (
                candidate_triplets[k][0],
                candidate_triplets[k][1],
                candidate_triplets[k][2],
            ),
        )
        oce, ocp, ocs = candidate_triplets[oracle_hyp]
        oracle_candidate_exact += oce
        oracle_candidate_pixel += ocp
        oracle_candidate_shape += ocs
        oracle_hyp_sum += oracle_hyp

        # Oracle state: target chooses any hypothesis at any depth.
        all_states = []
        for k, info in enumerate(multi["hypotheses"]):
            for depth, state in enumerate(info["states"]):
                triplet = _grid_scores(_decode_grid(model, state), target)
                all_states.append((triplet, k, depth))
        best_state = max(
            all_states,
            key=lambda x: (x[0][0], x[0][1], x[0][2]),
        )
        ose, osp, oss = best_state[0]
        oracle_state_exact += ose
        oracle_state_pixel += osp
        oracle_state_shape += oss

        diversity_sum += _decoded_candidate_diversity(candidate_preds)
        n += 1

    if was_training:
        model.train()

    d = max(n, 1)
    selected_pixel_acc = selected_pixel / d
    direct_pixel_acc = direct_pixel / d

    return {
        "exact": selected_exact / d,
        "pixel_acc": selected_pixel_acc,
        "shape_acc": selected_shape / d,

        "adaptive_exact": selected_exact / d,
        "adaptive_pixel_acc": selected_pixel_acc,
        "adaptive_shape_acc": selected_shape / d,

        # Backward-compatible keys. Candidate selection replaces fixed-depth.
        "fixed_exact": selected_exact / d,
        "fixed_pixel_acc": selected_pixel_acc,
        "fixed_shape_acc": selected_shape / d,

        "direct_exact": direct_exact / d,
        "direct_pixel_acc": direct_pixel_acc,
        "direct_shape_acc": direct_shape / d,

        "oracle_candidate_exact": oracle_candidate_exact / d,
        "oracle_candidate_pixel_acc": oracle_candidate_pixel / d,
        "oracle_candidate_shape_acc": oracle_candidate_shape / d,

        "oracle_exact": oracle_state_exact / d,
        "oracle_pixel_acc": oracle_state_pixel / d,
        "oracle_shape_acc": oracle_state_shape / d,

        "recursive_gain": selected_pixel_acc - direct_pixel_acc,
        "avg_chosen_step": selected_hyp_sum / d,
        "avg_ensemble_depth": 0.0,
        "avg_oracle_step": oracle_hyp_sum / d,
        "candidate_diversity": diversity_sum / d,
        "count": n,
    }

