from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from yetirah_arc_v1_training.yetirah_arc_v1.sefer.controllers.reasoner import TinyReasoner


def goal_error(model: TinyReasoner, current_H: torch.Tensor, target_H: torch.Tensor) -> torch.Tensor:
    current = model.decode_query(current_H)
    target = model.decode_query(target_H)
    # Support program-bank dimensions, e.g. current=[B,P,N], target=[B,N].
    while target.ndim < current.ndim:
        target = target.unsqueeze(1)
    return (current - target).pow(2).mean(dim=-1)


@torch.no_grad()
def terminal_goal_score(
    model: TinyReasoner, current_H: torch.Tensor, target_H: torch.Tensor, beta: float = 2.0
) -> torch.Tensor:
    return torch.exp(-beta * goal_error(model, current_H, target_H))


@torch.no_grad()
def exact_horizon_value_target(
    model: TinyReasoner,
    current_H: torch.Tensor,
    target_H: torch.Tensor,
    remaining_steps: int,
    action_limit: int = 4,
    beta: float = 2.0,
) -> torch.Tensor:
    """Exact finite-horizon value target, breadth-vectorized on GPU.

    Evaluates all primitive programs up to ``remaining_steps`` while treating
    every intermediate depth as a valid STOP point. For h<=4 this is tiny
    (1+4+16+64+256 states/sample) and avoids hundreds of recursive Python calls.
    No autograd graph is built because Bellman values are supervision targets.
    """
    h = max(0, int(remaining_steps))
    best = terminal_goal_score(model, current_H, target_H, beta=beta)
    if h == 0:
        return best

    # frontier: [B,P,N,D], P starts at one empty program.
    frontier = current_H.unsqueeze(1)
    target_bank = target_H.unsqueeze(1)
    primitive_count = min(4, model.operator_count, int(action_limit))
    for _depth in range(h):
        next_by_action = [model.core.apply_operator(frontier, a) for a in range(primitive_count)]
        # [B,P,A,N,D] -> [B,P*A,N,D]
        frontier = torch.stack(next_by_action, dim=2)
        B, P, A, N, D = frontier.shape
        frontier = frontier.reshape(B, P * A, N, D)
        scores = terminal_goal_score(model, frontier, target_bank, beta=beta)
        best = torch.maximum(best, scores.max(dim=1).values)
    return best


@torch.no_grad()
def exact_horizon_policy_target(
    model: TinyReasoner,
    current_H: torch.Tensor,
    target_H: torch.Tensor,
    remaining_steps: int,
    beta: float = 2.0,
    margin: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = current_H.shape[0]
    stop_q = terminal_goal_score(model, current_H, target_H, beta=beta)
    if remaining_steps <= 0:
        target = torch.zeros(B, 5, device=current_H.device, dtype=current_H.dtype)
        target[:, 4] = 1.0
        q = torch.zeros_like(target)
        q[:, 4] = stop_q
        return target, q

    q_terms = []
    for a in range(4):
        next_H = model.transition(current_H, a)
        q_terms.append(exact_horizon_value_target(
            model, next_H, target_H, remaining_steps - 1, action_limit=4, beta=beta
        ))
    q_terms.append(stop_q)
    q = torch.stack(q_terms, dim=-1)
    best = q.max(dim=-1, keepdim=True).values
    near_best = (q >= (best - float(margin))).to(q.dtype)
    target = near_best / near_best.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return target, q


def soft_policy_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
