from __future__ import annotations

import math
import random
import os
import pickle
import tempfile
import textwrap
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sefer.controllers.reasoner import TinyReasoner

def goal_error(model: TinyReasoner, current_H: torch.Tensor, target_H: torch.Tensor) -> torch.Tensor:
    """Per-sample support-to-goal MSE used as an observable planning signal."""
    current = model.decode_query(current_H)
    target = model.decode_query(target_H)
    return F.mse_loss(current, target, reduction="none").mean(dim=-1)

def terminal_goal_score(
    model: TinyReasoner, current_H: torch.Tensor, target_H: torch.Tensor, beta: float = 2.0
) -> torch.Tensor:
    """Terminal utility: 1 for exact goal match, decaying with support MSE."""
    return torch.exp(-beta * goal_error(model, current_H, target_H).detach())

def exact_horizon_value_target(
    model: TinyReasoner,
    current_H: torch.Tensor,
    target_H: torch.Tensor,
    remaining_steps: int,
    action_limit: int = 4,
    beta: float = 2.0,
) -> torch.Tensor:
    """Exact finite-horizon Bellman target for the synthetic proof.

    V*_h(s,g) = max over all h-step primitive programs of terminal_goal_score.
    With four actions and h<=2 this is cheap (at most 16 leaves/sample) and
    gives the learned value precisely the semantics PUCT requires.
    """
    if remaining_steps <= 0:
        return terminal_goal_score(model, current_H, target_H, beta=beta)
    candidates = [terminal_goal_score(model, current_H, target_H, beta=beta)]  # STOP now
    for a in range(min(4, model.operator_count)):
        next_H = model.transition(current_H, a)
        candidates.append(
            exact_horizon_value_target(
                model, next_H, target_H, remaining_steps - 1, action_limit, beta
            )
        )
    return torch.stack(candidates, dim=0).max(dim=0).values

def exact_horizon_policy_target(
    model: TinyReasoner,
    current_H: torch.Tensor,
    target_H: torch.Tensor,
    remaining_steps: int,
    beta: float = 2.0,
    margin: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional-equivalence-aware *decisive* policy target.

    v29 used a softmax over exact Bellman Q values. Because several actions can
    have numerically close Q values, that target often remained too diffuse.
    v30 instead gives equal probability only to actions within ``margin`` of the
    exact best Q. This preserves genuine functional equivalence while providing
    a much sharper training signal for the learned prior.
    """
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
        q_terms.append(
            exact_horizon_value_target(
                model, next_H, target_H, remaining_steps - 1, action_limit=5, beta=beta
            )
        )
    q_terms.append(stop_q)
    q = torch.stack(q_terms, dim=-1)
    best = q.max(dim=-1, keepdim=True).values
    near_best = (q >= (best - float(margin))).to(q.dtype)
    target = near_best / near_best.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return target, q

def soft_policy_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
