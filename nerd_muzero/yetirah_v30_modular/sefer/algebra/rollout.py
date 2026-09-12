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

def differentiable_rollout(
    model: TinyReasoner,
    rule: torch.Tensor,
    H: torch.Tensor,
    steps: int = 1,
    temperature: float = 1.0,
    hard: bool = True,
):
    """Near-discrete operator rollout over a task-independent query workspace."""
    A = model.core.adjacency()
    routing_stats = []
    for _ in range(steps):
        logits, _ = model.policy_value(H, rule)
        op_logits = logits[:, :model.operator_count]
        probs = F.gumbel_softmax(op_logits, tau=temperature, hard=hard, dim=-1)
        all_next = [model.core.apply_operator(H, k, A) for k in range(model.operator_count)]
        stack = torch.stack(all_next, dim=1)
        H = torch.einsum("bk,bknd->bnd", probs, stack)
        routing_stats.append(torch.softmax(op_logits / max(temperature, 1e-4), dim=-1))
    return H, routing_stats

def deterministic_operator_rollout(
    model: TinyReasoner,
    rule: torch.Tensor,
    H: torch.Tensor,
    steps: int = 1,
):
    A = model.core.adjacency()
    chosen_steps = []
    for _ in range(steps):
        logits, _ = model.policy_value(H, rule)
        actions = logits[:, :model.operator_count].argmax(dim=-1)
        chosen_steps.append(actions)
        H = apply_selected_actions(model, H, actions, A)
    return H, torch.stack(chosen_steps, dim=1)

def deterministic_primitive_rollout_with_halt(
    model: TinyReasoner, rule: torch.Tensor, H: torch.Tensor, max_steps: int = 4
):
    """Primitive-policy rollout that respects the model's HALT action.

    Unlike the legacy diagnostic, this does not force ten operator applications.
    Once HALT is selected the sample stays fixed for the remainder of the trace.
    """
    A = model.core.adjacency()
    B = H.shape[0]
    active = torch.ones(B, dtype=torch.bool, device=H.device)
    traces = []
    for _ in range(max_steps):
        phase = 0 if len(traces) == 0 else 1
        logits, _ = model.policy_value(H, rule, primitive_phase=phase)
        actions = logits.argmax(dim=-1)
        actions = torch.where(active, actions, torch.full_like(actions, model.HALT_ACTION))
        traces.append(actions)
        do_op = active & (actions < model.operator_count)
        if do_op.any():
            H_new = H.clone()
            H_new[do_op] = apply_selected_actions(model, H[do_op], actions[do_op], A)
            H = H_new
        active = active & (actions != model.HALT_ACTION)
        if not active.any():
            break
    if not traces:
        traces = [torch.full((B,), model.HALT_ACTION, device=H.device, dtype=torch.long)]
    return H, torch.stack(traces, dim=1)

def apply_selected_actions(model, H, actions, adjacency=None):
    if adjacency is None:
        adjacency = model.core.adjacency()
    all_next = torch.stack(
        [model.core.apply_operator(H, k, adjacency) for k in range(model.operator_count)],
        dim=1,
    )
    selector = F.one_hot(actions, num_classes=model.operator_count).to(H.dtype)
    return torch.einsum("bk,bknd->bnd", selector, all_next)

def apply_program_actions(model, H, actions, adjacency=None):
    """Apply primitive actions 0..3; STOP(4) leaves state unchanged."""
    out = H.clone()
    active = actions < 4
    if active.any():
        out[active] = apply_selected_actions(model, H[active], actions[active], adjacency)
    return out

def demo_guided_two_step_search(model, probe_x, probe_y, query_x):
    """Search all 22x22 programs on a known probe pair.

    Execution is rule-blind: candidate operators act only on query-derived Q.
    The known probe output is used solely as a program-selection objective.
    """
    B = query_x.shape[0]
    K = model.operator_count
    A = model.core.adjacency()
    H_probe = model.encode_query(probe_x)
    first = torch.stack([model.core.apply_operator(H_probe, k, A) for k in range(K)], dim=1)
    first_flat = first.reshape(B * K, first.shape[-2], first.shape[-1])
    second = torch.stack([model.core.apply_operator(first_flat, k, A) for k in range(K)], dim=1)
    all_states = second.reshape(B, K, K, second.shape[-2], second.shape[-1])
    flat_states = all_states.reshape(B * K * K, second.shape[-2], second.shape[-1])
    pred_probe = model.decode_query(flat_states).reshape(B, K, K, -1)
    score = (pred_probe - probe_y[:, None, None, :]).pow(2).mean(dim=-1)
    best_flat = score.reshape(B, K * K).argmin(dim=-1)
    first_action = best_flat // K
    second_action = best_flat % K
    sequences = torch.stack([first_action, second_action], dim=-1)
    best_probe_mse = score.reshape(B, K * K).gather(1, best_flat[:, None]).squeeze(1)
    H_query = model.encode_query(query_x)
    H_query = apply_selected_actions(model, H_query, first_action, A)
    H_query = apply_selected_actions(model, H_query, second_action, A)
    return model.decode_query(H_query), sequences, best_probe_mse

def operator_separation_loss(
    model: TinyReasoner,
    H: torch.Tensor,
    margin: float = 0.03,
    max_batch: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encourage distinct operator effects without rewarding unbounded motion.

    The loss is a normalized margin penalty on pairwise distances between
    operator *outputs*. Once two operators differ by at least `margin`, there is
    no further reward for pushing them apart.
    """
    Hs = H[:max_batch]
    A = model.core.adjacency()
    states = torch.stack(
        [model.core.apply_operator(Hs, k, A) for k in range(model.operator_count)],
        dim=1,
    )
    diff = states[:, :, None] - states[:, None, :]
    dist = (diff.pow(2).mean(dim=(-1, -2)) + 1e-8).sqrt()
    upper = torch.triu(
        torch.ones(model.operator_count, model.operator_count, device=H.device, dtype=torch.bool),
        diagonal=1,
    )
    pair = dist[:, upper]
    loss = F.relu(1.0 - pair / margin).pow(2).mean()
    return loss, pair.mean()
