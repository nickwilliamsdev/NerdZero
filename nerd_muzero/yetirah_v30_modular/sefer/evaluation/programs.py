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

from sefer.algebra.rollout import apply_program_actions
from sefer.controllers.reasoner import TinyReasoner
from sefer.planning.bellman import goal_error
from sefer.tasks.synthetic_algebra import enumerate_primitive_programs

def evaluate_greedy_program_accuracy(model, task_source, device, batch_size: int = 256, max_depth: int = 4):
    model.eval()
    out = {}
    for name, seq in task_source.COMPOSITIONS.items():
        demo_x, demo_y, query_x, query_y = task_source.sample_composition(name, batch_size, device)
        support = model.encode_query(demo_x)
        target_support = model.encode_query(demo_y)
        query = model.encode_query(query_x)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        chosen_steps = []
        for step_idx in range(max_depth):
            remaining = max_depth - step_idx
            logits = model.program_policy(support, target_support, remaining_steps=remaining)[:, :5]
            action = logits.argmax(dim=-1)
            action = torch.where(stopped, torch.full_like(action, 4), action)
            chosen_steps.append(action)
            newly_stop = action == 4
            support = apply_program_actions(model, support, action)
            query = apply_program_actions(model, query, action)
            stopped |= newly_stop
        chosen = torch.stack(chosen_steps, dim=1)
        target = torch.full((batch_size, max_depth), 4, device=device, dtype=torch.long)
        target[:, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
        exact = (chosen == target).all(dim=-1).float().mean()
        mse = F.mse_loss(model.decode_query(query), query_y)
        functional, _ = functional_equivalence_rate(
            model, model.encode_query(query_x), query_y, chosen, tuple(seq), tolerance=0.03
        )
        out[name] = (float(exact.item()), float(mse.item()), functional)
    return out

def per_sample_output_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="none").mean(dim=-1)

def functional_equivalence_rate(
    model: TinyReasoner,
    query0: torch.Tensor,
    query_y: torch.Tensor,
    chosen_actions: torch.Tensor,
    reference_program: Tuple[int, ...],
    tolerance: float = 0.03,
) -> Tuple[float, float]:
    """Score programs by behavior rather than literal action-string identity.

    A chosen program is functionally equivalent when its query error is within
    `tolerance` MSE of the known semantic program on the same sample. This is
    intentionally tolerant of commuting/cancelling/shorter equivalent programs.
    """
    h_chosen = query0
    stopped = torch.zeros(query0.shape[0], dtype=torch.bool, device=query0.device)
    for t in range(chosen_actions.shape[1]):
        a = chosen_actions[:, t]
        effective = torch.where(stopped, torch.full_like(a, 4), a)
        h_chosen = apply_program_actions(model, h_chosen, effective)
        stopped |= effective == 4
    pred = model.decode_query(h_chosen)
    chosen_err = per_sample_output_mse(pred, query_y)

    h_ref = apply_fixed_program(model, query0, tuple(reference_program))
    ref_pred = model.decode_query(h_ref)
    behavior_delta = per_sample_output_mse(pred, ref_pred)
    equiv = behavior_delta <= tolerance
    return float(equiv.float().mean().item()), float(chosen_err.mean().item())

def apply_fixed_program(model: TinyReasoner, H: torch.Tensor, program: Tuple[int, ...]) -> torch.Tensor:
    out = H
    for action in program:
        out = model.transition(out, int(action))
    return out

def evaluate_exact_variable_program_search(
    model, task_source, device, batch_size: int = 16, max_depth: int = 4
):
    """Exact support-selected oracle over the same primitive+STOP search space as PUCT.

    Enumerates all sum_{d=0..D} 4^d programs. A program may stop at any depth
    because every shorter tuple is represented explicitly. Selection uses only
    support/demo error; the selected program is then transferred to the query.
    """
    model.eval()
    programs = enumerate_primitive_programs(max_depth)
    out = {}
    for name, true_seq in task_source.COMPOSITIONS.items():
        demo_x, demo_y, query_x, query_y = task_source.sample_composition(name, batch_size, device)
        support0 = model.encode_query(demo_x)
        target_support = model.encode_query(demo_y)
        query0 = model.encode_query(query_x)

        best_err = torch.full((batch_size,), float('inf'), device=device)
        best_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        for idx, program in enumerate(programs):
            hs = apply_fixed_program(model, support0, program)
            err = goal_error(model, hs, target_support)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_idx = torch.where(better, torch.full_like(best_idx, idx), best_idx)

        pred_q = torch.empty_like(query_x)
        selected = []
        for idx, program in enumerate(programs):
            mask = best_idx == idx
            if mask.any():
                hq = apply_fixed_program(model, query0[mask], program)
                pred_q[mask] = model.decode_query(hq)
                selected.extend([program] * int(mask.sum().item()))
        search_mse = F.mse_loss(pred_q, query_y)

        oracle_h = apply_fixed_program(model, query0, tuple(true_seq))
        oracle_mse = F.mse_loss(model.decode_query(oracle_h), query_y)
        exact_string = torch.tensor(
            [programs[int(i)] == tuple(true_seq) for i in best_idx.detach().cpu().tolist()],
            device=device, dtype=torch.float32,
        ).mean()
        out[name] = {
            'search_mse': float(search_mse.item()),
            'oracle_mse': float(oracle_mse.item()),
            'exact_string': float(exact_string.item()),
            'mean_support_mse': float(best_err.mean().item()),
        }
    return out
