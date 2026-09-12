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

from sefer.algebra.rollout import apply_selected_actions, demo_guided_two_step_search, deterministic_operator_rollout, deterministic_primitive_rollout_with_halt, operator_separation_loss
from sefer.controllers.reasoner import TinyReasoner
from sefer.tasks.synthetic_algebra import SyntheticTaskBatch

def evaluate_held_out(
    model: TinyReasoner,
    task_source: SyntheticTaskBatch,
    device,
    batch_size: int = 1024,
):
    """Evaluate factorized rule inference, direct baseline, and query autoencoding."""
    model.eval()
    demo_x, demo_y, query_x, query_y, task_ids = task_source.sample(batch_size, device)
    rule = model.encode_rule(demo_x, demo_y)
    H0 = model.encode_query(query_x)
    direct = model.decode_direct(H0, rule)
    identity = model.decode_query(H0)
    task_logits = model.task_logits_rule(rule)
    direct_per = F.mse_loss(direct, query_y, reduction="none").mean(dim=-1)
    identity_per = F.mse_loss(identity, query_x, reduction="none").mean(dim=-1)
    task_acc = (task_logits.argmax(dim=-1) == task_ids).float().mean()
    per_task = {}
    for task_id, name in enumerate(task_source.TRAIN_NAMES):
        mask = task_ids == task_id
        if mask.any():
            per_task[name] = (
                float(direct_per[mask].mean().item()),
                float(identity_per[mask].mean().item()),
                float((task_logits[mask].argmax(dim=-1) == task_ids[mask]).float().mean().item()),
                int(mask.sum().item()),
            )
    return {
        "direct_mse": float(direct_per.mean().item()),
        "identity_mse": float(identity_per.mean().item()),
        "task_acc_rule": float(task_acc.item()),
        "per_task": per_task,
    }

def _sequence_summary(sequences: torch.Tensor, top_k: int = 5):
    patterns = Counter(tuple(int(v) for v in row) for row in sequences.cpu().tolist())
    total = max(sum(patterns.values()), 1)
    return [(pattern, count, count / total) for pattern, count in patterns.most_common(top_k)]

def evaluate_operator_rollout(model: TinyReasoner, task_source: SyntheticTaskBatch, device, batch_size: int = 1024, rollout_steps: int = 2):
    model.eval()
    demo_x, demo_y, query_x, query_y, task_ids = task_source.sample(batch_size, device)
    rule = model.encode_rule(demo_x, demo_y)
    H0 = model.encode_query(query_x)
    direct = model.decode_direct(H0, rule)
    H, sequences = deterministic_primitive_rollout_with_halt(model, rule, H0, max_steps=rollout_steps)
    pred = model.decode_query(H)
    direct_per = F.mse_loss(direct, query_y, reduction="none").mean(dim=-1)
    rollout_per = F.mse_loss(pred, query_y, reduction="none").mean(dim=-1)
    target_energy = query_y.pow(2).mean(dim=-1).clamp_min(1e-6)
    first_actions = sequences[:, 0]
    first_action_acc = (first_actions == task_ids).float().mean()
    # Measure completion separately from first-action selection using the known
    # anchored primitive transition, so a bad first action cannot contaminate the
    # HALT diagnostic.
    H_oracle = apply_selected_actions(model, H0, task_ids)
    post_logits, _ = model.policy_value(H_oracle, rule, primitive_phase=1)
    post_halt_acc = (post_logits.argmax(dim=-1) == model.HALT_ACTION).float().mean()
    usage = torch.bincount(first_actions, minlength=model.num_actions).float() / batch_size
    top_usage = torch.topk(usage, k=min(8, model.num_actions))
    sep, pair = operator_separation_loss(model, H0, max_batch=min(32, batch_size))
    per_task = {}
    for task_id, name in enumerate(task_source.TRAIN_NAMES):
        mask = task_ids == task_id
        if mask.any():
            per_task[name] = {
                "direct_mse": float(direct_per[mask].mean().item()),
                "rollout_mse": float(rollout_per[mask].mean().item()),
                "rollout_nmse": float((rollout_per[mask] / target_energy[mask]).mean().item()),
                "sequences": _sequence_summary(sequences[mask]),
            }
    return {
        "direct_mse": float(direct_per.mean().item()),
        "rollout_mse": float(rollout_per.mean().item()),
        "rollout_nmse": float((rollout_per / target_energy).mean().item()),
        "op_pair": float(pair.item()),
        "sep_loss": float(sep.item()),
        "top_ops": top_usage.indices.tolist(),
        "top_usage": top_usage.values.tolist(),
        "first_action_acc": float(first_action_acc.item()),
        "post_halt_acc": float(post_halt_acc.item()),
        "per_task": per_task,
    }

def evaluate_compositional_generalization(model, task_source, device, batch_size: int = 128):
    model.eval()
    results = {}
    for name, primitive_sequence in task_source.COMPOSITIONS.items():
        support_x, support_y, probe_x, probe_y, query_x, query_y = task_source.sample_composition_triplet(name, batch_size, device)
        rule = model.encode_rule(support_x, support_y)
        H0 = model.encode_query(query_x)
        direct_pred = model.decode_direct(H0, rule)
        direct_per = F.mse_loss(direct_pred, query_y, reduction="none").mean(dim=-1)
        H_greedy, greedy_sequences = deterministic_operator_rollout(model, rule, H0, steps=2)
        greedy_pred = model.decode_query(H_greedy)
        greedy_per = F.mse_loss(greedy_pred, query_y, reduction="none").mean(dim=-1)
        search_pred, search_sequences, probe_mse = demo_guided_two_step_search(model, probe_x, probe_y, query_x)
        search_per = F.mse_loss(search_pred, query_y, reduction="none").mean(dim=-1)

        # Apply the *known anchored primitive program* directly. If this is low,
        # the learned operators themselves form the intended algebra regardless
        # of whether the OOD policy/search identifies that program.
        H_oracle = H0
        for action in primitive_sequence:
            H_oracle = model.core.apply_operator(H_oracle, int(action))
        oracle_pred = model.decode_query(H_oracle)
        oracle_per = F.mse_loss(oracle_pred, query_y, reduction="none").mean(dim=-1)
        target_energy = query_y.pow(2).mean(dim=-1).clamp_min(1e-6)
        results[name] = {
            "primitive_sequence": primitive_sequence,
            "direct_mse": float(direct_per.mean().item()),
            "greedy_mse": float(greedy_per.mean().item()),
            "search_mse": float(search_per.mean().item()),
            "oracle_program_mse": float(oracle_per.mean().item()),
            "oracle_program_nmse": float((oracle_per / target_energy).mean().item()),
            "search_nmse": float((search_per / target_energy).mean().item()),
            "search_improvement": float((direct_per.mean() - search_per.mean()).item()),
            "probe_mse": float(probe_mse.mean().item()),
            "greedy_sequences": _sequence_summary(greedy_sequences),
            "search_sequences": _sequence_summary(search_sequences),
        }
    return results
