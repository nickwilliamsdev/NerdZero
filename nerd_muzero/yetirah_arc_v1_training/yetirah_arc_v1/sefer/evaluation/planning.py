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
from sefer.evaluation.programs import functional_equivalence_rate
from sefer.planning.puct import SearchConfig, goal_puct_search, root_action_diagnostics

def evaluate_muzero_program_search(
    model, task_source, device, batch_size: int = 32, simulations: int = 96, max_depth: int = 4
):
    model.eval()
    out = {}
    for name, seq in task_source.COMPOSITIONS.items():
        demo_x, demo_y, query_x, query_y = task_source.sample_composition(name, batch_size, device)
        target_support = model.encode_query(demo_y)
        support = model.encode_query(demo_x)
        query = model.encode_query(query_x)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        actions = []
        for decision_idx in range(max_depth):
            remaining = max_depth - decision_idx
            step_actions = []
            for b in range(batch_size):
                if stopped[b]:
                    step_actions.append(4)
                    continue
                a, _, _ = goal_puct_search(
                    model, support[b:b+1], target_support[b:b+1],
                    SearchConfig(simulations=simulations, max_depth=remaining, action_limit=5)
                )
                step_actions.append(a)
            action = torch.tensor(step_actions, device=device, dtype=torch.long)
            actions.append(action)
            support = apply_program_actions(model, support, action)
            query = apply_program_actions(model, query, action)
            stopped |= action == 4
        chosen = torch.stack(actions, dim=1)
        target = torch.full((batch_size, max_depth), 4, device=device, dtype=torch.long)
        target[:, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
        exact = (chosen == target).all(dim=-1).float().mean()
        mse = F.mse_loss(model.decode_query(query), query_y)
        functional, _ = functional_equivalence_rate(
            model, model.encode_query(query_x), query_y, chosen, tuple(seq), tolerance=0.03
        )
        out[name] = (float(exact.item()), float(mse.item()), functional)
    return out

def evaluate_root_search_diagnostics(
    model, task_source, device, simulations: int = 384, samples_per_task: int = 8, max_depth: int = 4
):
    """Compare learned-prior and PUCT first actions to the exact Bellman-best root action."""
    model.eval()
    out = {}
    for name in task_source.COMPOSITIONS:
        demo_x, demo_y, _, _ = task_source.sample_composition(name, samples_per_task, device)
        support = model.encode_query(demo_x)
        target = model.encode_query(demo_y)
        prior_hits = 0
        puct_hits = 0
        selected_value_ratio = []
        for b in range(samples_per_task):
            rows = root_action_diagnostics(
                model, support[b:b+1], target[b:b+1], remaining_steps=max_depth, action_limit=5
            )
            priors = torch.tensor([r[1] for r in rows])
            vals = torch.tensor([r[2] for r in rows])
            vmax = float(vals.max().item())
            # Functional equivalences can produce ties, so count any nearly-best action.
            best = vals >= (vmax - 1e-5)
            prior_a = int(priors.argmax().item())
            prior_hits += int(bool(best[prior_a]))
            puct_a, _, _ = goal_puct_search(
                model, support[b:b+1], target[b:b+1],
                SearchConfig(simulations=simulations, max_depth=max_depth, action_limit=5)
            )
            puct_hits += int(bool(best[puct_a]))
            selected_value_ratio.append(float(vals[puct_a].item()) / max(vmax, 1e-8))
        out[name] = {
            'prior_best_rate': prior_hits / samples_per_task,
            'puct_best_rate': puct_hits / samples_per_task,
            'puct_value_ratio': sum(selected_value_ratio) / len(selected_value_ratio),
        }
    return out
