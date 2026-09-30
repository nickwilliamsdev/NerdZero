
from __future__ import annotations

ARC_TRAINER_PATCH_ID = "arc-scratch-v1"

import copy
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sefer.controllers.arc_reasoner import ARCReasoner
from sefer.representation.arc_grid import arc_grid_loss
from sefer.evaluation.arc import evaluate_arc, evaluate_arc_direct
from sefer.evolution.arc_neat_outer import evolve_arc_cppn, initialize_arc_cppn_from_scratch, restore_arc_cppn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_arc_from_scratch(model: ARCReasoner, cfg):
    """Fresh random neural parameters + newly created ARC CPPN, no old weights."""
    initialize_arc_cppn_from_scratch(model, cfg)
    model.ensure_operator_bank()
    print("ARC initialization: fresh neural parameters + new ARC-native CPPN")


def _decode_loss(model, H, target, target_shape):
    c, h, w = model.decode_grid(H)
    color_loss, stats = arc_grid_loss(
        c, h, w, target, target_shape,
        foreground_boost=model.cfg.foreground_boost,
        balance_mix=model.cfg.color_balance_mix,
    )
    return color_loss, stats["shape_loss"], stats


def _set_requires_grad(module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


def _program_stage(cfg, step: int) -> Tuple[int, int, int]:
    """Return (stage_index, active_operator_count, max_depth)."""
    progress = (step - 1) / max(cfg.program_steps - 1, 1)
    b0, b1, b2 = cfg.program_stage_fractions
    if progress < b0:
        idx = 0
    elif progress < b1:
        idx = 1
    elif progress < b2:
        idx = 2
    else:
        idx = 3
    active = min(int(cfg.program_stage_active_ops[idx]), int(cfg.operator_count))
    depth = min(int(cfg.program_stage_depths[idx]), int(cfg.max_program_steps))
    return idx, max(active, 1), max(depth, 1)


@torch.no_grad()
def _beam_teacher_actions(model, state, rule, target, active_indices, lookahead: int, width: int):
    """Training-only shallow lookahead teacher.

    Returns the first action of the best program found within ``lookahead``
    operator applications. STOP is represented by keeping the current state.
    The target latent is used only to construct this supervision target.
    """
    B, N, D = state.shape
    device = state.device
    stop = model.cfg.operator_count
    active_indices = active_indices.to(device=device, dtype=torch.long)
    A = int(active_indices.numel())
    width = max(int(width), 1)
    lookahead = max(int(lookahead), 1)

    # Global best includes immediate STOP.
    best_score = (state - target).pow(2).mean(dim=(1, 2))
    best_first = torch.full((B,), stop, device=device, dtype=torch.long)

    frontier = state[:, None]  # [B,K,N,D]
    first = torch.full((B, 1), stop, device=device, dtype=torch.long)
    for depth in range(lookahead):
        K = frontier.shape[1]
        flat_state = frontier.reshape(B * K, N, D)
        flat_rule = rule[:, None].expand(B, K, rule.shape[-1]).reshape(B * K, -1)
        all_ops = model.operator_bank.apply_all(flat_state, flat_rule)[:, active_indices]
        cand = all_ops.reshape(B, K * A, N, D)

