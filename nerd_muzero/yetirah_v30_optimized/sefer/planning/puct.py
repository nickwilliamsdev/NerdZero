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
from sefer.planning.bellman import exact_horizon_value_target, goal_error, terminal_goal_score

class SearchConfig:
    def __init__(self,
                 simulations: int = 96,
                 max_depth: int = 4,
                 action_limit: int =5) -> None:
        self.simulations = simulations
        self.max_depth = max_depth
        self.action_limit = action_limit
    c_puct: float = 1.5
    discount: float = 1.0
    terminal_beta: float = 2.0
    prior_uniform_mix: float = 0.10
    force_unvisited: bool = True
    use_transpositions: bool = True
    transposition_decimals: int = 4

class SearchNode:
    def __init__(
        self,
        state: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
        depth: int = 0,
    ):
        self.state = state
        self.prior = prior
        self.depth = depth
        self.visit = None
        self.value_sum = None
        self.children: Dict[int, "SearchNode"] = {}

    def init_actions(self, n_actions: int, device):
        if self.visit is None:
            self.visit = torch.zeros(n_actions, device=device)
            self.value_sum = torch.zeros(n_actions, device=device)

    def q(self):
        return self.value_sum / self.visit.clamp_min(1.0)

def latent_state_signature(state: torch.Tensor, remaining_steps: int, decimals: int = 4):
    """Stable approximate key for deterministic synthetic search states.

    The remaining horizon is part of the key because the same latent state has
    different planning value with different numbers of actions left.
    """
    scale = float(10 ** decimals)
    q = torch.round(state.detach().float().cpu() * scale).to(torch.int32).contiguous()
    return (int(remaining_steps), q.numpy().tobytes())

def goal_puct_search(
    model: TinyReasoner,
    root_state: torch.Tensor,
    target_state: torch.Tensor,
    cfg: SearchConfig,
) -> Tuple[int, torch.Tensor, float]:
    """Finite-horizon goal-conditioned PUCT over the learned operator algebra.

    v17 gives the policy/value the exact number of decisions remaining. Leaves
    at horizon zero are scored by actual terminal support-goal fit; earlier
    leaves use V_theta(s,g,h), trained against exact finite-horizon Bellman
    targets. There is deliberately no mixed-scale pseudo-reward in the backup.
    """
    assert root_state.shape[0] == 1 and target_state.shape[0] == 1
    device = root_state.device
    n_actions = cfg.action_limit

    root_logits, _ = model.program_policy_value(root_state, target_state, cfg.max_depth)
    root_prior = torch.softmax(root_logits[0, :n_actions], dim=-1)
    if cfg.prior_uniform_mix > 0.0:
        root_prior = (1.0 - cfg.prior_uniform_mix) * root_prior + cfg.prior_uniform_mix / n_actions
    root = SearchNode(root_state.clone(), root_prior, depth=0)
    root.init_actions(n_actions, device)
    transpositions = {}
    if cfg.use_transpositions:
        transpositions[latent_state_signature(root.state, cfg.max_depth, cfg.transposition_decimals)] = root

    for _ in range(cfg.simulations):
        node = root
        path: List[Tuple[SearchNode, int]] = []
        while True:
            remaining = max(cfg.max_depth - node.depth, 0)
            logits, leaf_v = model.program_policy_value(node.state, target_state, remaining)
            prior = torch.softmax(logits[0, :n_actions], dim=-1)
            if cfg.prior_uniform_mix > 0.0:
                prior = (1.0 - cfg.prior_uniform_mix) * prior + cfg.prior_uniform_mix / n_actions
            node.init_actions(n_actions, device)
            node.prior = prior

            if remaining <= 0:
                value = float(terminal_goal_score(
                    model, node.state, target_state, beta=cfg.terminal_beta
                ).item())
                break

            total_n = node.visit.sum()
            q = node.q()
            u = cfg.c_puct * prior * torch.sqrt(total_n + 1.0) / (1.0 + node.visit)
            # The learned prior becomes extremely sharp in this synthetic proof.
            # Guarantee every primitive is tested at least once at each node so
            # order-sensitive programs cannot be hidden by a bad prior.
            if cfg.force_unvisited and bool((node.visit == 0).any()):
                unvisited = torch.nonzero(node.visit == 0, as_tuple=False).squeeze(-1)
                action = int(unvisited[torch.argmax(prior[unvisited])].item())
            else:
                action = int(torch.argmax(q + u).item())

            path.append((node, action))
            # Action 4 is STOP: terminate immediately at the current state.
            if action == 4:
                value = float(terminal_goal_score(
                    model, node.state, target_state, beta=cfg.terminal_beta
                ).item())
                break
            if action not in node.children:
                next_state = model.transition(node.state, action)
                next_depth = node.depth + 1
                child_remaining = max(cfg.max_depth - next_depth, 0)
                if cfg.use_transpositions:
                    key = latent_state_signature(next_state, child_remaining, cfg.transposition_decimals)
                    child = transpositions.get(key)
                    if child is None:
                        child = SearchNode(next_state.clone(), depth=next_depth)
                        transpositions[key] = child
                    node.children[action] = child
                else:
                    node.children[action] = SearchNode(next_state.clone(), depth=next_depth)
            child = node.children[action]

            child_remaining = max(cfg.max_depth - child.depth, 0)
            if child.visit is None:
                child.init_actions(n_actions, device)
                if child_remaining <= 0:
                    value = float(terminal_goal_score(
                        model, child.state, target_state, beta=cfg.terminal_beta
                    ).item())
                else:
                    _, child_v = model.program_policy_value(
                        child.state, target_state, child_remaining
                    )
                    value = float(child_v.item())
                break
            node = child

        # Zero intermediate reward; back up predicted/observed terminal utility.
        G = value
        for parent, action in reversed(path):
            parent.visit[action] += 1.0
            parent.value_sum[action] += G
            G = cfg.discount * G

    probs = root.visit / root.visit.sum().clamp_min(1.0)
    action = int(torch.argmax(probs).item())
    return action, probs, float(goal_error(model, root_state, target_state).item())

def root_action_diagnostics(model, root_state, target_state, remaining_steps: int = 2, action_limit: int = 4):
    """Report learned prior and exact best continuation value per root action."""
    logits, _ = model.program_policy_value(root_state, target_state, remaining_steps)
    priors = torch.softmax(logits[0, :action_limit], dim=-1)
    rows = []
    for a in range(action_limit):
        if a == 4:
            exact_v = terminal_goal_score(model, root_state, target_state, beta=2.0)
        else:
            s1 = model.transition(root_state, a)
            exact_v = exact_horizon_value_target(
                model, s1, target_state, max(remaining_steps - 1, 0),
                action_limit=action_limit, beta=2.0
            )
        rows.append((a, float(priors[a].item()), float(exact_v.item())))
    return rows
