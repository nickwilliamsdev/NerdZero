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


class SmallDeltaStateEncoder(nn.Module):
    """Small DeltaNet-style fast-weight encoder for goal-conditioned state.

    The program controller already receives five 32-wide relational views
    (current, target, delta, sum, product) plus seven scalar relation/horizon
    features.  We treat those views as a short token sequence and build a
    per-sample fast-weight matrix with the delta rule

        W <- W + beta * (v - W k) outer k

    before reading the final memory with a learned query.  W is ephemeral: it
    is rebuilt from the current/goal pair on every policy/value call, so PUCT
    remains Markov in the explicit substrate state while gaining a compact
    adaptive relational representation.
    """
    def __init__(self, vector_dim: int = 32, state_dim: int = 32):
        super().__init__()
        self.vector_dim = vector_dim
        self.state_dim = state_dim
        self.token_proj = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.GELU(),
            nn.Linear(64, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.key = nn.Linear(state_dim, state_dim, bias=False)
        self.value = nn.Linear(state_dim, state_dim, bias=False)
        self.query = nn.Linear(state_dim, state_dim, bias=False)
        self.beta = nn.Sequential(nn.Linear(state_dim, 1), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, state_dim),
            nn.LayerNorm(state_dim),
        )
        # v29: start the fast-weight branch as a modest residual contribution.
        # The controller can increase this gate if history-like relational memory
        # helps, but the explicit geometric features remain the stable baseline.
        self.output_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, goal_feat: torch.Tensor) -> torch.Tensor:
        # goal_feat layout: five vector_dim blocks + 7 relation/horizon scalars.
        B = goal_feat.shape[0]
        main = goal_feat[:, :5 * self.vector_dim].reshape(B, 5, self.vector_dim)
        tail = goal_feat[:, 5 * self.vector_dim:]
        tail_token = F.pad(tail, (0, self.vector_dim - tail.shape[-1]))[:, None, :]
        tokens = torch.cat([main, tail_token], dim=1)  # [B,6,vector_dim]
        z = self.token_proj(tokens)                    # [B,6,D]

        W = torch.zeros(B, self.state_dim, self.state_dim, device=z.device, dtype=z.dtype)
        reads = []
        for t in range(z.shape[1]):
            h = z[:, t]
            k = F.normalize(self.key(h), dim=-1)
            v = self.value(h)
            q = F.normalize(self.query(h), dim=-1)
            old = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
            residual = v - old
            b = self.beta(h)
            W = W + b.unsqueeze(-1) * residual.unsqueeze(-1) * k.unsqueeze(1)
            reads.append(torch.bmm(W, q.unsqueeze(-1)).squeeze(-1))

        read = reads[-1]
        pooled = z.mean(dim=1)
        state = self.out(torch.cat([read, pooled], dim=-1))
        return torch.sigmoid(self.output_gate) * state
