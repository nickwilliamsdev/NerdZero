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


class CPPN(nn.Module):
    """
    Geometry -> edge weight/gate.

    For each ordered pair of 5D substrate locations i,j we feed:
        s_i
        s_j
        s_i - s_j
        s_i * s_j
        ||s_i - s_j||

    The CPPN outputs:
        raw_weight
        raw_gate
    """

    def __init__(self, coord_dim: int = 5, hidden: int = 32):
        super().__init__()
        in_dim = coord_dim * 4 + 1
        out = nn.Linear(hidden, 2)
        nn.init.normal_(out.weight, std=0.02)
        nn.init.zeros_(out.bias)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            out,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [N, D]
        returns adjacency: [N, N]
        """
        n, d = coords.shape
        si = coords[:, None, :].expand(n, n, d)
        sj = coords[None, :, :].expand(n, n, d)
        diff = si - sj
        prod = si * sj
        dist = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)

        x = torch.cat([si, sj, diff, prod, dist], dim=-1)
        out = self.net(x)

        weight = torch.tanh(out[..., 0])
        gate = torch.sigmoid(out[..., 1])

        A = weight * gate

        # Remove self-loops here; residual connection exists elsewhere.
        eye = torch.eye(n, device=coords.device, dtype=coords.dtype)
        A = A * (1.0 - eye)

        # Normalize for stable recursive application.
        denom = A.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return A / denom
