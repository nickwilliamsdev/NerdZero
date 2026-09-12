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

from sefer.evolution.neat_runtime import EvolvedTorchCPPN

class OperatorHyperNet(nn.Module):
    """Generate an operator-specific transport law over the 32-vertex substrate.

    v10 changes the operator from FiLM over a shared message graph to a compact
    code-conditioned *transport matrix*.  This is the form needed for a real
    algebra over a position-preserving query representation: roll/flip can move
    information between vertices, while a generated signed feature transform
    can implement operations such as negate.
    """

    def __init__(
        self,
        coord_dim: int = 5,
        code_dim: int = 8,
        hidden_dim: int = 64,
        node_dim: int = 32,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.code_dim = code_dim
        self.node_dim = node_dim
        self.evolved_cppn = None
        self.neat_genome = None
        self.neat_config = None

        # Edge law gets source/destination geometry plus normalized vertex index.
        edge_in = coord_dim * 4 + 4 + code_dim
        self.edge_net = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.edge_net[-1].weight, std=0.02)
        nn.init.zeros_(self.edge_net[-1].bias)

        # Feature law is deliberately signed.  The previous GELU + positive
        # alpha parameterization made an exact negation unnecessarily difficult.
        self.feature_net = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, node_dim * 2),
        )
        nn.init.zeros_(self.feature_net[-1].weight)
        nn.init.zeros_(self.feature_net[-1].bias)

        # Start close to identity transport, then let each code move away.
        self.diag_bias_net = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.diag_bias_net[-1].weight)
        nn.init.zeros_(self.diag_bias_net[-1].bias)

    def install_evolved_cppn(self, genome, config, create_cppn_fn) -> None:
        self.evolved_cppn = EvolvedTorchCPPN(genome, config, create_cppn_fn)
        self.neat_genome = genome
        self.neat_config = config
        # The old MLP remains in the state_dict only as a v20 comparison/control;
        # it is no longer used after a winner is installed.
        for p in self.edge_net.parameters():
            p.requires_grad_(False)
        for p in self.diag_bias_net.parameters():
            p.requires_grad_(False)

    def _edge_features(self, coords: torch.Tensor, code: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = coords.shape[0]
        dst = coords[:, None, :].expand(n, n, -1)
        src = coords[None, :, :].expand(n, n, -1)
        diff = dst - src
        prod = dst * src
        dist = diff.pow(2).sum(dim=-1).sqrt()
        idx = torch.linspace(-1.0, 1.0, n, device=coords.device, dtype=coords.dtype)
        dst_idx = idx[:, None].expand(n, n)
        src_idx = idx[None, :].expand(n, n)
        feats: Dict[str, torch.Tensor] = {}
        for i in range(self.coord_dim):
            feats[f"dst_{i}"] = dst[..., i]
            feats[f"src_{i}"] = src[..., i]
            feats[f"diff_{i}"] = diff[..., i]
            feats[f"prod_{i}"] = prod[..., i]
        feats["dist"] = dist
        feats["dst_idx"] = dst_idx
        feats["src_idx"] = src_idx
        feats["idx_diff"] = dst_idx - src_idx
        feats["idx_sum"] = dst_idx + src_idx
        feats["is_diag"] = torch.eye(n, device=coords.device, dtype=coords.dtype)

        # Generic cyclic coordinates. These expose the ring topology without
        # hard-coding a particular primitive such as roll+1.
        phase = torch.arange(n, device=coords.device, dtype=coords.dtype) * (2.0 * math.pi / n)
        dst_phase = phase[:, None].expand(n, n)
        src_phase = phase[None, :].expand(n, n)
        rel_phase = src_phase - dst_phase
        feats["dst_phase_sin"] = torch.sin(dst_phase)
        feats["dst_phase_cos"] = torch.cos(dst_phase)
        feats["src_phase_sin"] = torch.sin(src_phase)
        feats["src_phase_cos"] = torch.cos(src_phase)
        feats["rel_phase_sin"] = torch.sin(rel_phase)
        feats["rel_phase_cos"] = torch.cos(rel_phase)
        c = code.detach()
        for i in range(self.code_dim):
            feats[f"op_{i}"] = torch.ones_like(dist) * c[i]
        return feats

    def transport(self, coords: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """Row-stochastic destination<-source transport from fixed MLP or evolved CPPN."""
        if self.evolved_cppn is not None:
            logits = self.evolved_cppn.edge_logits(self._edge_features(coords, code))
            if logits.ndim > 2:
                logits = logits.squeeze(-1)
            return torch.softmax(logits, dim=-1)

        # v20 bootstrap/control transport before a NEAT winner is installed.
        n = coords.shape[0]
        dst = coords[:, None, :].expand(n, n, -1)
        src = coords[None, :, :].expand(n, n, -1)
        diff = dst - src
        prod = dst * src
        dist = diff.pow(2).sum(dim=-1, keepdim=True).sqrt()
        idx = torch.linspace(-1.0, 1.0, n, device=coords.device, dtype=coords.dtype)
        dst_idx = idx[:, None, None].expand(n, n, 1)
        src_idx = idx[None, :, None].expand(n, n, 1)
        idx_diff = dst_idx - src_idx
        c = code.view(1, 1, -1).expand(n, n, -1)
        feat = torch.cat([dst, src, diff, prod, dist, dst_idx, src_idx, idx_diff, c], dim=-1)
        logits = self.edge_net(feat).squeeze(-1)
        diag_bias = 6.0 * torch.tanh(self.diag_bias_net(code).squeeze())
        logits = logits + diag_bias * torch.eye(n, device=coords.device, dtype=coords.dtype)
        return torch.softmax(logits, dim=-1)

    def feature_affine(self, code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.feature_net(code)
        scale_raw, bias_raw = raw.chunk(2, dim=-1)
        # Identity at initialization; can smoothly reach negative scale.
        scale = 1.0 + 2.0 * torch.tanh(scale_raw)
        bias = 0.10 * torch.tanh(bias_raw)
        return scale, bias
