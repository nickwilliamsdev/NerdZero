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

from sefer.algebra.cppn import CPPN
from sefer.algebra.operator_hypernet import OperatorHyperNet
from sefer.geometry.hypercube import make_hypercube_vertices

class YetirahCore(nn.Module):
    """
    32-node, 5D hypercube cognitive substrate with 22 generated operators.
    """

    def __init__(
        self,
        coord_dim: int = 5,
        node_dim: int = 32,
        operator_count: int = 22,
        operator_code_dim: int = 8,
        cppn_hidden: int = 32,
        op_hidden: int = 32,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.node_dim = node_dim
        self.operator_count = operator_count
        coords: torch.Tensor
        coords = make_hypercube_vertices(coord_dim)
        self.register_buffer(
            "coords",
            coords,
            persistent=True,
        )

        self.cppn = CPPN(coord_dim, cppn_hidden)

        self.operator_codes = nn.Parameter(
            torch.randn(operator_count, operator_code_dim) / math.sqrt(operator_code_dim)
        )

        self.op_hyper = OperatorHyperNet(
            coord_dim=coord_dim,
            code_dim=operator_code_dim,
            hidden_dim=op_hidden,
            node_dim=node_dim,
        )


    def adjacency(self) -> torch.Tensor:
        return self.cppn(self.coords)

    def apply_operator(
        self,
        H: torch.Tensor,
        operator_idx: int,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        H: [B,N,D]
        """
        if adjacency is None:
            adjacency = self.adjacency()

        code = self.operator_codes[operator_idx]
        transport = self.op_hyper.transport(self.coords, code)
        scale, bias = self.op_hyper.feature_affine(code)
        moved = torch.einsum("ij,bjd->bid", transport, H)
        return moved * scale[None, None, :] + bias[None, None, :]

    def apply_operator_batch(
        self,
        H: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Different action per batch item.
        H: [B,N,D]
        actions: [B]
        """
        A = self.adjacency()
        outs = []
        for b in range(H.shape[0]):
            outs.append(self.apply_operator(H[b:b+1], int(actions[b]), A))
        return torch.cat(outs, dim=0)
