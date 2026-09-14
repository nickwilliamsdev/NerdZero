from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from yetirah_arc_v1_training.yetirah_arc_v1.sefer.algebra.cppn import CPPN
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.algebra.operator_hypernet import OperatorHyperNet
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.geometry.hypercube import make_hypercube_vertices


class YetirahCore(nn.Module):
    """32-node, 5D hypercube substrate with generated operators.

    During algebra learning operators are generated dynamically. Once the
    algebra is frozen, :meth:`materialize_operator_bank` caches all 22
    transports and feature-affine transforms so program training/search reduces
    to batched tensor gathers and matrix multiplies.
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
        self.register_buffer("coords", make_hypercube_vertices(coord_dim), persistent=True)
        self.cppn = CPPN(coord_dim, cppn_hidden)
        self.operator_codes = nn.Parameter(
            torch.randn(operator_count, operator_code_dim) / math.sqrt(operator_code_dim)
        )
        self.op_hyper = OperatorHyperNet(
            coord_dim=coord_dim, code_dim=operator_code_dim,
            hidden_dim=op_hidden, node_dim=node_dim,
        )
        # Non-persistent caches: regenerated from the learned algebra after load.
        self.register_buffer("_transport_bank", torch.empty(0), persistent=False)
        self.register_buffer("_scale_bank", torch.empty(0), persistent=False)
        self.register_buffer("_bias_bank", torch.empty(0), persistent=False)

    @property
    def has_materialized_operator_bank(self) -> bool:
        return self._transport_bank.numel() > 0

    def adjacency(self) -> torch.Tensor:
        return self.cppn(self.coords)

    @torch.no_grad()
    def materialize_operator_bank(self) -> None:
        """Cache all generated operator laws after the algebra is frozen."""
        transports, scales, biases = [], [], []
        for k in range(self.operator_count):
            code = self.operator_codes[k]
            transports.append(self.op_hyper.transport(self.coords, code).detach())
            scale, bias = self.op_hyper.feature_affine(code)
            scales.append(scale.detach())
            biases.append(bias.detach())
        self._transport_bank = torch.stack(transports, dim=0).contiguous()
        self._scale_bank = torch.stack(scales, dim=0).contiguous()
        self._bias_bank = torch.stack(biases, dim=0).contiguous()

    def clear_operator_bank(self) -> None:
        device, dtype = self.coords.device, self.coords.dtype
        self._transport_bank = torch.empty(0, device=device, dtype=dtype)
        self._scale_bank = torch.empty(0, device=device, dtype=dtype)
        self._bias_bank = torch.empty(0, device=device, dtype=dtype)

    def _law(self, operator_idx: int):
        if self.has_materialized_operator_bank:
            return (self._transport_bank[operator_idx],
                    self._scale_bank[operator_idx],
                    self._bias_bank[operator_idx])
        code = self.operator_codes[operator_idx]
        transport = self.op_hyper.transport(self.coords, code)
        scale, bias = self.op_hyper.feature_affine(code)
        return transport, scale, bias

    def apply_operator(
        self, H: torch.Tensor, operator_idx: int,
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # adjacency is retained for API compatibility; transport operators do not use it.
        transport, scale, bias = self._law(int(operator_idx))
        # Ellipsis lets this operate on [B,N,D] and batched program banks
        # such as [B,P,N,D] without reshaping in Python.
        moved = torch.einsum("ij,...jd->...id", transport, H)
        return moved * scale + bias

    def apply_operator_batch(self, H: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply a different operator to each item with no per-sample Python loop.

        Fast path uses the frozen operator bank. During algebra learning we group
        by the small number of unique actions, which is still far cheaper than a
        loop over every batch element.
        """
        actions = actions.long()
        if self.has_materialized_operator_bank:
            T = self._transport_bank.index_select(0, actions)          # [B,N,N]
            s = self._scale_bank.index_select(0, actions)              # [B,D]
            b = self._bias_bank.index_select(0, actions)               # [B,D]
            moved = torch.bmm(T, H)
            return moved * s[:, None, :] + b[:, None, :]

        out = torch.empty_like(H)
        for action in actions.unique(sorted=False).tolist():
            mask = actions == int(action)
            out[mask] = self.apply_operator(H[mask], int(action))
        return out

    def apply_all_operators(self, H: torch.Tensor) -> torch.Tensor:
        """Return [B,K,N,D] states for every operator."""
        if self.has_materialized_operator_bank:
            # [K,N,N] x [B,N,D] -> [B,K,N,D]
            moved = torch.einsum("kij,bjd->bkid", self._transport_bank, H)
            return (moved * self._scale_bank[None, :, None, :]
                    + self._bias_bank[None, :, None, :])
        return torch.stack([self.apply_operator(H, k) for k in range(self.operator_count)], dim=1)
