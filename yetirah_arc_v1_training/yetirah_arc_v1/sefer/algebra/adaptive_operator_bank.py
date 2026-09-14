from __future__ import annotations

import math
import torch
import torch.nn as nn


class AdaptiveOperatorBank(nn.Module):
    """Low-rank ARC adaptation around a frozen Yetirah operator bank.

    The validated synthetic algebra remains immutable.  ARC learns a small
    residual in log-transport space plus feature-affine residuals.  With all
    residual parameters at zero this module is exactly the supplied base bank.
    """

    def __init__(
        self,
        base_transport: torch.Tensor,
        base_scale: torch.Tensor,
        base_bias: torch.Tensor,
        rank: int = 4,
        transport_residual_scale: float = 0.35,
        feature_residual_scale: float = 0.10,
    ):
        super().__init__()
        if base_transport.ndim != 3:
            raise ValueError("base_transport must be [K,N,N]")
        K, N, _ = base_transport.shape
        D = base_scale.shape[-1]
        self.K, self.N, self.D, self.rank = K, N, D, rank
        self.transport_residual_scale = transport_residual_scale
        self.feature_residual_scale = feature_residual_scale
        self.register_buffer("base_log_transport", base_transport.clamp_min(1e-8).log())
        self.register_buffer("base_scale", base_scale)
        self.register_buffer("base_bias", base_bias)
        # Zero V makes the transport exactly base at initialization; U can be
        # random so useful residual gradients reach V immediately.
        self.u = nn.Parameter(torch.randn(K, N, rank) * 0.02)
        self.v = nn.Parameter(torch.zeros(K, N, rank))
        self.scale_delta = nn.Parameter(torch.zeros(K, D))
        self.bias_delta = nn.Parameter(torch.zeros(K, D))

    def transport_bank(self) -> torch.Tensor:
        delta = torch.einsum("knr,kmr->knm", self.u, self.v) / math.sqrt(max(self.rank, 1))
        logits = self.base_log_transport + self.transport_residual_scale * delta
        return torch.softmax(logits, dim=-1)

    def feature_bank(self):
        scale = self.base_scale * (1.0 + self.feature_residual_scale * torch.tanh(self.scale_delta))
        bias = self.base_bias + self.feature_residual_scale * torch.tanh(self.bias_delta)
        return scale, bias

    def apply_all(self, H: torch.Tensor) -> torch.Tensor:
        T = self.transport_bank()
        scale, bias = self.feature_bank()
        moved = torch.einsum("kij,bjd->bkid", T, H)
        return moved * scale[None, :, None, :] + bias[None, :, None, :]

    def apply_actions(self, H: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        T = self.transport_bank().index_select(0, actions.long())
        scale, bias = self.feature_bank()
        scale = scale.index_select(0, actions.long())
        bias = bias.index_select(0, actions.long())
        moved = torch.bmm(T, H)
        return moved * scale[:, None, :] + bias[:, None, :]

    def regularization(self) -> torch.Tensor:
        return (
            self.v.pow(2).mean()
            + 0.25 * self.scale_delta.pow(2).mean()
            + 0.25 * self.bias_delta.pow(2).mean()
        )
