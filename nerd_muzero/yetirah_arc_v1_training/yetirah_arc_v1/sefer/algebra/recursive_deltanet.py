from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveDeltaNetCell(nn.Module):
    """Task-conditioned recurrent cell with an explicit fast-weight DeltaNet state.

    fast_state is [B,D,D].  At each reasoning step the cell predicts a key/value
    pair from the task + current state, computes the fast-weight prediction at
    that key, and applies a delta-rule update:

        error = value - W_fast @ key
        W_fast <- decay * W_fast + eta * outer(error, key)

    The updated fast matrix is then queried independently by every latent slot.
    """

    def __init__(self, state_dim: int, task_dim: int, heads: int = 4,
                 update_scale: float = 0.35, residual_scale: float = 0.50,
                 decay_min: float = 0.90, decay_max: float = 0.999):
        super().__init__()
        self.state_dim = int(state_dim)
        self.task_dim = int(task_dim)
        self.update_scale = float(update_scale)
        self.residual_scale = float(residual_scale)
        self.decay_min = float(decay_min)
        self.decay_max = float(decay_max)

        context_dim = self.state_dim + self.task_dim + 1
        self.key_proj = nn.Linear(context_dim, self.state_dim)
        self.value_proj = nn.Linear(context_dim, self.state_dim)
        self.lr_proj = nn.Linear(context_dim, 1)
        self.decay_proj = nn.Linear(context_dim, 1)

        self.query_norm = nn.LayerNorm(self.state_dim)
        self.query_proj = nn.Linear(self.state_dim, self.state_dim)
        self.task_to_slot = nn.Linear(self.task_dim, self.state_dim)

        self.slot_attn_norm = nn.LayerNorm(self.state_dim)
        self.slot_attn = nn.MultiheadAttention(
            self.state_dim, heads, batch_first=True
        )

        self.recurrent = nn.Sequential(
            nn.LayerNorm(self.state_dim * 3),
            nn.Linear(self.state_dim * 3, self.state_dim * 4),
            nn.GELU(),
            nn.Linear(self.state_dim * 4, self.state_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.state_dim * 2 + self.task_dim),
            nn.Linear(self.state_dim * 2 + self.task_dim, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, 1),
        )
        self.halt_head = nn.Sequential(
            nn.LayerNorm(self.state_dim + self.task_dim),
            nn.Linear(self.state_dim + self.task_dim, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, 1),
        )

    def init_fast_state(self, task: torch.Tensor) -> torch.Tensor:
        b = task.shape[0]
        return torch.zeros(
            b, self.state_dim, self.state_dim,
            device=task.device, dtype=task.dtype,
        )

    def forward(self, state: torch.Tensor, task: torch.Tensor,
                fast_state: torch.Tensor, step_fraction: float):
        b, n, d = state.shape
        pooled = state.mean(dim=1)
        sf = torch.full(
            (b, 1), float(step_fraction),
            device=state.device, dtype=state.dtype,
        )
        context = torch.cat([pooled, task, sf], dim=-1)

        key = F.normalize(self.key_proj(context), dim=-1)
        value = torch.tanh(self.value_proj(context))

        pred = torch.bmm(fast_state, key.unsqueeze(-1)).squeeze(-1)
        error = value - pred

        eta = torch.sigmoid(self.lr_proj(context)) * self.update_scale
        decay01 = torch.sigmoid(self.decay_proj(context))
        decay = self.decay_min + (self.decay_max - self.decay_min) * decay01

        delta = error.unsqueeze(-1) * key.unsqueeze(-2)
        fast_next = decay[:, :, None] * fast_state + eta[:, :, None] * delta

        q = F.normalize(self.query_proj(self.query_norm(state)), dim=-1)
        fast_delta = torch.einsum("bnd,bed->bne", q, fast_next)

        z = self.slot_attn_norm(state)
        mixed, _ = self.slot_attn(z, z, z, need_weights=False)
        task_slot = self.task_to_slot(task)[:, None, :].expand(-1, n, -1)

        recurrent_delta = self.recurrent(
            torch.cat([state, mixed, task_slot], dim=-1)
        )
        gate = torch.sigmoid(
            self.gate(torch.cat([
                state,
                fast_delta,
                task[:, None, :].expand(-1, n, -1),
            ], dim=-1))
        )

        proposal = fast_delta + recurrent_delta
        state_next = state + self.residual_scale * gate * torch.tanh(proposal)

        halt_logit = self.halt_head(
            torch.cat([state_next.mean(dim=1), task], dim=-1)
        ).squeeze(-1)

        stats = {
            "eta": eta.squeeze(-1),
            "decay": decay.squeeze(-1),
            "fast_norm": fast_next.square().mean(dim=(1, 2)).sqrt(),
            "gate": gate.mean(dim=(1, 2)),
            "halt_prob": torch.sigmoid(halt_logit),
        }
        return state_next, fast_next, halt_logit, stats
