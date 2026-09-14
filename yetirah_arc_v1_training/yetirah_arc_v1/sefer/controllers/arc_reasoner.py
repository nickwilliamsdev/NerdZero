from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yetirah_arc_v1_training.yetirah_arc_v1.sefer.algebra.core import YetirahCore
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.algebra.adaptive_operator_bank import AdaptiveOperatorBank
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.representation.arc_grid import ARCGridEncoder, ARCGridDecoder


class ARCReasoner(nn.Module):
    """ARC-v1 reasoner over the existing 32-node Yetirah substrate.

    Demonstrations -> rule embedding.
    Query grid -> 32 substrate slots.
    Rule + current slots -> a short program over 22 operators + STOP.
    The target grid is used only during training as a latent/output teacher.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.operator_count = cfg.operator_count
        self.stop_action = cfg.operator_count
        self.num_actions = cfg.operator_count + 1
        self.n_slots = cfg.n_slots
        self.node_dim = cfg.node_dim

        self.core = YetirahCore(
            coord_dim=cfg.coord_dim,
            node_dim=cfg.node_dim,
            operator_count=cfg.operator_count,
            operator_code_dim=cfg.operator_code_dim,
        )
        self.grid_encoder = ARCGridEncoder(
            max_size=cfg.max_grid_size,
            colors=cfg.color_count,
            cell_dim=cfg.cell_dim,
            slot_dim=cfg.node_dim,
            n_slots=cfg.n_slots,
            heads=cfg.attention_heads,
            dropout=cfg.codec_dropout,
        )
        self.grid_decoder = ARCGridDecoder(
            max_size=cfg.max_grid_size,
            colors=cfg.color_count,
            cell_dim=cfg.cell_dim,
            slot_dim=cfg.node_dim,
            heads=cfg.attention_heads,
            dropout=cfg.codec_dropout,
        )

        pair_dim = cfg.node_dim * 4 + 4
        self.demo_pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, 256),
            nn.GELU(),
            nn.Linear(256, cfg.rule_dim),
            nn.GELU(),
        )
        self.rule_encoder = nn.Sequential(
            nn.LayerNorm(cfg.rule_dim),
            nn.Linear(cfg.rule_dim, cfg.rule_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.rule_dim * 2, cfg.rule_dim),
        )

        flat_dim = cfg.n_slots * cfg.node_dim
        self.direct_rule_to_slots = nn.Sequential(
            nn.Linear(cfg.rule_dim, 256),
            nn.GELU(),
            nn.Linear(256, flat_dim),
        )
        self.direct_norm = nn.LayerNorm(cfg.node_dim)

        policy_in = flat_dim + cfg.rule_dim + cfg.node_dim + 2
        self.arc_program_controller = nn.Sequential(
            nn.Linear(policy_in, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_actions),
        )
        self.arc_program_value = nn.Sequential(
            nn.Linear(policy_in, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.operator_bank: AdaptiveOperatorBank | None = None

    def encode_grid(self, grid: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        return self.grid_encoder(grid, shapes)

    def decode_grid(self, H: torch.Tensor):
        return self.grid_decoder(H)

    def encode_rule(
        self,
        demos_x: torch.Tensor,
        demos_y: torch.Tensor,
        demos_x_shapes: torch.Tensor,
        demos_y_shapes: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, D, S, _ = demos_x.shape
        flat_x = demos_x.reshape(B * D, S, S)
        flat_y = demos_y.reshape(B * D, S, S)
        sx = demos_x_shapes.reshape(B * D, 2)
        sy = demos_y_shapes.reshape(B * D, 2)
        hx = self.encode_grid(flat_x, sx).mean(dim=1).reshape(B, D, self.node_dim)
        hy = self.encode_grid(flat_y, sy).mean(dim=1).reshape(B, D, self.node_dim)
        shape_feat = torch.cat([
            demos_x_shapes.float() / float(S),
            demos_y_shapes.float() / float(S),
        ], dim=-1)
        pair_feat = torch.cat([hx, hy, hy - hx, hx * hy, shape_feat], dim=-1)
        z = self.demo_pair_encoder(pair_feat)
        m = demo_mask.float()[..., None]
        pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.rule_encoder(pooled)

    def direct_transform(self, H: torch.Tensor, rule: torch.Tensor) -> torch.Tensor:
        delta = self.direct_rule_to_slots(rule).view(H.shape[0], self.n_slots, self.node_dim)
        return self.direct_norm(H + 0.25 * torch.tanh(delta))

    def _program_features(self, H: torch.Tensor, rule: torch.Tensor, step: int, max_steps: int):
        pooled = H.mean(dim=1)
        step_frac = torch.full(
            (H.shape[0], 1), float(step) / max(max_steps, 1), device=H.device, dtype=H.dtype
        )
        rem_frac = torch.full(
            (H.shape[0], 1), float(max_steps - step) / max(max_steps, 1), device=H.device, dtype=H.dtype
        )
        return torch.cat([H.flatten(1), rule, pooled, step_frac, rem_frac], dim=-1)

    def program_policy_value(self, H: torch.Tensor, rule: torch.Tensor, step: int, max_steps: int):
        feat = self._program_features(H, rule, step, max_steps)
        return self.arc_program_controller(feat), self.arc_program_value(feat).squeeze(-1)

    @torch.no_grad()
    def install_base_operator_bank_from_core(self):
        self.core.materialize_operator_bank()
        self.operator_bank = AdaptiveOperatorBank(
            self.core._transport_bank.detach().clone(),
            self.core._scale_bank.detach().clone(),
            self.core._bias_bank.detach().clone(),
            rank=self.cfg.operator_residual_rank,
            transport_residual_scale=self.cfg.operator_residual_scale,
            feature_residual_scale=self.cfg.feature_residual_scale,
        ).to(self.core.coords.device)
        for p in self.core.parameters():
            p.requires_grad_(False)

    def ensure_operator_bank(self):
        if self.operator_bank is None:
            self.install_base_operator_bank_from_core()

    def rollout_program(
        self,
        H0: torch.Tensor,
        rule: torch.Tensor,
        max_steps: int | None = None,
        temperature: float = 1.0,
        hard: bool = True,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        self.ensure_operator_bank()
        steps = max_steps or self.cfg.max_program_steps
        H = H0
        # 0 = still active, 1 = already stopped. Straight-through stop weights
        # make STOP absorbing while keeping gradients during training.
        halted = torch.zeros(H.shape[0], 1, 1, device=H.device, dtype=H.dtype)
        action_weights: List[torch.Tensor] = []
        value_preds: List[torch.Tensor] = []
        for t in range(steps):
            logits, value = self.program_policy_value(H, rule, t, steps)
            if greedy:
                idx = logits.argmax(dim=-1)
                w = F.one_hot(idx, self.num_actions).to(H.dtype)
            else:
                w = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

            # Once a sample has stopped, force every later reported action to
            # STOP as well. This keeps length/usage losses and diagnostics aligned
            # with the absorbing execution semantics.
            active = (1.0 - halted).squeeze(-1)  # [B,1]
            stop_only = torch.zeros_like(w)
            stop_only[:, self.stop_action] = 1.0
            w_eff = active * w + (1.0 - active) * stop_only

            all_ops = self.operator_bank.apply_all(H)  # [B,K,N,D]
            op_state = (all_ops * w_eff[:, : self.operator_count, None, None]).sum(dim=1)
            stop_w = w_eff[:, self.stop_action:self.stop_action + 1, None]
            candidate = op_state + stop_w * H
            H = halted * H + (1.0 - halted) * candidate
            halted = halted + (1.0 - halted) * stop_w
            action_weights.append(w_eff)
            value_preds.append(value)
        return H, torch.stack(action_weights, dim=1), value_preds

    @torch.no_grad()
    def greedy_actions(self, H0: torch.Tensor, rule: torch.Tensor, max_steps: int | None = None):
        H, weights, _ = self.rollout_program(H0, rule, max_steps=max_steps, greedy=True)
        return H, weights.argmax(dim=-1)
