from __future__ import annotations

# Kept for compatibility with the current launcher.
ARC_REASONER_PATCH_ID = "v1.8-refreshable-cppn-base"
ARC_REASONER_ARCH = "recursive-deltanet-v1-state-ranker-v6"

import torch
import torch.nn as nn

from sefer.representation.arc_grid import ARCGridEncoder, ARCGridDecoder
from sefer.algebra.recursive_deltanet import RecursiveDeltaNetCell


class ARCReasoner(nn.Module):
    """ARC reasoner with task-conditioned recursive fast-weight adaptation."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_slots = cfg.n_slots
        self.node_dim = cfg.node_dim

        self.grid_encoder = ARCGridEncoder(
            max_size=cfg.max_grid_size,
            colors=cfg.color_count,
            cell_dim=cfg.cell_dim,
            slot_dim=cfg.node_dim,
            n_slots=cfg.n_slots,
            heads=cfg.attention_heads,
            dropout=cfg.codec_dropout,
            latent_layers=cfg.codec_latent_layers,
        )
        self.grid_decoder = ARCGridDecoder(
            max_size=cfg.max_grid_size,
            colors=cfg.color_count,
            cell_dim=cfg.cell_dim,
            slot_dim=cfg.node_dim,
            heads=cfg.attention_heads,
            dropout=cfg.codec_dropout,
        )

        pair_dim = cfg.node_dim * 4
        self.demo_slot_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 256),
            nn.GELU(),
            nn.Linear(256, cfg.rule_dim),
            nn.GELU(),
        )
        self.demo_slot_score = nn.Linear(cfg.rule_dim, 1)
        self.demo_pair_encoder = nn.Sequential(
            nn.Linear(cfg.rule_dim + 4, 256),
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

        self.query_slot_proj = nn.Sequential(
            nn.LayerNorm(cfg.node_dim),
            nn.Linear(cfg.node_dim, cfg.rule_dim),
            nn.GELU(),
        )
        self.query_rule_query = nn.Linear(cfg.rule_dim, cfg.rule_dim)
        self.query_rule_refiner = nn.Sequential(
            nn.LayerNorm(cfg.rule_dim * 2),
            nn.Linear(cfg.rule_dim * 2, cfg.rule_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.rule_dim * 2, cfg.rule_dim),
        )
        self.query_rule_norm = nn.LayerNorm(cfg.rule_dim)

        flat_dim = cfg.n_slots * cfg.node_dim
        self.direct_rule_to_slots = nn.Sequential(
            nn.Linear(cfg.rule_dim, 256),
            nn.GELU(),
            nn.Linear(256, flat_dim),
        )
        self.direct_norm = nn.LayerNorm(cfg.node_dim)

        self.recursive_cell = RecursiveDeltaNetCell(
            state_dim=cfg.node_dim,
            task_dim=cfg.rule_dim,
            heads=cfg.recursive_heads,
            update_scale=cfg.fast_update_scale,
            residual_scale=cfg.recursive_residual_scale,
            decay_min=cfg.fast_decay_min,
            decay_max=cfg.fast_decay_max,
        )

        # Scores direct state (0) and each recursive state jointly.
        # Inputs: task summary, state summary, state delta, fast norm, step fraction.
        ranker_in = cfg.node_dim * 3 + 2
        self.state_ranker = nn.Sequential(
            nn.Linear(ranker_in, cfg.rule_dim),
            nn.GELU(),
            nn.Linear(cfg.rule_dim, cfg.rule_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.rule_dim // 2, 1),
        )

    def encode_grid(self, grid: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        return self.grid_encoder(grid, shapes)

    def decode_grid(self, H: torch.Tensor):
        return self.grid_decoder(H)

    def encode_rule(self, demos_x, demos_y, demos_x_shapes, demos_y_shapes, demo_mask):
        B, D, S, _ = demos_x.shape
        hx = self.encode_grid(
            demos_x.reshape(B * D, S, S),
            demos_x_shapes.reshape(B * D, 2),
        ).reshape(B, D, self.n_slots, self.node_dim)
        hy = self.encode_grid(
            demos_y.reshape(B * D, S, S),
            demos_y_shapes.reshape(B * D, 2),
        ).reshape(B, D, self.n_slots, self.node_dim)

        slot_feat = torch.cat([hx, hy, hy - hx, hx * hy], dim=-1)
        slot_z = self.demo_slot_encoder(slot_feat)
        slot_score = self.demo_slot_score(slot_z).squeeze(-1)
        slot_attn = torch.softmax(slot_score, dim=-1)[..., None]
        spatial_pair = (slot_z * slot_attn).sum(dim=2)

        shape_feat = torch.cat([
            demos_x_shapes.float() / float(S),
            demos_y_shapes.float() / float(S),
        ], dim=-1)
        pair_z = self.demo_pair_encoder(
            torch.cat([spatial_pair, shape_feat], dim=-1)
        )
        m = demo_mask.float()[..., None]
        pooled = (pair_z * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.rule_encoder(pooled)

    def condition_rule_on_query(self, rule: torch.Tensor, H: torch.Tensor):
        qslots = self.query_slot_proj(H)
        q = self.query_rule_query(rule)[:, None, :]
        score = (qslots * q).sum(dim=-1) / (self.cfg.rule_dim ** 0.5)
        attn = torch.softmax(score, dim=-1)[..., None]
        context = (qslots * attn).sum(dim=1)
        delta = self.query_rule_refiner(torch.cat([rule, context], dim=-1))
        return self.query_rule_norm(rule + 0.5 * torch.tanh(delta))

    def predict_goal(self, H: torch.Tensor, rule: torch.Tensor):
        delta = self.direct_rule_to_slots(rule).view(
            H.shape[0], self.n_slots, self.node_dim
        )
        return self.direct_norm(H + 0.25 * torch.tanh(delta))

    def recursive_reason(self, H: torch.Tensor, rule: torch.Tensor,
                         steps: int | None = None, return_trace: bool = False):
        steps = int(steps or self.cfg.recursive_steps)
        state = self.predict_goal(H, rule)
        fast = self.recursive_cell.init_fast_state(rule)
        trace = [state]
        halt_logits = []
        stats = []

        for t in range(steps):
            state, fast, halt_logit, step_stats = self.recursive_cell(
                state, rule, fast, (t + 1) / max(steps, 1)
            )
            trace.append(state)
            halt_logits.append(halt_logit)
            stats.append(step_stats)

        if return_trace:
            return state, {
                "states": trace,
                "halt_logits": halt_logits,
                "stats": stats,
                "fast_state": fast,
            }
        return state

    def score_recursive_states(self, rule, info):
        """Return learned scores for state 0..T using only inference-time signals."""
        states = info["states"]
        B = rule.shape[0]
        D = self.cfg.node_dim

        # Lightweight projection without adding another large task encoder.
        if rule.shape[-1] >= D:
            task_vec = rule[..., :D]
        else:
            task_vec = torch.cat(
                [
                    rule,
                    torch.zeros(
                        B, D - rule.shape[-1],
                        device=rule.device, dtype=rule.dtype,
                    ),
                ],
                dim=-1,
            )

        scores = []
        prev = states[0]
        total_steps = max(len(states) - 1, 1)

        for i, state in enumerate(states):
            state_vec = state.mean(dim=1)
            delta_vec = (
                torch.zeros_like(state_vec)
                if i == 0 else (state - prev).mean(dim=1)
            )

            fast_norm = torch.zeros(B, 1, device=state.device, dtype=state.dtype)
            if i > 0 and len(info.get("stats", [])) >= i:
                v = info["stats"][i - 1].get("fast_norm")
                if torch.is_tensor(v):
                    fast_norm = (
                        v.expand(B).unsqueeze(-1)
                        if v.ndim == 0
                        else v.reshape(B, -1).mean(dim=1, keepdim=True)
                    )

            step_frac = torch.full(
                (B, 1), float(i) / float(total_steps),
                device=state.device, dtype=state.dtype,
            )
            feat = torch.cat(
                [task_vec, state_vec, delta_vec, fast_norm, step_frac], dim=-1
            )
            scores.append(self.state_ranker(feat).squeeze(-1))
            prev = state

        return torch.stack(scores, dim=1)

    def select_ranked_state(self, rule, info):
        scores = self.score_recursive_states(rule, info)
        chosen = scores.argmax(dim=1)
        stacked = torch.stack(info["states"], dim=1)
        bidx = torch.arange(stacked.shape[0], device=stacked.device)
        selected = stacked[bidx, chosen]
        return selected, chosen, scores


    def select_adaptive_state(self, info):
        """Choose a recursive state using the learned halt logits only.

        halt_logits[t] scores state[t+1].  No target information is used.
        """
        states = info["states"]
        halt_logits = info["halt_logits"]
        if not halt_logits:
            return states[-1], 0

        logits = torch.stack(halt_logits, dim=1)  # [B,T]
        min_step = max(1, int(getattr(self.cfg, "adaptive_halt_min_step", 1)))

        if getattr(self.cfg, "adaptive_halt_use_argmax", True):
            chosen = logits.argmax(dim=1) + 1
        else:
            probs = torch.sigmoid(logits)
            threshold = float(getattr(self.cfg, "halt_threshold", 0.90))
            chosen = torch.full(
                (logits.shape[0],), len(halt_logits),
                device=logits.device, dtype=torch.long,
            )
            for t in range(max(min_step - 1, 0), len(halt_logits)):
                take = (probs[:, t] >= threshold) & (chosen == len(halt_logits))
                chosen[take] = t + 1

        chosen = chosen.clamp(min=min_step, max=len(halt_logits))
        stacked = torch.stack(states, dim=1)  # [B,T+1,N,D]
        bidx = torch.arange(stacked.shape[0], device=stacked.device)
        selected = stacked[bidx, chosen]
        return selected, chosen

    def forward_episode(self, demos_x, demos_y, demos_x_shapes, demos_y_shapes,
                        demo_mask, query_x, query_shape, steps: int | None = None,
                        return_trace: bool = False):
        rule = self.encode_rule(
            demos_x, demos_y, demos_x_shapes, demos_y_shapes, demo_mask
        )
        Hq = self.encode_grid(query_x, query_shape)
        rule = self.condition_rule_on_query(rule, Hq)
        if return_trace:
            Hr, info = self.recursive_reason(Hq, rule, steps=steps, return_trace=True)
            info["rule"] = rule
            info["query_state"] = Hq
            info["direct_state"] = info["states"][0]
            return Hr, info
        return self.recursive_reason(Hq, rule, steps=steps)
