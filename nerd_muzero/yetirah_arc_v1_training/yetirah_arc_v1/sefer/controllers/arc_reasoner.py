from __future__ import annotations

# Kept for compatibility with the current launcher.
ARC_REASONER_PATCH_ID = "v1.8-refreshable-cppn-base"
ARC_REASONER_ARCH = "recursive-deltanet-v1-demo-controller-v9"

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

        # Rich task-conditioned state ranker.
        # Ranking features are detached from the solver so selector training
        # cannot distort a recursive trajectory that is already useful.
        self.ranker_task_proj = nn.Sequential(
            nn.LayerNorm(cfg.rule_dim),
            nn.Linear(cfg.rule_dim, cfg.node_dim),
            nn.GELU(),
        )
        ranker_in = cfg.node_dim * 5 + 4
        self.state_ranker = nn.Sequential(
            nn.LayerNorm(ranker_in),
            nn.Linear(ranker_in, cfg.rule_dim),
            nn.GELU(),
            nn.Linear(cfg.rule_dim, cfg.rule_dim),
            nn.GELU(),
            nn.Linear(cfg.rule_dim, 1),
        )

        # v9 controller: maps a task's leave-one-demo-out depth profile plus
        # query trajectory diagnostics to the depth to use on the real query.
        self.depth_controller_task_proj = nn.Sequential(
            nn.LayerNorm(cfg.rule_dim),
            nn.Linear(cfg.rule_dim, cfg.node_dim),
            nn.GELU(),
        )
        depth_count = cfg.recursive_steps + 1
        controller_in = cfg.node_dim + depth_count * 3
        self.depth_controller = nn.Sequential(
            nn.LayerNorm(controller_in),
            nn.Linear(controller_in, cfg.rule_dim),
            nn.GELU(),
            nn.Linear(cfg.rule_dim, cfg.rule_dim),
            nn.GELU(),
            nn.Linear(cfg.rule_dim, depth_count),
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

    def score_demo_conditioned_depth(self, rule, info, demo_depth_profile):
        """Score depth 0..T from task-local LOO evidence + query trajectory.

        demo_depth_profile: [B,T+1], lower is better.  It is normalized per
        task before entering the controller so scale differences across ARC
        tasks do not dominate.
        """
        states = info["states"]
        B = rule.shape[0]
        K = len(states)

        profile = demo_depth_profile.detach()
        pmean = profile.mean(dim=1, keepdim=True)
        pstd = profile.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        profile_z = (profile - pmean) / pstd

        state_norms = []
        delta_norms = []
        prev = states[0].detach()
        for i, state in enumerate(states):
            s = state.detach()
            state_norms.append(
                s.pow(2).mean(dim=(1, 2)).sqrt()
            )
            if i == 0:
                delta_norms.append(torch.zeros(B, device=s.device, dtype=s.dtype))
            else:
                delta_norms.append(
                    (s - prev).pow(2).mean(dim=(1, 2)).sqrt()
                )
            prev = s

        state_norms = torch.stack(state_norms, dim=1)
        delta_norms = torch.stack(delta_norms, dim=1)

        task_vec = self.depth_controller_task_proj(rule.detach())
        feat = torch.cat(
            [task_vec, profile_z, state_norms, delta_norms],
            dim=-1,
        )
        return self.depth_controller(feat)

    def select_demo_conditioned_depth(self, rule, info, demo_depth_profile):
        logits = self.score_demo_conditioned_depth(
            rule, info, demo_depth_profile
        )
        chosen = logits.argmax(dim=1)
        stacked = torch.stack(info["states"], dim=1)
        bidx = torch.arange(stacked.shape[0], device=stacked.device)
        selected = stacked[bidx, chosen]
        return selected, chosen, logits


    def score_recursive_states(self, rule, info):
        """Score state 0..T using detached task/state diagnostics.

        Features:
          task projection,
          state mean + std,
          delta mean + absolute delta mean,
          state norm,
          fast-state norm,
          halt probability,
          normalized step index.
        """
        states = info["states"]
        B = rule.shape[0]
        total_steps = max(len(states) - 1, 1)

        task_vec = self.ranker_task_proj(rule.detach())
        scores = []
        prev = states[0].detach()

        for i, state in enumerate(states):
            s = state.detach()
            state_mean = s.mean(dim=1)
            state_std = s.std(dim=1, unbiased=False)

            if i == 0:
                delta = torch.zeros_like(s)
            else:
                delta = s - prev
            delta_mean = delta.mean(dim=1)
            delta_abs = delta.abs().mean(dim=1)

            state_norm = s.pow(2).mean(dim=(1, 2), keepdim=False).sqrt().unsqueeze(-1)

            fast_norm = torch.zeros(B, 1, device=s.device, dtype=s.dtype)
            halt_prob = torch.zeros(B, 1, device=s.device, dtype=s.dtype)
            if i > 0 and len(info.get("stats", [])) >= i:
                stat = info["stats"][i - 1]
                v = stat.get("fast_norm")
                if torch.is_tensor(v):
                    fast_norm = (
                        v.detach().expand(B).unsqueeze(-1)
                        if v.ndim == 0
                        else v.detach().reshape(B, -1).mean(dim=1, keepdim=True)
                    )
                if len(info.get("halt_logits", [])) >= i:
                    h = info["halt_logits"][i - 1].detach()
                    halt_prob = torch.sigmoid(h).reshape(B, -1).mean(dim=1, keepdim=True)

            step_frac = torch.full(
                (B, 1),
                float(i) / float(total_steps),
                device=s.device,
                dtype=s.dtype,
            )

            feat = torch.cat(
                [
                    task_vec,
                    state_mean,
                    state_std,
                    delta_mean,
                    delta_abs,
                    state_norm,
                    fast_norm,
                    halt_prob,
                    step_frac,
                ],
                dim=-1,
            )
            scores.append(self.state_ranker(feat).squeeze(-1))
            prev = s

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
