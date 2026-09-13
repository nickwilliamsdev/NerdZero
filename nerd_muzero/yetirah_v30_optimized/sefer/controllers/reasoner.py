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

from sefer.algebra.core import YetirahCore
from sefer.representation.delta_state import SmallDeltaStateEncoder
from sefer.representation.scalar_codec import FixedScalarLift, FixedScalarReadout

class TinyReasoner(nn.Module):
    """Factorized rule/query reasoner.

    The demonstration pair is encoded into a compact rule latent R. The raw
    query is encoded independently into a geometric workspace Q. Routing may
    inspect both R and Q, but the generated operators act only on Q. This is
    the key v7 compositionality constraint: operator semantics cannot depend on
    the current task embedding being entangled inside the state they transform.
    """

    HALT_ACTION = 22

    def __init__(
        self,
        input_dim: int = 32,
        node_dim: int = 32,
        coord_dim: int = 5,
        operator_count: int = 22,
        operator_code_dim: int = 8,
        rule_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.rule_dim = rule_dim
        self.operator_count = operator_count
        self.num_actions = operator_count + 1
        self.core = YetirahCore(
            coord_dim=coord_dim,
            node_dim=node_dim,
            operator_count=operator_count,
            operator_code_dim=operator_code_dim,
        )
        n_nodes = 2 ** coord_dim
        self.n_nodes = n_nodes

        # Rule stream: demonstration-only. Query information never enters R.
        relation_scalar_dim = 6
        rule_input_dim = input_dim * 5 + relation_scalar_dim
        self.rule_encoder = nn.Sequential(
            nn.Linear(rule_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, rule_dim),
            nn.Tanh(),
        )

        # Query stream: query-only geometric workspace. This is the state that
        # operators transform, independent of which rule is currently inferred.
        # Position-preserving query lift: input position i always maps to
        # substrate vertex i.  The representation can no longer hide position
        # inside an arbitrary dense 32->1024 transform.
        # v11: exact algebra-friendly coordinate system. Input scalar x_i is
        # stored directly in feature channel 0 at substrate vertex i. There is
        # no learned codec that can rotate or rescale the transform space.
        self.query_encoder = FixedScalarLift(node_dim)

        self.pool_query = nn.Parameter(torch.randn(node_dim) / math.sqrt(node_dim))

        # Controller may inspect both the fixed rule latent and evolving query
        # workspace. Operator execution itself never receives R.
        # v27: primitive control is explicitly phase-aware. For distribution-
        # preserving transforms (roll/negate/flip), the post-transform state can
        # be statistically indistinguishable from a fresh pre-transform state;
        # identity is literally unchanged. A stationary H+rule policy therefore
        # cannot consistently learn "apply once, then HALT". We append a scalar
        # primitive phase: 0 before the atomic action, 1 after it.
        primitive_policy_in = n_nodes * node_dim + rule_dim + 1
        primitive_summary_in = node_dim + rule_dim + 1
        self.full_state_to_policy = nn.Sequential(
            nn.Linear(primitive_policy_in, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, operator_code_dim),
        )
        self.halt_head = nn.Sequential(
            nn.Linear(primitive_summary_in, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(primitive_summary_in, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Direct diagnostic path is explicitly rule-conditioned. It measures how
        # much can be solved without the operator algebra.
        self.direct_decoder = nn.Sequential(
            nn.Linear(n_nodes * node_dim + rule_dim, 192),
            nn.GELU(),
            nn.Linear(192, input_dim),
        )

        # Operator decoder is deliberately rule-blind. Q must be transformed
        # into a decodable target state by the chosen operator program.
        # Shared per-vertex readout preserves the same positional semantics.
        self.query_decoder = FixedScalarReadout()

        # Query autoencoder control: before applying any operator, Q should still
        # decode back to the raw query. This gives the operators a stable,
        # task-independent substrate on which to act.
        self.task_probe_rule = nn.Sequential(
            nn.Linear(rule_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

        # v15: separate goal-conditioned program selector. It never shares
        # parameters with the primitive rule encoder/policy validated in v12.
        # Given the *current support state* and the demonstrated target state,
        # it selects the next primitive operator. This makes program inference
        # iterative rather than classifying a whole program from a static rule.
        goal_relation_dim = input_dim * 5 + 15
        # v29: augment the explicit current/goal relation with a compact
        # DeltaNet-style fast-weight summary. The raw relation is retained, so
        # this is additive representation capacity rather than a bottleneck.
        self.delta_state_dim = 32
        self.delta_state = SmallDeltaStateEncoder(input_dim, self.delta_state_dim)
        program_state_dim = goal_relation_dim + self.delta_state_dim
        self.program_controller = nn.Sequential(
            nn.Linear(program_state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 5),  # four primitive operators + STOP
        )

        # MuZero-style goal value. Unlike the primitive value head, this is
        # conditioned directly on current support state vs demonstrated goal,
        # matching the state distribution encountered by program search.
        self.program_value = nn.Sequential(
            nn.Linear(program_state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _normalized_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        num = (a * b).mean(dim=-1, keepdim=True)
        den = (
            a.pow(2).mean(dim=-1, keepdim=True).sqrt()
            * b.pow(2).mean(dim=-1, keepdim=True).sqrt()
        ).clamp_min(1e-6)
        return num / den

    def encode_rule(self, demo_x: torch.Tensor, demo_y: torch.Tensor) -> torch.Tensor:
        delta = demo_y - demo_x
        summed = demo_y + demo_x
        product = demo_x * demo_y
        shift_corrs = [
            self._normalized_corr(torch.roll(demo_x, shifts=shift, dims=-1), demo_y)
            for shift in (-2, -1, 0, 1, 2)
        ]
        reverse_corr = self._normalized_corr(demo_x.flip(-1), demo_y)
        relation_scalars = torch.cat(shift_corrs + [reverse_corr], dim=-1)
        context = torch.cat(
            [demo_x, demo_y, delta, summed, product, relation_scalars], dim=-1
        )
        return self.rule_encoder(context)

    def goal_features(
        self,
        current_H: torch.Tensor,
        target_H: torch.Tensor,
        remaining_steps: int | torch.Tensor = 1,
    ) -> torch.Tensor:
        # FixedScalarLift makes decoding exact, so the controller/value compare
        # the support object's current value directly with the demonstrated goal.
        # v17 additionally conditions both policy and value on the finite
        # planning horizon. This distinguishes, e.g., "one action from goal"
        # from "two actions from goal" even when current/target are identical.
        current = self.decode_query(current_H)
        target = self.decode_query(target_H)
        delta = target - current
        summed = target + current
        product = target * current
        # v29: the program horizon is four, so expose cyclic evidence across
        # that whole range rather than only +/-2. Reversed+shifted correlations
        # help distinguish order-sensitive roll/flip compositions.
        shift_corrs = [
            self._normalized_corr(torch.roll(current, shifts=shift, dims=-1), target)
            for shift in range(-4, 5)
        ]
        reversed_current = current.flip(-1)
        reverse_shift_corrs = [
            self._normalized_corr(torch.roll(reversed_current, shifts=shift, dims=-1), target)
            for shift in (-2, -1, 0, 1, 2)
        ]
        rel = torch.cat(shift_corrs + reverse_shift_corrs, dim=-1)
        if torch.is_tensor(remaining_steps):
            horizon = remaining_steps.to(current.device, current.dtype).reshape(-1, 1)
            if horizon.shape[0] == 1 and current.shape[0] != 1:
                horizon = horizon.expand(current.shape[0], 1)
        else:
            horizon = torch.full(
                (current.shape[0], 1), float(remaining_steps),
                device=current.device, dtype=current.dtype,
            )
        # Normalize by the synthetic proof's maximum program horizon.
        horizon = horizon / 4.0
        return torch.cat([current, target, delta, summed, product, rel, horizon], dim=-1)

    def program_state_features(
        self,
        current_H: torch.Tensor,
        target_H: torch.Tensor,
        remaining_steps: int | torch.Tensor = 1,
    ) -> torch.Tensor:
        """Explicit goal relation plus a DeltaNet fast-weight state summary."""
        base = self.goal_features(current_H, target_H, remaining_steps)
        delta_state = self.delta_state(base)
        return torch.cat([base, delta_state], dim=-1)

    def program_policy(
        self, current_H: torch.Tensor, target_H: torch.Tensor, remaining_steps: int | torch.Tensor = 1
    ) -> torch.Tensor:
        return self.program_controller(self.program_state_features(current_H, target_H, remaining_steps))

    def program_policy_value(
        self,
        current_H: torch.Tensor,
        target_H: torch.Tensor,
        remaining_steps: int | torch.Tensor = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.program_state_features(current_H, target_H, remaining_steps)
        logits = self.program_controller(feat)
        value = self.program_value(feat).squeeze(-1)
        return logits, value

    def encode_query(self, query_x: torch.Tensor) -> torch.Tensor:
        # [B,32] -> [B,32,D], one scalar per fixed hypercube vertex.
        return self.query_encoder(query_x)

    def encode(
        self,
        demo_x: torch.Tensor,
        demo_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_rule(demo_x, demo_y), self.encode_query(query_x)

    def pool(self, H: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bnd,d->bn", H, self.pool_query)
        attn = torch.softmax(score / math.sqrt(self.node_dim), dim=-1)
        return torch.einsum("bn,bnd->bd", attn, H)

    def policy_value(
        self,
        H: torch.Tensor,
        rule: torch.Tensor,
        primitive_phase: int | float | torch.Tensor = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Primitive policy/value with explicit pre/post-action phase.

        primitive_phase=0 means choose the atomic operator.
        primitive_phase=1 means the atomic action has been applied and HALT is
        now a valid completion decision. This resolves an identifiability problem
        for distribution-preserving primitives and identity.
        """
        z = self.pool(H)
        flat = H.flatten(start_dim=1)
        if torch.is_tensor(primitive_phase):
            phase = primitive_phase.to(H.device, H.dtype).reshape(-1, 1)
            if phase.shape[0] == 1 and H.shape[0] != 1:
                phase = phase.expand(H.shape[0], 1)
        else:
            phase = torch.full((H.shape[0], 1), float(primitive_phase), device=H.device, dtype=H.dtype)
        q = self.full_state_to_policy(torch.cat([flat, rule, phase], dim=-1))
        op_logits = q @ self.core.operator_codes.t() / math.sqrt(q.shape[-1])
        summary = torch.cat([z, rule, phase], dim=-1)
        halt_logit = self.halt_head(summary)
        logits = torch.cat([op_logits, halt_logit], dim=-1)
        value = self.value_head(summary).squeeze(-1)
        return logits, value

    def transition(self, H: torch.Tensor, action: int) -> torch.Tensor:
        if action == self.HALT_ACTION:
            return H
        return self.core.apply_operator(H, action)

    def decode_direct(self, H: torch.Tensor, rule: torch.Tensor) -> torch.Tensor:
        return self.direct_decoder(torch.cat([H.flatten(start_dim=1), rule], dim=-1))

    def decode_query(self, H: torch.Tensor) -> torch.Tensor:
        return self.query_decoder(H)

    def decode(self, H: torch.Tensor) -> torch.Tensor:
        return self.decode_query(H)

    def task_logits_rule(self, rule: torch.Tensor) -> torch.Tensor:
        return self.task_probe_rule(rule)
