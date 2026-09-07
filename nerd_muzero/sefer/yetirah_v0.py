"""
yetirah_v0.py

Version 16 finite-horizon goal-conditioned MuZero-style program-search follow-up to the supervised transport algebra of a geometry-generated operator model inspired by the architecture
we discussed:

5D hypercube substrate (32 vertices)
    -> differentiable CPPN connectivity
    -> 22 shared/generated operators
    -> tiny policy/value heads
    -> shallow PUCT search over operator sequences
    -> antithetic ES step in genotype space
    -> ordinary backprop refinement

This file intentionally uses a synthetic "reasoning target" smoke test first.
The point is to verify that:
  1) CPPN generation is differentiable,
  2) operator identities specialize,
  3) PUCT can search operator sequences,
  4) ES + Adam can coexist without breaking autograd.

After this works reliably, the next step is replacing SyntheticTaskBatch with
an ARC task encoder/decoder while leaving the substrate/search machinery intact.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_all(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_hypercube_vertices(dim: int = 5, device=None) -> torch.Tensor:
    """Return all 2^dim vertices in {-1,+1}^dim."""
    n = 2 ** dim
    vals = []
    for i in range(n):
        bits = [(1.0 if ((i >> b) & 1) else -1.0) for b in range(dim)]
        vals.append(bits)
    return torch.tensor(vals, dtype=torch.float32, device=device)


def flatten_params(module: nn.Module) -> torch.Tensor:
    """Flatten trainable parameters into one vector."""
    return torch.cat([p.detach().reshape(-1) for p in module.parameters() if p.requires_grad])


@torch.no_grad()
def add_flat_delta_(module: nn.Module, delta: torch.Tensor):
    """Add a flat delta vector to trainable parameters in-place."""
    offset = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.add_(delta[offset:offset+n].view_as(p))
        offset += n
    assert offset == delta.numel()


@torch.no_grad()
def set_flat_params_(module: nn.Module, flat: torch.Tensor):
    """Overwrite trainable parameters from one flat vector."""
    offset = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.copy_(flat[offset:offset+n].view_as(p))
        offset += n
    assert offset == flat.numel()


# ---------------------------------------------------------------------------
# Differentiable CPPN connectivity
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 22 generated operators
# ---------------------------------------------------------------------------

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

    def transport(
        self,
        coords: torch.Tensor,
        code: torch.Tensor,
    ) -> torch.Tensor:
        """Return row-stochastic [N,N] destination<-source transport matrix."""
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


# ---------------------------------------------------------------------------
# Tiny representation, policy, value, decoder
# ---------------------------------------------------------------------------



class FixedScalarLift(nn.Module):
    """Exact scalar -> feature lift: x is stored in feature channel 0."""
    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*x.shape, self.node_dim, device=x.device, dtype=x.dtype)
        out[..., 0] = x
        return out


class FixedScalarReadout(nn.Module):
    """Exact inverse of FixedScalarLift."""
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return H[..., 0]


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
        self.full_state_to_policy = nn.Sequential(
            nn.Linear(n_nodes * node_dim + rule_dim, 192),
            nn.GELU(),
            nn.Linear(192, operator_code_dim),
        )
        self.halt_head = nn.Sequential(
            nn.Linear(node_dim + rule_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(node_dim + rule_dim, 96),
            nn.GELU(),
            nn.Linear(96, 1),
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
        goal_relation_dim = input_dim * 5 + 7
        self.program_controller = nn.Sequential(
            nn.Linear(goal_relation_dim, 192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, operator_count),
        )

        # MuZero-style goal value. Unlike the primitive value head, this is
        # conditioned directly on current support state vs demonstrated goal,
        # matching the state distribution encountered by program search.
        self.program_value = nn.Sequential(
            nn.Linear(goal_relation_dim, 192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, 1),
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
        # v16 additionally conditions both policy and value on the finite
        # planning horizon. This distinguishes, e.g., "one action from goal"
        # from "two actions from goal" even when current/target are identical.
        current = self.decode_query(current_H)
        target = self.decode_query(target_H)
        delta = target - current
        summed = target + current
        product = target * current
        shift_corrs = [
            self._normalized_corr(torch.roll(current, shifts=shift, dims=-1), target)
            for shift in (-2, -1, 0, 1, 2)
        ]
        reverse_corr = self._normalized_corr(current.flip(-1), target)
        rel = torch.cat(shift_corrs + [reverse_corr], dim=-1)
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
        horizon = horizon / 2.0
        return torch.cat([current, target, delta, summed, product, rel, horizon], dim=-1)

    def program_policy(
        self, current_H: torch.Tensor, target_H: torch.Tensor, remaining_steps: int | torch.Tensor = 1
    ) -> torch.Tensor:
        return self.program_controller(self.goal_features(current_H, target_H, remaining_steps))

    def program_policy_value(
        self,
        current_H: torch.Tensor,
        target_H: torch.Tensor,
        remaining_steps: int | torch.Tensor = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.goal_features(current_H, target_H, remaining_steps)
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

    def policy_value(self, H: torch.Tensor, rule: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.pool(H)
        flat = H.flatten(start_dim=1)
        q = self.full_state_to_policy(torch.cat([flat, rule], dim=-1))
        op_logits = q @ self.core.operator_codes.t() / math.sqrt(q.shape[-1])
        halt_logit = self.halt_head(torch.cat([z, rule], dim=-1))
        logits = torch.cat([op_logits, halt_logit], dim=-1)
        value = self.value_head(torch.cat([z, rule], dim=-1)).squeeze(-1)
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


# ---------------------------------------------------------------------------
# Goal-conditioned MuZero-style PUCT
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    simulations: int = 64
    max_depth: int = 2
    c_puct: float = 1.5
    discount: float = 1.0
    terminal_beta: float = 2.0
    action_limit: int = 4  # synthetic proof: search the four validated primitives


class SearchNode:
    def __init__(
        self,
        state: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
        depth: int = 0,
    ):
        self.state = state
        self.prior = prior
        self.depth = depth
        self.visit = None
        self.value_sum = None
        self.children: Dict[int, "SearchNode"] = {}

    def init_actions(self, n_actions: int, device):
        if self.visit is None:
            self.visit = torch.zeros(n_actions, device=device)
            self.value_sum = torch.zeros(n_actions, device=device)

    def q(self):
        return self.value_sum / self.visit.clamp_min(1.0)


def goal_error(model: TinyReasoner, current_H: torch.Tensor, target_H: torch.Tensor) -> torch.Tensor:
    """Per-sample support-to-goal MSE used as an observable planning signal."""
    current = model.decode_query(current_H)
    target = model.decode_query(target_H)
    return F.mse_loss(current, target, reduction="none").mean(dim=-1)


def terminal_goal_score(
    model: TinyReasoner, current_H: torch.Tensor, target_H: torch.Tensor, beta: float = 2.0
) -> torch.Tensor:
    """Terminal utility: 1 for exact goal match, decaying with support MSE."""
    return torch.exp(-beta * goal_error(model, current_H, target_H).detach())


@torch.no_grad()
def exact_horizon_value_target(
    model: TinyReasoner,
    current_H: torch.Tensor,
    target_H: torch.Tensor,
    remaining_steps: int,
    action_limit: int = 4,
    beta: float = 2.0,
) -> torch.Tensor:
    """Exact finite-horizon Bellman target for the synthetic proof.

    V*_h(s,g) = max over all h-step primitive programs of terminal_goal_score.
    With four actions and h<=2 this is cheap (at most 16 leaves/sample) and
    gives the learned value precisely the semantics PUCT requires.
    """
    if remaining_steps <= 0:
        return terminal_goal_score(model, current_H, target_H, beta=beta)
    candidates = []
    for a in range(min(action_limit, model.operator_count)):
        next_H = model.transition(current_H, a)
        candidates.append(
            exact_horizon_value_target(
                model, next_H, target_H, remaining_steps - 1, action_limit, beta
            )
        )
    return torch.stack(candidates, dim=0).max(dim=0).values


@torch.no_grad()
def goal_puct_search(
    model: TinyReasoner,
    root_state: torch.Tensor,
    target_state: torch.Tensor,
    cfg: SearchConfig,
) -> Tuple[int, torch.Tensor, float]:
    """Finite-horizon goal-conditioned PUCT over the learned operator algebra.

    v16 gives the policy/value the exact number of decisions remaining. Leaves
    at horizon zero are scored by actual terminal support-goal fit; earlier
    leaves use V_theta(s,g,h), trained against exact finite-horizon Bellman
    targets. There is deliberately no mixed-scale pseudo-reward in the backup.
    """
    assert root_state.shape[0] == 1 and target_state.shape[0] == 1
    device = root_state.device
    n_actions = min(cfg.action_limit, model.operator_count)

    root_logits, _ = model.program_policy_value(root_state, target_state, cfg.max_depth)
    root_prior = torch.softmax(root_logits[0, :n_actions], dim=-1)
    root = SearchNode(root_state.clone(), root_prior, depth=0)
    root.init_actions(n_actions, device)

    for _ in range(cfg.simulations):
        node = root
        path: List[Tuple[SearchNode, int]] = []
        while True:
            remaining = max(cfg.max_depth - node.depth, 0)
            logits, leaf_v = model.program_policy_value(node.state, target_state, remaining)
            prior = torch.softmax(logits[0, :n_actions], dim=-1)
            node.init_actions(n_actions, device)
            node.prior = prior

            if remaining <= 0:
                value = float(terminal_goal_score(
                    model, node.state, target_state, beta=cfg.terminal_beta
                ).item())
                break

            total_n = node.visit.sum()
            q = node.q()
            u = cfg.c_puct * prior * torch.sqrt(total_n + 1.0) / (1.0 + node.visit)
            action = int(torch.argmax(q + u).item())

            if action not in node.children:
                next_state = model.transition(node.state, action)
                node.children[action] = SearchNode(next_state.clone(), depth=node.depth + 1)
            child = node.children[action]
            path.append((node, action))

            child_remaining = max(cfg.max_depth - child.depth, 0)
            if child.visit is None:
                child.init_actions(n_actions, device)
                if child_remaining <= 0:
                    value = float(terminal_goal_score(
                        model, child.state, target_state, beta=cfg.terminal_beta
                    ).item())
                else:
                    _, child_v = model.program_policy_value(
                        child.state, target_state, child_remaining
                    )
                    value = float(child_v.item())
                break
            node = child

        # Zero intermediate reward; back up predicted/observed terminal utility.
        G = value
        for parent, action in reversed(path):
            parent.visit[action] += 1.0
            parent.value_sum[action] += G
            G = cfg.discount * G

    probs = root.visit / root.visit.sum().clamp_min(1.0)
    action = int(torch.argmax(probs).item())
    return action, probs, float(goal_error(model, root_state, target_state).item())


def soft_policy_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

# ---------------------------------------------------------------------------
# Synthetic smoke-test tasks
# ---------------------------------------------------------------------------

class SyntheticTaskBatch:
    """Primitive training tasks plus strictly held-out compositions."""

    TRAIN_NAMES = ("roll+1", "negate", "flip", "identity")
    COMPOSITIONS = {
        "roll2": (0, 0),
        "neg_after_roll": (0, 1),
        "flip_after_roll": (0, 2),
        "roll_after_flip": (2, 0),
        "neg_after_flip": (2, 1),
    }

    def __init__(self, dim: int = 32):
        self.dim = dim

    def apply_primitive(self, x: torch.Tensor, primitive_id: int) -> torch.Tensor:
        if primitive_id == 0:
            return torch.roll(x, shifts=1, dims=-1)
        if primitive_id == 1:
            return -x
        if primitive_id == 2:
            return x.flip(-1)
        if primitive_id == 3:
            return x
        raise ValueError(f"unknown primitive_id={primitive_id}")

    def apply_task(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        for task_id in range(len(self.TRAIN_NAMES)):
            mask = task_ids == task_id
            if mask.any():
                y[mask] = self.apply_primitive(x[mask], task_id)
        return y

    def apply_composition(self, x: torch.Tensor, primitive_sequence: Tuple[int, ...]) -> torch.Tensor:
        y = x
        for primitive_id in primitive_sequence:
            y = self.apply_primitive(y, primitive_id)
        return y

    def sample(self, batch_size: int, device):
        task_ids = torch.randint(0, len(self.TRAIN_NAMES), (batch_size,), device=device)
        demo_x = torch.randn(batch_size, self.dim, device=device)
        query_x = torch.randn(batch_size, self.dim, device=device)
        demo_y = self.apply_task(demo_x, task_ids)
        query_y = self.apply_task(query_x, task_ids)
        return demo_x, demo_y, query_x, query_y, task_ids

    def sample_composition(self, name: str, batch_size: int, device):
        primitive_sequence = self.COMPOSITIONS[name]
        demo_x = torch.randn(batch_size, self.dim, device=device)
        query_x = torch.randn(batch_size, self.dim, device=device)
        demo_y = self.apply_composition(demo_x, primitive_sequence)
        query_y = self.apply_composition(query_x, primitive_sequence)
        return demo_x, demo_y, query_x, query_y

    def sample_composition_triplet(self, name: str, batch_size: int, device):
        """Support demo, independent probe demo, and held-out query."""
        primitive_sequence = self.COMPOSITIONS[name]
        support_x = torch.randn(batch_size, self.dim, device=device)
        probe_x = torch.randn(batch_size, self.dim, device=device)
        query_x = torch.randn(batch_size, self.dim, device=device)
        support_y = self.apply_composition(support_x, primitive_sequence)
        probe_y = self.apply_composition(probe_x, primitive_sequence)
        query_y = self.apply_composition(query_x, primitive_sequence)
        return support_x, support_y, probe_x, probe_y, query_x, query_y

    def program_training_sequences(self):
        """Two-step programs used to train the controller in v13.

        The five evaluation programs are excluded exactly. The controller sees
        other combinations of the same primitive alphabet and must generalize
        to the held-out ordered programs.
        """
        held_out = set(self.COMPOSITIONS.values())
        return tuple((a, b) for a in range(4) for b in range(4) if (a, b) not in held_out)

    def sample_program_batch(self, batch_size: int, device):
        seqs = self.program_training_sequences()
        which = torch.randint(0, len(seqs), (batch_size,), device=device)
        program = torch.tensor([seqs[int(i)] for i in which.cpu().tolist()], device=device, dtype=torch.long)
        demo_x = torch.randn(batch_size, self.dim, device=device)
        query_x = torch.randn(batch_size, self.dim, device=device)
        demo_y = torch.empty_like(demo_x)
        query_y = torch.empty_like(query_x)
        for i, seq in enumerate(program.tolist()):
            demo_y[i:i+1] = self.apply_composition(demo_x[i:i+1], tuple(seq))
            query_y[i:i+1] = self.apply_composition(query_x[i:i+1], tuple(seq))
        return demo_x, demo_y, query_x, query_y, program


def differentiable_rollout(
    model: TinyReasoner,
    rule: torch.Tensor,
    H: torch.Tensor,
    steps: int = 1,
    temperature: float = 1.0,
    hard: bool = True,
):
    """Near-discrete operator rollout over a task-independent query workspace."""
    A = model.core.adjacency()
    routing_stats = []
    for _ in range(steps):
        logits, _ = model.policy_value(H, rule)
        op_logits = logits[:, :model.operator_count]
        probs = F.gumbel_softmax(op_logits, tau=temperature, hard=hard, dim=-1)
        all_next = [model.core.apply_operator(H, k, A) for k in range(model.operator_count)]
        stack = torch.stack(all_next, dim=1)
        H = torch.einsum("bk,bknd->bnd", probs, stack)
        routing_stats.append(torch.softmax(op_logits / max(temperature, 1e-4), dim=-1))
    return H, routing_stats


@torch.no_grad()
def deterministic_operator_rollout(
    model: TinyReasoner,
    rule: torch.Tensor,
    H: torch.Tensor,
    steps: int = 1,
):
    A = model.core.adjacency()
    chosen_steps = []
    for _ in range(steps):
        logits, _ = model.policy_value(H, rule)
        actions = logits[:, :model.operator_count].argmax(dim=-1)
        chosen_steps.append(actions)
        H = apply_selected_actions(model, H, actions, A)
    return H, torch.stack(chosen_steps, dim=1)


@torch.no_grad()
def apply_selected_actions(model, H, actions, adjacency=None):
    if adjacency is None:
        adjacency = model.core.adjacency()
    all_next = torch.stack(
        [model.core.apply_operator(H, k, adjacency) for k in range(model.operator_count)],
        dim=1,
    )
    selector = F.one_hot(actions, num_classes=model.operator_count).to(H.dtype)
    return torch.einsum("bk,bknd->bnd", selector, all_next)


@torch.no_grad()
def demo_guided_two_step_search(model, probe_x, probe_y, query_x):
    """Search all 22x22 programs on a known probe pair.

    Execution is rule-blind: candidate operators act only on query-derived Q.
    The known probe output is used solely as a program-selection objective.
    """
    B = query_x.shape[0]
    K = model.operator_count
    A = model.core.adjacency()
    H_probe = model.encode_query(probe_x)
    first = torch.stack([model.core.apply_operator(H_probe, k, A) for k in range(K)], dim=1)
    first_flat = first.reshape(B * K, first.shape[-2], first.shape[-1])
    second = torch.stack([model.core.apply_operator(first_flat, k, A) for k in range(K)], dim=1)
    all_states = second.reshape(B, K, K, second.shape[-2], second.shape[-1])
    flat_states = all_states.reshape(B * K * K, second.shape[-2], second.shape[-1])
    pred_probe = model.decode_query(flat_states).reshape(B, K, K, -1)
    score = (pred_probe - probe_y[:, None, None, :]).pow(2).mean(dim=-1)
    best_flat = score.reshape(B, K * K).argmin(dim=-1)
    first_action = best_flat // K
    second_action = best_flat % K
    sequences = torch.stack([first_action, second_action], dim=-1)
    best_probe_mse = score.reshape(B, K * K).gather(1, best_flat[:, None]).squeeze(1)
    H_query = model.encode_query(query_x)
    H_query = apply_selected_actions(model, H_query, first_action, A)
    H_query = apply_selected_actions(model, H_query, second_action, A)
    return model.decode_query(H_query), sequences, best_probe_mse


def operator_separation_loss(
    model: TinyReasoner,
    H: torch.Tensor,
    margin: float = 0.03,
    max_batch: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encourage distinct operator effects without rewarding unbounded motion.

    The loss is a normalized margin penalty on pairwise distances between
    operator *outputs*. Once two operators differ by at least `margin`, there is
    no further reward for pushing them apart.
    """
    Hs = H[:max_batch]
    A = model.core.adjacency()
    states = torch.stack(
        [model.core.apply_operator(Hs, k, A) for k in range(model.operator_count)],
        dim=1,
    )
    diff = states[:, :, None] - states[:, None, :]
    dist = (diff.pow(2).mean(dim=(-1, -2)) + 1e-8).sqrt()
    upper = torch.triu(
        torch.ones(model.operator_count, model.operator_count, device=H.device, dtype=torch.bool),
        diagonal=1,
    )
    pair = dist[:, upper]
    loss = F.relu(1.0 - pair / margin).pow(2).mean()
    return loss, pair.mean()


# ---------------------------------------------------------------------------
# Local antithetic ES around the generative subsystem
# ---------------------------------------------------------------------------

def genotype_modules(model: TinyReasoner) -> nn.Module:
    """
    For V0 we treat the CPPN + operator hypernetwork + operator codes as the
    continuous 'genotype-like' parameters.

    To keep this easy to flatten, wrap references in a lightweight container.
    """
    holder = nn.Module()
    holder.add_module("cppn", model.core.cppn)
    holder.add_module("op_hyper", model.core.op_hyper)

    # operator_codes is not a module, so flatten ES manually elsewhere.
    return holder


def get_genotype_vector(model: TinyReasoner) -> torch.Tensor:
    chunks = [
        flatten_params(model.core.cppn),
        flatten_params(model.core.op_hyper),
        model.core.operator_codes.detach().reshape(-1),
    ]
    return torch.cat(chunks)


@torch.no_grad()
def set_genotype_vector_(model: TinyReasoner, flat: torch.Tensor):
    n_cppn = sum(p.numel() for p in model.core.cppn.parameters() if p.requires_grad)
    n_op = sum(p.numel() for p in model.core.op_hyper.parameters() if p.requires_grad)

    set_flat_params_(model.core.cppn, flat[:n_cppn])
    set_flat_params_(model.core.op_hyper, flat[n_cppn:n_cppn+n_op])

    code_flat = flat[n_cppn+n_op:]
    model.core.operator_codes.copy_(code_flat.view_as(model.core.operator_codes))


@torch.no_grad()
def evaluate_fixed_batch(
    model: TinyReasoner,
    demo_x: torch.Tensor,
    demo_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    rollout_steps: int = 1,
) -> float:
    rule = model.encode_rule(demo_x, demo_y)
    H0 = model.encode_query(query_x)
    H, _ = differentiable_rollout(model, rule, H0, steps=rollout_steps, hard=True)
    pred = model.decode_query(H)
    return float(F.mse_loss(pred, query_y).item())


@torch.no_grad()
def antithetic_es_step_(
    model: TinyReasoner,
    task_source: SyntheticTaskBatch,
    device,
    pairs: int = 4,
    sigma: float = 0.01,
    lr: float = 0.02,
    eval_batch_size: int = 32,
    rollout_steps: int = 3,
):
    """
    Minimize loss using a tiny antithetic ES neighborhood.

    Important: perturb the compact generative parameter vector, not the
    generated adjacency/output tensors.
    """
    base = get_genotype_vector(model)
    (
        demo_x,
        demo_y,
        query_x,
        query_y,
        _,
    ) = task_source.sample(
        eval_batch_size,
        device,
    )
    noises = []
    losses_plus = []
    losses_minus = []

    # Use the exact same batch for +epsilon and -epsilon so the
    # antithetic difference isolates the parameter perturbation.
    for _ in range(pairs):
        eps = torch.randn_like(base)
        noises.append(eps)

        set_genotype_vector_(model, base + sigma * eps)
        lp = evaluate_fixed_batch(
            model,
            demo_x,
            demo_y,
            query_x,
            query_y,
            rollout_steps,
        )
        set_genotype_vector_(model, base - sigma * eps)
        lm = evaluate_fixed_batch(
            model,
            demo_x,
            demo_y,
            query_x,
            query_y,
            rollout_steps,
        )

        losses_plus.append(lp)
        losses_minus.append(lm)

    set_genotype_vector_(model, base)

    # Gradient estimate for minimizing loss.
    g = torch.zeros_like(base)
    for eps, lp, lm in zip(noises, losses_plus, losses_minus):
        g += (lp - lm) * eps
    g /= (2.0 * pairs * sigma)

    # Only normalize/apply ES when the antithetic signal is meaningfully nonzero.
    raw_grad_norm = g.norm()
    if raw_grad_norm > 1e-6:
        g = g / raw_grad_norm
        set_genotype_vector_(model, base - lr * g)
    else:
        set_genotype_vector_(model, base)

    return {
        "es_loss_plus": sum(losses_plus) / len(losses_plus),
        "es_loss_minus": sum(losses_minus) / len(losses_minus),
        "es_grad_norm": float(raw_grad_norm.item()),
    }


@torch.no_grad()
def evaluate_held_out(
    model: TinyReasoner,
    task_source: SyntheticTaskBatch,
    device,
    batch_size: int = 1024,
):
    """Evaluate factorized rule inference, direct baseline, and query autoencoding."""
    model.eval()
    demo_x, demo_y, query_x, query_y, task_ids = task_source.sample(batch_size, device)
    rule = model.encode_rule(demo_x, demo_y)
    H0 = model.encode_query(query_x)
    direct = model.decode_direct(H0, rule)
    identity = model.decode_query(H0)
    task_logits = model.task_logits_rule(rule)
    direct_per = F.mse_loss(direct, query_y, reduction="none").mean(dim=-1)
    identity_per = F.mse_loss(identity, query_x, reduction="none").mean(dim=-1)
    task_acc = (task_logits.argmax(dim=-1) == task_ids).float().mean()
    per_task = {}
    for task_id, name in enumerate(task_source.TRAIN_NAMES):
        mask = task_ids == task_id
        if mask.any():
            per_task[name] = (
                float(direct_per[mask].mean().item()),
                float(identity_per[mask].mean().item()),
                float((task_logits[mask].argmax(dim=-1) == task_ids[mask]).float().mean().item()),
                int(mask.sum().item()),
            )
    return {
        "direct_mse": float(direct_per.mean().item()),
        "identity_mse": float(identity_per.mean().item()),
        "task_acc_rule": float(task_acc.item()),
        "per_task": per_task,
    }


@torch.no_grad()
def _sequence_summary(sequences: torch.Tensor, top_k: int = 5):
    patterns = Counter(tuple(int(v) for v in row) for row in sequences.cpu().tolist())
    total = max(sum(patterns.values()), 1)
    return [(pattern, count, count / total) for pattern, count in patterns.most_common(top_k)]


@torch.no_grad()
def evaluate_operator_rollout(model: TinyReasoner, task_source: SyntheticTaskBatch, device, batch_size: int = 1024, rollout_steps: int = 1):
    model.eval()
    demo_x, demo_y, query_x, query_y, task_ids = task_source.sample(batch_size, device)
    rule = model.encode_rule(demo_x, demo_y)
    H0 = model.encode_query(query_x)
    direct = model.decode_direct(H0, rule)
    H, sequences = deterministic_operator_rollout(model, rule, H0, steps=rollout_steps)
    pred = model.decode_query(H)
    direct_per = F.mse_loss(direct, query_y, reduction="none").mean(dim=-1)
    rollout_per = F.mse_loss(pred, query_y, reduction="none").mean(dim=-1)
    target_energy = query_y.pow(2).mean(dim=-1).clamp_min(1e-6)
    first_actions = sequences[:, 0]
    usage = torch.bincount(first_actions, minlength=model.operator_count).float() / batch_size
    top_usage = torch.topk(usage, k=min(8, model.operator_count))
    sep, pair = operator_separation_loss(model, H0, max_batch=min(32, batch_size))
    per_task = {}
    for task_id, name in enumerate(task_source.TRAIN_NAMES):
        mask = task_ids == task_id
        if mask.any():
            per_task[name] = {
                "direct_mse": float(direct_per[mask].mean().item()),
                "rollout_mse": float(rollout_per[mask].mean().item()),
                "rollout_nmse": float((rollout_per[mask] / target_energy[mask]).mean().item()),
                "sequences": _sequence_summary(sequences[mask]),
            }
    return {
        "direct_mse": float(direct_per.mean().item()),
        "rollout_mse": float(rollout_per.mean().item()),
        "rollout_nmse": float((rollout_per / target_energy).mean().item()),
        "op_pair": float(pair.item()),
        "sep_loss": float(sep.item()),
        "top_ops": top_usage.indices.tolist(),
        "top_usage": top_usage.values.tolist(),
        "per_task": per_task,
    }


@torch.no_grad()
def evaluate_compositional_generalization(model, task_source, device, batch_size: int = 128):
    model.eval()
    results = {}
    for name, primitive_sequence in task_source.COMPOSITIONS.items():
        support_x, support_y, probe_x, probe_y, query_x, query_y = task_source.sample_composition_triplet(name, batch_size, device)
        rule = model.encode_rule(support_x, support_y)
        H0 = model.encode_query(query_x)
        direct_pred = model.decode_direct(H0, rule)
        direct_per = F.mse_loss(direct_pred, query_y, reduction="none").mean(dim=-1)
        H_greedy, greedy_sequences = deterministic_operator_rollout(model, rule, H0, steps=2)
        greedy_pred = model.decode_query(H_greedy)
        greedy_per = F.mse_loss(greedy_pred, query_y, reduction="none").mean(dim=-1)
        search_pred, search_sequences, probe_mse = demo_guided_two_step_search(model, probe_x, probe_y, query_x)
        search_per = F.mse_loss(search_pred, query_y, reduction="none").mean(dim=-1)

        # Apply the *known anchored primitive program* directly. If this is low,
        # the learned operators themselves form the intended algebra regardless
        # of whether the OOD policy/search identifies that program.
        H_oracle = H0
        for action in primitive_sequence:
            H_oracle = model.core.apply_operator(H_oracle, int(action))
        oracle_pred = model.decode_query(H_oracle)
        oracle_per = F.mse_loss(oracle_pred, query_y, reduction="none").mean(dim=-1)
        target_energy = query_y.pow(2).mean(dim=-1).clamp_min(1e-6)
        results[name] = {
            "primitive_sequence": primitive_sequence,
            "direct_mse": float(direct_per.mean().item()),
            "greedy_mse": float(greedy_per.mean().item()),
            "search_mse": float(search_per.mean().item()),
            "oracle_program_mse": float(oracle_per.mean().item()),
            "oracle_program_nmse": float((oracle_per / target_energy).mean().item()),
            "search_nmse": float((search_per / target_energy).mean().item()),
            "search_improvement": float((direct_per.mean() - search_per.mean()).item()),
            "probe_mse": float(probe_mse.mean().item()),
            "greedy_sequences": _sequence_summary(greedy_sequences),
            "search_sequences": _sequence_summary(search_sequences),
        }
    return results


def set_query_codec_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze the task-independent query coordinate system.

    Once warmup has learned E_Q(x) <-> x, operator learning should happen
    inside that fixed coordinate system rather than moving the coordinate
    system to accommodate each primitive.
    """
    for module in (model.query_encoder, model.query_decoder):
        for param in module.parameters():
            param.requires_grad_(trainable)




def primitive_transport_sources(primitive_id: int, n: int, device) -> torch.Tensor:
    dst = torch.arange(n, device=device)
    if primitive_id == 0:  # torch.roll(x, +1): y[dst] = x[dst-1]
        return (dst - 1) % n
    if primitive_id == 1:  # negate keeps position
        return dst
    if primitive_id == 2:  # flip: y[dst] = x[n-1-dst]
        return (n - 1) - dst
    if primitive_id == 3:  # identity
        return dst
    raise ValueError(primitive_id)


def transport_supervision_loss(model: TinyReasoner) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """Directly supervise the four anchored primitive transport matrices.

    P[dst, src] is row-stochastic. We minimize -log P[dst, desired_src].
    Returns the mean CE plus per-primitive (argmax accuracy, normalized row entropy).
    """
    n = model.n_nodes
    losses = []
    stats = {}
    for pid in range(4):
        code = model.core.operator_codes[pid]
        P = model.core.op_hyper.transport(model.core.coords, code)
        target_src = primitive_transport_sources(pid, n, P.device)
        chosen = P[torch.arange(n, device=P.device), target_src].clamp_min(1e-8)
        losses.append(-chosen.log().mean())
        acc = (P.argmax(dim=-1) == target_src).float().mean()
        ent = -(P * torch.log(P.clamp_min(1e-8))).sum(dim=-1).mean() / math.log(n)
        stats[pid] = (acc, ent)
    return torch.stack(losses).mean(), stats

def set_algebra_trainable(model: TinyReasoner, trainable: bool):
    """Freeze/unfreeze the learned operator algebra while training program inference."""
    for p in model.core.parameters():
        p.requires_grad_(trainable)


def freeze_for_program_phase(model: TinyReasoner):
    """Preserve the validated algebra; train only goal policy + goal value."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.program_controller, model.program_value):
        for p in module.parameters():
            p.requires_grad_(True)


@torch.no_grad()
def evaluate_greedy_program_accuracy(model, task_source, device, batch_size: int = 256):
    """v14 held-out inference using only support-current -> support-goal feedback."""
    model.eval()
    out = {}
    for name, seq in task_source.COMPOSITIONS.items():
        demo_x, demo_y, query_x, query_y = task_source.sample_composition(name, batch_size, device)
        support = model.encode_query(demo_x)
        target_support = model.encode_query(demo_y)
        query = model.encode_query(query_x)
        chosen_steps = []
        for step_idx in range(2):
            logits = model.program_policy(support, target_support, remaining_steps=2 - step_idx)
            action = logits.argmax(dim=-1)
            chosen_steps.append(action)
            support = apply_selected_actions(model, support, action)
            query = apply_selected_actions(model, query, action)
        chosen = torch.stack(chosen_steps, dim=1)
        target = torch.tensor(seq, device=device, dtype=torch.long)[None, :].expand(batch_size, -1)
        exact = (chosen == target).all(dim=-1).float().mean()
        pred = model.decode_query(query)
        mse = F.mse_loss(pred, query_y)
        out[name] = (float(exact.item()), float(mse.item()))
    return out


@torch.no_grad()
def evaluate_muzero_program_search(
    model, task_source, device, batch_size: int = 64, simulations: int = 64
):
    """Execute two sequential MCTS decisions from the support demonstration."""
    model.eval()
    out = {}
    for name, seq in task_source.COMPOSITIONS.items():
        demo_x, demo_y, query_x, query_y = task_source.sample_composition(name, batch_size, device)
        target_support = model.encode_query(demo_y)
        support = model.encode_query(demo_x)
        query = model.encode_query(query_x)
        actions = []
        for decision_idx in range(2):
            remaining = 2 - decision_idx
            cfg = SearchConfig(
                simulations=simulations, max_depth=remaining, action_limit=4
            )
            step_actions = []
            for b in range(batch_size):
                a, _, _ = goal_puct_search(
                    model, support[b:b+1], target_support[b:b+1], cfg
                )
                step_actions.append(a)
            action = torch.tensor(step_actions, device=device, dtype=torch.long)
            actions.append(action)
            support = apply_selected_actions(model, support, action)
            query = apply_selected_actions(model, query, action)
        chosen = torch.stack(actions, dim=1)
        target = torch.tensor(seq, device=device, dtype=torch.long)[None, :].expand(batch_size, -1)
        exact = (chosen == target).all(dim=-1).float().mean()
        mse = F.mse_loss(model.decode_query(query), query_y)
        out[name] = (float(exact.item()), float(mse.item()))
    return out


def train_smoke_test(
    steps: int = 1500,
    batch_size: int = 32,
    inner_rollout_steps: int = 1,
    warmup_steps: int = 250,
    algebra_steps: int = 1000,
    es_every: int = 0,
    diagnostic_every: int = 25,
    mcts_train_every: int = 5,
    mcts_train_samples: int = 4,
    mcts_simulations: int = 32,
    mcts_eval_batch: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_all(0)
    torch_device = torch.device(device)
    model = TinyReasoner().to(torch_device)
    tasks = SyntheticTaskBatch(dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"device={device}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"substrate_nodes={model.core.coords.shape[0]}")
    print(f"operators={model.operator_count} + HALT")
    print(f"mode=v16 warmup({warmup_steps}) -> supervised transport algebra -> frozen-algebra goal-conditioned program inference")
    print(f"es_every={es_every} mcts_every={mcts_train_every} mcts_samples={mcts_train_samples} sims={mcts_simulations}")

    for step in range(1, steps + 1):
        model.train()
        in_program_phase = step > algebra_steps
        if step == algebra_steps + 1:
            # The algebra is already established. From here on only learn how
            # to infer/operator-sequence programs from demonstrations.
            freeze_for_program_phase(model)

        if in_program_phase:
            demo_x, demo_y, query_x, query_y, program_targets = tasks.sample_program_batch(batch_size, device)
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            demo_x, demo_y, query_x, query_y, task_ids = tasks.sample(batch_size, device)
            program_targets = None
        rule = model.encode_rule(demo_x, demo_y)
        H0 = model.encode_query(query_x)

        # Stable task-independent query workspace.
        identity_pred = model.decode_query(H0)
        identity_loss = F.mse_loss(identity_pred, query_x)
        direct_pred = model.decode_direct(H0, rule)
        direct_loss = F.mse_loss(direct_pred, query_y)

        task_logits = model.task_logits_rule(rule)
        if in_program_phase:
            task_loss = torch.zeros((), device=torch_device)
            task_acc = torch.ones((), device=torch_device)
        else:
            task_loss = F.cross_entropy(task_logits, task_ids)
            task_acc = (task_logits.argmax(dim=-1) == task_ids).float().mean()

        in_warmup = step <= warmup_steps
        # Freeze the query coordinate system exactly when operator training starts.
        # The optimizer may still contain these parameters; requires_grad=False
        # prevents subsequent operator losses from moving them.
        if step == warmup_steps + 1:
            set_query_codec_trainable(model, False)
        routing_stats = []

        # Synthetic-stage semantic anchors: four primitive tasks map to four
        # distinct operator identities. This isolates algebra/composition from
        # unsupervised symbol discovery; the other 18 operators remain free.
        primitive_operator_targets = task_ids  # task 0..3 -> operator 0..3
        policy_logits_h0, _ = model.policy_value(H0, rule)
        if in_program_phase:
            # v14 teacher forcing occurs in SUPPORT space. The controller sees
            # current demo state and final demo target, chooses the next action,
            # then observes the teacher-updated support state for step two.
            support_H0 = model.encode_query(demo_x)
            support_target = model.encode_query(demo_y)
            prog_logits_1 = model.program_policy(support_H0, support_target, remaining_steps=2)
            program_ce_1 = F.cross_entropy(prog_logits_1, program_targets[:, 0])
            support_teacher_1 = apply_selected_actions(model, support_H0, program_targets[:, 0])
            prog_logits_2 = model.program_policy(support_teacher_1, support_target, remaining_steps=1)
            program_ce_2 = F.cross_entropy(prog_logits_2, program_targets[:, 1])
            route_supervision_loss = 0.5 * (program_ce_1 + program_ce_2)
            program_acc_1 = (prog_logits_1.argmax(dim=-1) == program_targets[:, 0]).float().mean()
            program_acc_2 = (prog_logits_2.argmax(dim=-1) == program_targets[:, 1]).float().mean()

            # v16 finite-horizon Bellman value training. The value network is
            # trained on root/intermediate/off-policy states with the number of
            # actions remaining, and targets the *best reachable terminal fit*
            # rather than merely current-state closeness.
            horizon_state_pairs = [(support_H0, 2), (support_teacher_1, 1)]
            one_step_states = []
            for a in range(4):
                aa = torch.full((batch_size,), a, device=device, dtype=torch.long)
                s1 = apply_selected_actions(model, support_H0, aa)
                one_step_states.append(s1)
                horizon_state_pairs.append((s1, 1))
                horizon_state_pairs.append((s1, 0))
            # Include all depth-2 leaves with h=0 so terminal calibration is
            # learned on exactly the leaf distribution used by PUCT.
            for s1 in one_step_states:
                for b_action in range(4):
                    bb = torch.full((batch_size,), b_action, device=device, dtype=torch.long)
                    s2 = apply_selected_actions(model, s1, bb)
                    horizon_state_pairs.append((s2, 0))

            goal_value_losses = []
            for vs, horizon in horizon_state_pairs:
                _, vv = model.program_policy_value(vs, support_target, horizon)
                vt = exact_horizon_value_target(
                    model, vs, support_target, horizon, action_limit=4, beta=2.0
                )
                goal_value_losses.append(F.mse_loss(vv, vt))
            goal_value_loss = torch.stack(goal_value_losses).mean()

            # MuZero policy improvement: periodically run PUCT on a few support
            # states and distill normalized visit counts into the goal policy.
            mcts_policy_loss = torch.zeros((), device=torch_device)
            mcts_root_entropy = torch.zeros((), device=torch_device)
            if mcts_train_every > 0 and step % mcts_train_every == 0:
                cfg_root = SearchConfig(
                    simulations=mcts_simulations, max_depth=2, action_limit=4
                )
                cfg_second = SearchConfig(
                    simulations=mcts_simulations, max_depth=1, action_limit=4
                )
                n_mcts = min(mcts_train_samples, batch_size)
                mcts_losses = []
                mcts_entropies = []
                for b in range(n_mcts):
                    # Root target.
                    _, visit0, _ = goal_puct_search(
                        model, support_H0[b:b+1].detach(), support_target[b:b+1].detach(), cfg_root
                    )
                    live0 = model.program_policy(support_H0[b:b+1], support_target[b:b+1], remaining_steps=2)[:, :4]
                    mcts_losses.append(soft_policy_cross_entropy(live0, visit0[None, :]))
                    mcts_entropies.append(
                        -(visit0 * torch.log(visit0.clamp_min(1e-8))).sum() / math.log(4)
                    )
                    # Teacher-reached second decision state.
                    _, visit1, _ = goal_puct_search(
                        model, support_teacher_1[b:b+1].detach(), support_target[b:b+1].detach(), cfg_second
                    )
                    live1 = model.program_policy(support_teacher_1[b:b+1], support_target[b:b+1], remaining_steps=1)[:, :4]
                    mcts_losses.append(soft_policy_cross_entropy(live1, visit1[None, :]))
                if mcts_losses:
                    mcts_policy_loss = torch.stack(mcts_losses).mean()
                    mcts_root_entropy = torch.stack(mcts_entropies).mean()
        else:
            route_supervision_loss = F.cross_entropy(
                policy_logits_h0[:, :model.operator_count], primitive_operator_targets
            )
            program_ce_1 = program_ce_2 = torch.zeros((), device=torch_device)
            program_acc_1 = program_acc_2 = torch.zeros((), device=torch_device)
            goal_value_loss = torch.zeros((), device=torch_device)
            mcts_policy_loss = torch.zeros((), device=torch_device)
            mcts_root_entropy = torch.zeros((), device=torch_device)
        if in_warmup:
            H = H0
            pred = direct_pred
            recon = direct_loss
        else:
            if in_program_phase:
                # Keep reconstruction tied to the known closed algebra while the
                # controller learns program identification; do not backprop into
                # operator execution in this phase.
                H = H0
                H = apply_selected_actions(model, H, program_targets[:, 0])
                H = apply_selected_actions(model, H, program_targets[:, 1])
                pred = model.decode_query(H)
                recon = F.mse_loss(pred, query_y)
                routing_stats = [torch.softmax(prog_logits_1, dim=-1), torch.softmax(prog_logits_2, dim=-1)]
            else:
                progress = (step - warmup_steps) / max(algebra_steps - warmup_steps, 1)
                temperature = max(0.30, 1.0 - 0.70 * progress)
                H, routing_stats = differentiable_rollout(
                    model, rule, H0, steps=inner_rollout_steps,
                    temperature=temperature, hard=True,
                )
                pred = model.decode_query(H)
                recon = F.mse_loss(pred, query_y)

        per_sample_error = F.mse_loss(pred, query_y, reduction="none").mean(dim=-1)
        value_target = torch.exp(-per_sample_error.detach())
        _, value = model.policy_value(H, rule)
        value_loss = F.mse_loss(value, value_target)

        codes = F.normalize(model.core.operator_codes, dim=-1)
        gram = codes @ codes.t()
        eye = torch.eye(model.operator_count, device=torch_device)
        diversity = ((gram - eye) ** 2).mean()
        A = model.core.adjacency()
        graph_reg = A.pow(2).mean()

        op_sep_loss = torch.zeros((), device=torch_device)
        op_pair_train = torch.zeros((), device=torch_device)
        route_entropy = torch.zeros((), device=torch_device)
        task_route_entropy = torch.zeros((), device=torch_device)
        task_route_overlap = torch.zeros((), device=torch_device)
        oracle_atomic_loss = torch.zeros((), device=torch_device)
        latent_transition_loss = torch.zeros((), device=torch_device)
        latent_cosine = torch.zeros((), device=torch_device)
        transport_loss = torch.zeros((), device=torch_device)
        transport_stats = {}

        if (not in_warmup) and (not in_program_phase):
            op_sep_loss, op_pair_train = operator_separation_loss(model, H0)
            route_p = torch.stack(routing_stats, dim=0).mean(dim=0)
            route_entropy = -(route_p * torch.log(route_p + 1e-8)).sum(dim=-1).mean() / math.log(model.operator_count)

            # Same primitive should prefer a consistent routing prototype.
            prototypes = []
            proto_entropies = []
            for tid in range(len(tasks.TRAIN_NAMES)):
                mask = task_ids == tid
                if mask.any():
                    proto = route_p[mask].mean(dim=0)
                    proto = proto / proto.sum().clamp_min(1e-8)
                    prototypes.append(proto)
                    proto_entropies.append(-(proto * torch.log(proto + 1e-8)).sum() / math.log(model.operator_count))
            if proto_entropies:
                task_route_entropy = torch.stack(proto_entropies).mean()
            if len(prototypes) > 1:
                P = torch.stack(prototypes, dim=0)
                P = F.normalize(P, dim=-1)
                sim = P @ P.t()
                mask = ~torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
                task_route_overlap = sim[mask].mean()

            # Train the designated primitive operator directly on every sample so
            # early routing mistakes cannot starve it of useful gradients.
            A_oracle = model.core.adjacency()
            oracle_states = []
            for b_idx in range(H0.shape[0]):
                oracle_states.append(
                    model.core.apply_operator(
                        H0[b_idx:b_idx + 1],
                        int(primitive_operator_targets[b_idx].item()),
                        A_oracle,
                    )
                )
            H_oracle = torch.cat(oracle_states, dim=0)
            oracle_atomic_loss = F.mse_loss(model.decode_query(H_oracle), query_y)

            # Explicit algebraic closure target: an operator applied to the
            # encoding of x should land at the encoding of T_k(x).  The target
            # is detached because the query coordinate system is fixed after
            # warmup.  This is the key compositional objective in v10.
            with torch.no_grad():
                H_target = model.encode_query(query_y)
            # Do not average the closure error away over feature width. v10's
            # ordinary MSE diluted a one-channel positional error by node_dim.
            # Sum over feature channels, then average over batch and vertices.
            latent_transition_loss = (H_oracle - H_target).pow(2).sum(dim=-1).mean()
            h_oracle_flat = F.normalize(H_oracle.flatten(1), dim=-1)
            h_target_flat = F.normalize(H_target.flatten(1), dim=-1)
            latent_cosine = (h_oracle_flat * h_target_flat).sum(dim=-1).mean()

            # v12: directly teach the code-conditioned transport law the exact
            # permutation associated with each anchored primitive. This is
            # synthetic-stage supervision used to test whether the generated
            # operator family can represent a closed algebra at all.
            transport_loss, transport_stats = transport_supervision_loss(model)

        if in_program_phase:
            # Program phase: freeze the learned algebra and optimize only rule
            # inference/controller sequencing. A small primitive rehearsal batch
            # below keeps single-step routing from drifting.
            p_demo_x, p_demo_y, _, _, p_task_ids = tasks.sample(batch_size, device)
            p_support = model.encode_query(p_demo_x)
            p_target = model.encode_query(p_demo_y)
            p_logits = model.program_policy(p_support, p_target, remaining_steps=1)
            primitive_rehearsal = F.cross_entropy(p_logits, p_task_ids)
            # Only program_controller has gradients in this phase. Do not add
            # value/diversity losses from frozen modules to the optimization.
            loss = (
                0.25 * route_supervision_loss
                + 0.25 * primitive_rehearsal
                + 1.00 * goal_value_loss
                + 1.00 * mcts_policy_loss
            )
        elif in_warmup:
            loss = (
                direct_loss
                + 0.50 * identity_loss
                + 0.10 * task_loss
                + 0.10 * route_supervision_loss
                + 0.25 * value_loss
                + 0.01 * diversity
                + 1e-4 * graph_reg
            )
        else:
            loss = (
                recon
                + 0.00 * identity_loss
                + 0.10 * direct_loss
                + 0.10 * task_loss
                + 0.25 * value_loss
                + 0.01 * diversity
                + 1e-4 * graph_reg
                + 0.08 * op_sep_loss
                + 0.50 * route_supervision_loss
                + 0.50 * oracle_atomic_loss
                + 1.00 * latent_transition_loss
                + 1.50 * transport_loss
                + 0.010 * task_route_entropy
                + 0.010 * task_route_overlap
            )

        mean_op_delta = max_op_delta = normalized_entropy = operator_pair_distance = None
        if diagnostic_every > 0 and (step == 1 or step % diagnostic_every == 0):
            with torch.no_grad():
                A_diag = model.core.adjacency()
                op_states = torch.stack(
                    [model.core.apply_operator(H0, k, A_diag) for k in range(model.operator_count)],
                    dim=1,
                )
                deltas = (op_states - H0[:, None]).pow(2).mean(dim=(-1, -2)).sqrt()
                mean_op_delta = deltas.mean()
                max_op_delta = deltas.max()
                pair_diff = op_states[:, :, None] - op_states[:, None, :]
                pair_dist = (pair_diff.pow(2).mean(dim=(-1, -2)) + 1e-8).sqrt()
                upper = torch.triu(torch.ones(model.operator_count, model.operator_count, device=H0.device, dtype=torch.bool), diagonal=1)
                operator_pair_distance = pair_dist[:, upper].mean()
                logits_diag, _ = model.policy_value(H0, rule)
                p_diag = torch.softmax(logits_diag[:, :model.operator_count], dim=-1)
                entropy = -(p_diag * torch.log(p_diag + 1e-8)).sum(dim=-1).mean()
                normalized_entropy = entropy / math.log(model.operator_count)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        es_info = None
        if es_every > 0 and step % es_every == 0:
            model.eval()
            es_info = antithetic_es_step_(model, tasks, device, pairs=4, sigma=0.005, lr=0.01, eval_batch_size=24, rollout_steps=inner_rollout_steps)

        if step == 1 or step % 25 == 0:
            msg = (
                f"step={step:04d} loss={loss.item():.5f} recon={recon.item():.5f} "
                f"direct={direct_loss.item():.5f} identity={identity_loss.item():.5f} "
                f"task={task_loss.item():.4f} taskAcc={task_acc.item():.3f} "
                f"value={value_loss.item():.5f} div={diversity.item():.5f} "
                f"phase={'program' if in_program_phase else ('warmup' if in_warmup else 'ops')}"
                f" routeCE={route_supervision_loss.item():.4f}"
            )
            if in_program_phase:
                msg += (
                    f" progCE1={program_ce_1.item():.4f} progCE2={program_ce_2.item():.4f}"
                    f" progAcc1={program_acc_1.item():.3f} progAcc2={program_acc_2.item():.3f}"
                    f" goalV={goal_value_loss.item():.4f} mctsCE={mcts_policy_loss.item():.4f}"
                    f" mctsH={mcts_root_entropy.item():.3f}"
                )
            elif not in_warmup:
                msg += (
                    f" sep={op_sep_loss.item():.4f} trainPair={op_pair_train.item():.5f}"
                    f" routeH={route_entropy.item():.3f}"
                    f" taskRouteH={task_route_entropy.item():.3f}"
                    f" taskOverlap={task_route_overlap.item():.3f}"
                    f" routeCE={route_supervision_loss.item():.4f}"
                    f" oracle={oracle_atomic_loss.item():.5f}"
                    f" latent={latent_transition_loss.item():.5f}"
                    f" latentCos={latent_cosine.item():.3f}"
                    f" transportCE={transport_loss.item():.4f}"
                    + (
                        " " + " ".join(
                            f"T{pid}Acc={transport_stats[pid][0].item():.2f}/H={transport_stats[pid][1].item():.2f}"
                            for pid in sorted(transport_stats)
                        ) if transport_stats else ""
                    )
                )
            if mean_op_delta is not None:
                msg += (
                    f" opΔ={mean_op_delta.item():.5f} opΔmax={max_op_delta.item():.5f}"
                    f" opPair={operator_pair_distance.item():.5f} policyH={normalized_entropy.item():.3f}"
                )
            print(msg)

    heldout_program = evaluate_greedy_program_accuracy(model, tasks, device, batch_size=256)
    print("\nHeld-out goal-conditioned program inference (exact programs never used in program-training phase)")
    for name, (exact, mse) in heldout_program.items():
        print(f"  {name:16s} exact={exact:.3f} greedyMSE={mse:.5f}")

    mcts_eval = evaluate_muzero_program_search(
        model, tasks, device, batch_size=mcts_eval_batch, simulations=mcts_simulations
    )
    print("\nHeld-out finite-horizon MuZero-style goal-conditioned PUCT (2 decisions)")
    for name, (exact, mse) in mcts_eval.items():
        print(f"  {name:16s} exact={exact:.3f} mctsMSE={mse:.5f}")

    checkpoint_path = "yetirah_v16_posttrain.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "steps": steps}, checkpoint_path)
    print(f"\nsaved checkpoint: {checkpoint_path}")

    held_out = evaluate_held_out(model, tasks, device, batch_size=1024)
    print("\nHeld-out factorized representation diagnostics (1024 fresh tasks)")
    print(
        f"direct MSE: {held_out['direct_mse']:.5f} | query-identity MSE: {held_out['identity_mse']:.5f} | "
        f"rule task-ID: {held_out['task_acc_rule']:.3f}"
    )
    for name, (direct_mse, identity_mse, task_acc, count) in held_out["per_task"].items():
        print(f"  {name:8s} n={count:4d} direct={direct_mse:.5f} identity={identity_mse:.5f} taskRule={task_acc:.3f}")

    op_eval = evaluate_operator_rollout(model, tasks, device, batch_size=1024, rollout_steps=inner_rollout_steps)
    print("\nHeld-out atomic operator diagnostics (1024 fresh tasks)")
    print(
        f"direct MSE: {op_eval['direct_mse']:.5f} | operator-rollout MSE: {op_eval['rollout_mse']:.5f} | "
        f"opPair: {op_eval['op_pair']:.5f} | sepLoss: {op_eval['sep_loss']:.4f}"
    )
    print("top routed operators:", list(zip(op_eval['top_ops'], [round(x, 3) for x in op_eval['top_usage']])))
    print(f"rollout normalized MSE: {op_eval['rollout_nmse']:.5f}")
    print("primitive operator patterns:")
    for name, stats in op_eval["per_task"].items():
        print(f"  {name:8s} direct={stats['direct_mse']:.5f} rollout={stats['rollout_mse']:.5f} nmse={stats['rollout_nmse']:.4f}")
        for pattern, count, frac in stats["sequences"][:3]:
            print(f"    {pattern}  n={count:3d} frac={frac:.3f}")

    comp_eval = evaluate_compositional_generalization(model, tasks, device, batch_size=128)
    print("\nZERO-SHOT factorized compositional generalization")
    print("  greedy = OOD rule-conditioned policy | search = rule-blind exhaustive 2-step operator program search")
    for name, stats in comp_eval.items():
        print(
            f"  {name:16s} primitives={stats['primitive_sequence']} direct={stats['direct_mse']:.5f} "
            f"greedy={stats['greedy_mse']:.5f} search={stats['search_mse']:.5f} "
            f"oracle={stats['oracle_program_mse']:.5f}/{stats['oracle_program_nmse']:.4f} "
            f"nmse={stats['search_nmse']:.4f} improve={stats['search_improvement']:+.5f} probe={stats['probe_mse']:.5f}"
        )
        print("    greedy sequences:")
        for pattern, count, frac in stats["greedy_sequences"][:3]:
            print(f"      {pattern}  n={count:3d} frac={frac:.3f}")
        print("    searched sequences:")
        for pattern, count, frac in stats["search_sequences"][:3]:
            print(f"      {pattern}  n={count:3d} frac={frac:.3f}")

    # Goal-conditioned PUCT diagnostic on a held-out composition.
    model.eval()
    demo_x, demo_y, _, _ = tasks.sample_composition("roll2", 1, device)
    support = model.encode_query(demo_x)
    target_support = model.encode_query(demo_y)
    action, probs, root_err = goal_puct_search(
        model, support, target_support, SearchConfig(simulations=mcts_simulations)
    )
    print("\nFinite-horizon MuZero/PUCT smoke test (roll2)")
    print("root support-goal error:", root_err)
    print("chosen action:", action)
    top = torch.topk(probs, k=min(4, probs.numel()))
    print("top policy visits:", top.indices.tolist())
    print("top visit probs:", top.values.tolist())

    return model


if __name__ == "__main__":
    train_smoke_test()
