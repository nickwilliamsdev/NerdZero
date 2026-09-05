"""
yetirah_v0.py

Version 0 operator-training v4 of a geometry-generated operator model inspired by the architecture
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
    """
    Shared law for all 22 operators.

    Each operator has a compact learned code c_k.
    For each substrate location s_i, a shared hypernetwork emits:
       gamma, beta, alpha

    So operators differ by low-dimensional identity rather than independent
    networks.
    """

    def __init__(
        self,
        coord_dim: int = 5,
        code_dim: int = 8,
        hidden_dim: int = 32,
        node_dim: int = 32,
    ):
        super().__init__()
        self.node_dim = node_dim
        out = nn.Linear(hidden_dim, node_dim * 2 + 1)
        nn.init.normal_(out.weight, std=0.06)
        nn.init.zeros_(out.bias)

        self.net = nn.Sequential(
            nn.Linear(coord_dim + code_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            out,
        )


    def forward(
        self,
        coords: torch.Tensor,
        code: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        coords: [N, coord_dim]
        code: [code_dim]
        gamma,beta: [N,node_dim]
        alpha: [N,1]
        """
        n = coords.shape[0]
        c = code[None, :].expand(n, -1)
        out = self.net(torch.cat([coords, c], dim=-1))

        d = self.node_dim
        gamma = 1.0 + 0.20 * torch.tanh(out[:, :d])
        beta = 0.20 * torch.tanh(out[:, d:2*d])
        alpha = 0.15 * torch.sigmoid(out[:, 2*d:])
        return gamma, beta, alpha


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

        # Shared substrate message transform.
        self.message = nn.Linear(node_dim, node_dim, bias=False)
        nn.init.orthogonal_(self.message.weight, gain=0.5)

        self.norm = nn.LayerNorm(node_dim)

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
        gamma, beta, alpha = self.op_hyper(self.coords, code)

        # Geometry-defined message passing.
        msg = torch.einsum("ij,bjd->bid", adjacency, self.message(H))
        z = gamma[None] * msg + beta[None]
        delta = F.gelu(z)

        # Identity-like initialization: each operation is initially a small move.
        H_next = H + alpha[None] * delta
        return self.norm(H_next)

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

class TinyReasoner(nn.Module):
    """
    Minimal end-to-end wrapper.

    V0 input is a vector. Later, replace encoder/decoder with an ARC task
    representation while retaining YetirahCore.
    """

    HALT_ACTION = 22

    def __init__(
        self,
        input_dim: int = 32,
        node_dim: int = 32,
        coord_dim: int = 5,
        operator_count: int = 22,
        operator_code_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.operator_count = operator_count
        self.num_actions = operator_count + 1
        self.core = YetirahCore(
            coord_dim=coord_dim,
            node_dim=node_dim,
            operator_count=operator_count,
            operator_code_dim=operator_code_dim,
        )

        n_nodes = 2 ** coord_dim

        # Write demonstration/query context into the geometric workspace.
        # In addition to raw demo/query vectors, expose simple relational
        # features so the encoder can recognize rules such as negation, shifts,
        # and reversal without having to synthesize every comparison from one
        # affine projection. Six scalar correlation features are appended:
        # shifts -2,-1,0,+1,+2 and reversal correlation.
        relation_scalar_dim = 6
        encoder_input_dim = input_dim * 6 + relation_scalar_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, n_nodes * node_dim),
            nn.Tanh(),
        )

        # State summary.
        self.pool_query = nn.Parameter(torch.randn(node_dim) / math.sqrt(node_dim))

        # Operator routing reads the full substrate rather than only the pooled
        # summary. The v3 diagnostics showed task identity was nearly perfect in
        # H but substantially weaker after pooling. We keep compatibility against
        # learned operator codes so the action vocabulary remains generative.
        self.full_state_to_policy = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_nodes * node_dim, 128),
            nn.GELU(),
            nn.Linear(128, operator_code_dim),
        )
        self.halt_head = nn.Linear(node_dim, 1)

        self.value_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 1),
            nn.Tanh(),
        )

        # Diagnostic heads: compare pooled decoding against decoding from the
        # complete 32 x node_dim substrate. The full decoder is the primary
        # prediction path for this experiment; the pooled head remains as a
        # control so we can measure whether attention pooling is a bottleneck.
        self.pooled_decoder = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, input_dim),
        )

        self.full_decoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_nodes * node_dim, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

        # Diagnostic auxiliary heads: compare whether task identity is
        # recoverable from the pooled summary versus the complete substrate.
        self.task_probe_pooled = nn.Linear(node_dim, 4)
        self.task_probe_full = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_nodes * node_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )

    def encode(
        self,
        demo_x: torch.Tensor,
        demo_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        b = query_x.shape[0]
        n = self.core.coords.shape[0]

        delta = demo_y - demo_x
        summed = demo_y + demo_x
        product = demo_x * demo_y

        # Normalized demonstration correlations. These are intentionally small
        # diagnostic hints rather than a task ID: the downstream model still has
        # to infer which relation matters and apply it to a fresh query.
        def normalized_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            num = (a * b).mean(dim=-1, keepdim=True)
            den = (
                a.pow(2).mean(dim=-1, keepdim=True).sqrt()
                * b.pow(2).mean(dim=-1, keepdim=True).sqrt()
            ).clamp_min(1e-6)
            return num / den

        shift_corrs = [
            normalized_corr(torch.roll(demo_x, shifts=shift, dims=-1), demo_y)
            for shift in (-2, -1, 0, 1, 2)
        ]
        reverse_corr = normalized_corr(demo_x.flip(-1), demo_y)
        relation_scalars = torch.cat(shift_corrs + [reverse_corr], dim=-1)

        context = torch.cat(
            [
                demo_x,
                demo_y,
                delta,
                summed,
                product,
                query_x,
                relation_scalars,
            ],
            dim=-1,
        )

        return self.encoder(context).view(
            b,
            n,
            self.node_dim,
        )

    def pool(self, H: torch.Tensor) -> torch.Tensor:
        # Learned query attention over the 32 substrate locations.
        score = torch.einsum("bnd,d->bn", H, self.pool_query)
        attn = torch.softmax(score / math.sqrt(self.node_dim), dim=-1)
        return torch.einsum("bn,bnd->bd", attn, H)

    def policy_value(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.pool(H)

        q = self.full_state_to_policy(H)  # [B, code_dim]
        op_logits = q @ self.core.operator_codes.t() / math.sqrt(q.shape[-1])
        halt_logit = self.halt_head(z)
        logits = torch.cat([op_logits, halt_logit], dim=-1)

        value = self.value_head(z).squeeze(-1)
        return logits, value

    def transition(self, H: torch.Tensor, action: int) -> torch.Tensor:
        if action == self.HALT_ACTION:
            return H
        return self.core.apply_operator(H, action)

    def decode_pooled(self, H: torch.Tensor) -> torch.Tensor:
        return self.pooled_decoder(self.pool(H))

    def decode_full(self, H: torch.Tensor) -> torch.Tensor:
        return self.full_decoder(H)

    def decode(self, H: torch.Tensor) -> torch.Tensor:
        # Primary path for the V0 diagnostic.
        return self.decode_full(H)

    def task_logits_pooled(self, H: torch.Tensor) -> torch.Tensor:
        return self.task_probe_pooled(self.pool(H))

    def task_logits_full(self, H: torch.Tensor) -> torch.Tensor:
        return self.task_probe_full(H)


# ---------------------------------------------------------------------------
# Shallow PUCT
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    simulations: int = 24
    max_depth: int = 4
    c_puct: float = 1.5


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


@torch.no_grad()
def puct_search(
    model: TinyReasoner,
    root_state: torch.Tensor,
    cfg: SearchConfig,
) -> Tuple[int, torch.Tensor]:
    """
    Simple single-sample PUCT.
    Uses the model's value head as leaf evaluation.
    """
    assert root_state.shape[0] == 1
    device = root_state.device
    n_actions = model.num_actions

    root_logits, _ = model.policy_value(root_state)
    root_prior = torch.softmax(root_logits[0], dim=-1)
    root = SearchNode(root_state.clone(), root_prior, depth=0)
    root.init_actions(n_actions, device)

    for _ in range(cfg.simulations):
        node = root
        path: List[Tuple[SearchNode, int]] = []

        while True:
            logits, leaf_v = model.policy_value(node.state)
            prior = torch.softmax(logits[0], dim=-1)

            node.init_actions(n_actions, device)
            node.prior = prior

            if node.depth >= cfg.max_depth:
                value = float(leaf_v.item())
                break

            total_n = node.visit.sum()
            q = node.q()
            u = cfg.c_puct * prior * torch.sqrt(total_n + 1.0) / (1.0 + node.visit)
            action = int(torch.argmax(q + u).item())

            path.append((node, action))

            if action == model.HALT_ACTION:
                value = float(leaf_v.item())
                break

            if action not in node.children:
                next_state = model.transition(node.state, action)
                node.children[action] = SearchNode(
                    next_state.clone(),
                    depth=node.depth + 1,
                )
                child = node.children[action]
                _, v = model.policy_value(child.state)
                value = float(v.item())
                break

            node = node.children[action]

        for parent, action in path:
            parent.visit[action] += 1.0
            parent.value_sum[action] += value

    probs = root.visit / root.visit.sum().clamp_min(1.0)
    action = int(torch.argmax(probs).item())
    return action, probs


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


def differentiable_rollout(
    model: TinyReasoner,
    demo_x: torch.Tensor,
    demo_y: torch.Tensor,
    query_x: torch.Tensor,
    steps: int = 3,
    temperature: float = 1.0,
    hard: bool = True,
):
    """Differentiable operator rollout with near-discrete routing.

    Straight-through Gumbel-Softmax lets the forward pass execute essentially
    one operator per step while preserving gradients into the routing policy.
    This avoids the v3 failure mode where averaging all 22 nearly identical
    operators erased any pressure for an operator alphabet to specialize.
    """
    H = model.encode(demo_x, demo_y, query_x)
    A = model.core.adjacency()
    routing_stats = []

    for _ in range(steps):
        logits, _ = model.policy_value(H)
        op_logits = logits[:, :model.operator_count]

        probs = F.gumbel_softmax(
            op_logits,
            tau=temperature,
            hard=hard,
            dim=-1,
        )

        all_next = [
            model.core.apply_operator(H, k, A)
            for k in range(model.operator_count)
        ]
        stack = torch.stack(all_next, dim=1)
        H = torch.einsum("bk,bknd->bnd", probs, stack)

        soft_p = torch.softmax(op_logits / max(temperature, 1e-4), dim=-1)
        routing_stats.append(soft_p)

    return H, routing_stats


@torch.no_grad()
def deterministic_operator_rollout(
    model: TinyReasoner, demo_x: torch.Tensor, demo_y: torch.Tensor, query_x: torch.Tensor, steps: int = 3
):
    H = model.encode(demo_x, demo_y, query_x)
    A = model.core.adjacency()
    chosen_steps = []
    for _ in range(steps):
        logits, _ = model.policy_value(H)
        actions = logits[:, :model.operator_count].argmax(dim=-1)
        chosen_steps.append(actions)
        all_next = torch.stack([model.core.apply_operator(H, k, A) for k in range(model.operator_count)], dim=1)
        selector = F.one_hot(actions, num_classes=model.operator_count).to(H.dtype)
        H = torch.einsum("bk,bknd->bnd", selector, all_next)
    return H, torch.stack(chosen_steps, dim=1)


@torch.no_grad()
def apply_selected_actions(model, H, actions, adjacency=None):
    if adjacency is None:
        adjacency = model.core.adjacency()
    all_next = torch.stack([model.core.apply_operator(H, k, adjacency) for k in range(model.operator_count)], dim=1)
    selector = F.one_hot(actions, num_classes=model.operator_count).to(H.dtype)
    return torch.einsum("bk,bknd->bnd", selector, all_next)


@torch.no_grad()
def demo_guided_two_step_search(model, support_x, support_y, probe_x, probe_y, query_x):
    """Search all 22x22 programs on an independent known probe pair."""
    B = query_x.shape[0]
    K = model.operator_count
    A = model.core.adjacency()
    H_probe = model.encode(support_x, support_y, probe_x)
    first = torch.stack([model.core.apply_operator(H_probe, k, A) for k in range(K)], dim=1)
    first_flat = first.reshape(B * K, first.shape[-2], first.shape[-1])
    second = torch.stack([model.core.apply_operator(first_flat, k, A) for k in range(K)], dim=1)
    all_states = second.reshape(B, K, K, second.shape[-2], second.shape[-1])
    flat_states = all_states.reshape(B * K * K, second.shape[-2], second.shape[-1])
    pred_probe = model.decode_full(flat_states).reshape(B, K, K, -1)
    score = (pred_probe - probe_y[:, None, None, :]).pow(2).mean(dim=-1)
    best_flat = score.reshape(B, K * K).argmin(dim=-1)
    first_action = best_flat // K
    second_action = best_flat % K
    sequences = torch.stack([first_action, second_action], dim=-1)
    best_probe_mse = score.reshape(B, K * K).gather(1, best_flat[:, None]).squeeze(1)
    H_query = model.encode(support_x, support_y, query_x)
    H_query = apply_selected_actions(model, H_query, first_action, A)
    H_query = apply_selected_actions(model, H_query, second_action, A)
    return model.decode_full(H_query), sequences, best_probe_mse


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
    rollout_steps: int = 3,
) -> float:

    H, _ = differentiable_rollout(
        model,
        demo_x,
        demo_y,
        query_x,
        steps=rollout_steps,
        hard=True,
    )

    pred = model.decode(H)

    return float(
        F.mse_loss(
            pred,
            query_y,
        ).item()
    )


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
    """Evaluate task inference and both decoder paths on fresh tasks."""
    model.eval()
    demo_x, demo_y, query_x, query_y, task_ids = task_source.sample(
        batch_size, device
    )
    H0 = model.encode(demo_x, demo_y, query_x)

    pred_full = model.decode_full(H0)
    pred_pooled = model.decode_pooled(H0)
    task_logits_pooled = model.task_logits_pooled(H0)
    task_logits_full = model.task_logits_full(H0)

    full_per_sample = F.mse_loss(
        pred_full, query_y, reduction="none"
    ).mean(dim=-1)
    pooled_per_sample = F.mse_loss(
        pred_pooled, query_y, reduction="none"
    ).mean(dim=-1)

    task_acc_pooled = (
        task_logits_pooled.argmax(dim=-1) == task_ids
    ).float().mean()
    task_acc_full = (
        task_logits_full.argmax(dim=-1) == task_ids
    ).float().mean()

    names = task_source.TRAIN_NAMES
    per_task = {}
    for task_id, name in enumerate(names):
        mask = task_ids == task_id
        if mask.any():
            per_task[name] = (
                float(full_per_sample[mask].mean().item()),
                float(pooled_per_sample[mask].mean().item()),
                float((task_logits_full[mask].argmax(dim=-1) == task_ids[mask]).float().mean().item()),
                int(mask.sum().item()),
            )

    return {
        "full_mse": float(full_per_sample.mean().item()),
        "pooled_mse": float(pooled_per_sample.mean().item()),
        "task_acc_pooled": float(task_acc_pooled.item()),
        "task_acc_full": float(task_acc_full.item()),
        "per_task": per_task,
    }


@torch.no_grad()
def _sequence_summary(sequences: torch.Tensor, top_k: int = 5):
    patterns = Counter(tuple(int(v) for v in row) for row in sequences.cpu().tolist())
    total = max(sum(patterns.values()), 1)
    return [(pattern, count, count / total) for pattern, count in patterns.most_common(top_k)]


@torch.no_grad()
def evaluate_operator_rollout(model: TinyReasoner, task_source: SyntheticTaskBatch, device, batch_size: int = 1024, rollout_steps: int = 3):
    model.eval()
    demo_x, demo_y, query_x, query_y, task_ids = task_source.sample(batch_size, device)
    H0 = model.encode(demo_x, demo_y, query_x)
    direct = model.decode_full(H0)
    H, sequences = deterministic_operator_rollout(model, demo_x, demo_y, query_x, steps=rollout_steps)
    pred = model.decode_full(H)
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
        H0 = model.encode(support_x, support_y, query_x)
        direct_pred = model.decode_full(H0)
        direct_per = F.mse_loss(direct_pred, query_y, reduction="none").mean(dim=-1)
        H_greedy, greedy_sequences = deterministic_operator_rollout(model, support_x, support_y, query_x, steps=2)
        greedy_pred = model.decode_full(H_greedy)
        greedy_per = F.mse_loss(greedy_pred, query_y, reduction="none").mean(dim=-1)
        search_pred, search_sequences, probe_mse = demo_guided_two_step_search(model, support_x, support_y, probe_x, probe_y, query_x)
        search_per = F.mse_loss(search_pred, query_y, reduction="none").mean(dim=-1)
        target_energy = query_y.pow(2).mean(dim=-1).clamp_min(1e-6)
        results[name] = {
            "primitive_sequence": primitive_sequence,
            "direct_mse": float(direct_per.mean().item()),
            "greedy_mse": float(greedy_per.mean().item()),
            "search_mse": float(search_per.mean().item()),
            "search_nmse": float((search_per / target_energy).mean().item()),
            "search_improvement": float((direct_per.mean() - search_per.mean()).item()),
            "probe_mse": float(probe_mse.mean().item()),
            "greedy_sequences": _sequence_summary(greedy_sequences),
            "search_sequences": _sequence_summary(search_sequences),
        }
    return results


def train_smoke_test(
    steps: int = 1000,
    batch_size: int = 32,
    inner_rollout_steps: int = 1,
    warmup_steps: int = 250,
    es_every: int = 0,
    diagnostic_every: int = 25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_all(0)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(device)
    model = TinyReasoner().to(torch_device)
    tasks = SyntheticTaskBatch(dim=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"device={device}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"substrate_nodes={model.core.coords.shape[0]}")
    print(f"operators={model.operator_count} + HALT")
    print(f"mode=staged warmup({warmup_steps}) -> atomic one-step gumbel operators")
    print(f"es_every={es_every}")

    for step in range(1, steps + 1):
        model.train()
        (
            demo_x,
            demo_y,
            query_x,
            query_y,
            task_ids,
        ) = tasks.sample(
            batch_size,
            device,
        )

        H0 = model.encode(
            demo_x,
            demo_y,
            query_x,
        )

        in_warmup = step <= warmup_steps
        routing_stats = []
        if in_warmup:
            H = H0
        else:
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            temperature = max(0.30, 1.0 - 0.70 * progress)
            H, routing_stats = differentiable_rollout(
                model,
                demo_x,
                demo_y,
                query_x,
                steps=inner_rollout_steps,
                temperature=temperature,
                hard=True,
            )

        # Train the full-substrate decoder as the primary path and the pooled
        # decoder as a control. This directly tests whether pooling destroys
        # information needed to apply positional transformations.
        pred = model.decode_full(H)
        pooled_pred = model.decode_pooled(H)

        recon = F.mse_loss(pred, query_y)
        pooled_recon = F.mse_loss(pooled_pred, query_y)

        # Can the initial representation identify the demonstrated rule?
        # The full-state probe tells us whether task identity exists anywhere in
        # H0; the pooled probe separately measures information lost by pooling.
        task_logits_pooled = model.task_logits_pooled(H0)
        task_logits_full = model.task_logits_full(H0)
        task_loss_pooled = F.cross_entropy(task_logits_pooled, task_ids)
        task_loss_full = F.cross_entropy(task_logits_full, task_ids)
        task_acc_pooled = (
            task_logits_pooled.argmax(dim=-1) == task_ids
        ).float().mean()
        task_acc_full = (
            task_logits_full.argmax(dim=-1) == task_ids
        ).float().mean()

        per_sample_error = F.mse_loss(
            pred,
            query_y,
            reduction="none",
        ).mean(dim=-1)

        value_target = torch.exp(
            -per_sample_error.detach()
        )

        _, value = model.policy_value(H)

        value_loss = F.mse_loss(
            value,
            value_target,
        )
        # Keep operator codes spread out to discourage collapse.
        codes = F.normalize(model.core.operator_codes, dim=-1)
        gram = codes @ codes.t()
        eye = torch.eye(model.operator_count, device=torch_device)
        diversity = ((gram - eye) ** 2).mean()

        # Encourage generated updates to remain small early on.
        A = model.core.adjacency()
        graph_reg = A.pow(2).mean()

        # After the direct warmup, explicitly train an operator alphabet rather
        # than 22 copies of one transform. Margin separation prevents blow-up.
        if in_warmup:
            op_sep_loss = torch.zeros((), device=torch_device)
            op_pair_train = torch.zeros((), device=torch_device)
            route_entropy = torch.zeros((), device=torch_device)
            route_balance = torch.zeros((), device=torch_device)
        else:
            op_sep_loss, op_pair_train = operator_separation_loss(model, H0)
            route_p = torch.stack(routing_stats, dim=0).mean(dim=0)
            route_entropy = -(route_p * torch.log(route_p + 1e-8)).sum(dim=-1).mean() / math.log(model.operator_count)
            mean_route = route_p.mean(dim=0)
            uniform = torch.full_like(mean_route, 1.0 / model.operator_count)
            route_balance = (mean_route * torch.log((mean_route + 1e-8) / uniform)).sum()

        loss = (
            recon
            + 0.25 * pooled_recon
            + 0.10 * task_loss_full
            + 0.05 * task_loss_pooled
            + 0.25 * value_loss
            + 0.01 * diversity
            + 1e-4 * graph_reg
            + (0.08 * op_sep_loss if not in_warmup else 0.0)
            + (0.015 * route_entropy if not in_warmup else 0.0)
            + (0.03 * route_balance if not in_warmup else 0.0)
        )

        mean_op_delta = None
        max_op_delta = None
        normalized_entropy = None
        operator_pair_distance = None
        if diagnostic_every > 0 and (step == 1 or step % diagnostic_every == 0):
            with torch.no_grad():
                A_diag = model.core.adjacency()
                op_deltas = []
                for k in range(model.operator_count):
                    Hk = model.core.apply_operator(H0, k, A_diag)
                    delta_k = (Hk - H0).pow(2).mean().sqrt()
                    op_deltas.append(delta_k)

                op_deltas = torch.stack(op_deltas)
                mean_op_delta = op_deltas.mean()
                max_op_delta = op_deltas.max()

                # Pairwise output distance answers the more important question:
                # are the 22 operators actually producing distinct states?
                op_states = torch.stack(
                    [model.core.apply_operator(H0, k, A_diag)
                     for k in range(model.operator_count)],
                    dim=1,
                )  # [B,K,N,D]
                pair_diff = op_states[:, :, None] - op_states[:, None, :]
                pair_dist = (pair_diff.pow(2).mean(dim=(-1, -2)) + 1e-8).sqrt()
                upper = torch.triu(
                    torch.ones(
                        model.operator_count, model.operator_count,
                        device=H0.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                operator_pair_distance = pair_dist[:, upper].mean()

                logits_diag, _ = model.policy_value(H0)
                p_diag = torch.softmax(
                    logits_diag[:, :model.operator_count],
                    dim=-1,
                )
                entropy = -(
                    p_diag * torch.log(p_diag + 1e-8)
                ).sum(dim=-1).mean()
                normalized_entropy = entropy / math.log(model.operator_count)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        es_info = None
        if es_every > 0 and step % es_every == 0:
            model.eval()
            es_info = antithetic_es_step_(
                model,
                tasks,
                device,
                pairs=4,
                sigma=0.005,
                lr=0.01,
                eval_batch_size=24,
                rollout_steps=inner_rollout_steps,
            )

        if step == 1 or step % 25 == 0:
            msg = (
                f"step={step:04d} "
                f"loss={loss.item():.5f} "
                f"recon={recon.item():.5f} "
                f"pool={pooled_recon.item():.5f} "
                f"taskFull={task_loss_full.item():.4f} "
                f"taskAccFull={task_acc_full.item():.3f} "
                f"taskPool={task_loss_pooled.item():.4f} "
                f"taskAccPool={task_acc_pooled.item():.3f} "
                f"value={value_loss.item():.5f} "
                f"div={diversity.item():.5f} "
                f"phase={'warmup' if in_warmup else 'ops'}"
            )
            if not in_warmup:
                msg += (
                    f" sep={op_sep_loss.item():.4f}"
                    f" trainPair={op_pair_train.item():.5f}"
                    f" routeH={route_entropy.item():.3f}"
                    f" routeKL={route_balance.item():.3f}"
                )
            if mean_op_delta is not None:
                msg += (
                    f" opΔ={mean_op_delta.item():.5f}"
                    f" opΔmax={max_op_delta.item():.5f}"
                    f" opPair={operator_pair_distance.item():.5f}"
                    f" policyH={normalized_entropy.item():.3f}"
                )
            if es_info:
                msg += (
                    f" | es+={es_info['es_loss_plus']:.5f}"
                    f" es-={es_info['es_loss_minus']:.5f}"
                    f" es|g|={es_info['es_grad_norm']:.3e}"
                )
            print(msg)

    # Save the trained state before diagnostics so an evaluation bug never
    # forces a full retrain.
    checkpoint_path = "yetirah_v6_posttrain.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "steps": steps,
    }, checkpoint_path)
    print(f"\nsaved checkpoint: {checkpoint_path}")

    # Held-out evaluation over many fresh synthetic tasks.
    held_out = evaluate_held_out(
        model, tasks, device, batch_size=1024
    )
    print("\nHeld-out representation diagnostics (1024 fresh tasks)")
    print(
        f"full-substrate MSE: {held_out['full_mse']:.5f} | "
        f"pooled MSE: {held_out['pooled_mse']:.5f} | "
        f"task-ID full: {held_out['task_acc_full']:.3f} | "
        f"task-ID pooled: {held_out['task_acc_pooled']:.3f}"
    )
    for name, (full_mse, pooled_mse, task_acc_full, count) in held_out["per_task"].items():
        print(
            f"  {name:8s} n={count:4d} "
            f"full={full_mse:.5f} pooled={pooled_mse:.5f} "
            f"taskFull={task_acc_full:.3f}"
        )

    op_eval = evaluate_operator_rollout(
        model, tasks, device, batch_size=1024, rollout_steps=inner_rollout_steps
    )
    print("\nHeld-out operator rollout diagnostics (1024 fresh tasks)")
    print(
        f"direct MSE: {op_eval['direct_mse']:.5f} | "
        f"operator-rollout MSE: {op_eval['rollout_mse']:.5f} | "
        f"opPair: {op_eval['op_pair']:.5f} | sepLoss: {op_eval['sep_loss']:.4f}"
    )
    print("top routed operators:", list(zip(op_eval['top_ops'], [round(x, 3) for x in op_eval['top_usage']])))
    print(f"rollout normalized MSE: {op_eval['rollout_nmse']:.5f}")
    print("primitive operator sequence patterns:")
    for name, stats in op_eval["per_task"].items():
        print(f"  {name:8s} direct={stats['direct_mse']:.5f} rollout={stats['rollout_mse']:.5f} nmse={stats['rollout_nmse']:.4f}")
        for pattern, count, frac in stats["sequences"][:3]:
            print(f"    {pattern}  n={count:3d} frac={frac:.3f}")

    comp_eval = evaluate_compositional_generalization(model, tasks, device, batch_size=128)
    print("\nZERO-SHOT compositional generalization (never trained as targets)")
    print("  greedy = OOD policy | search = independent-demo-guided exhaustive 2-step program search")
    for name, stats in comp_eval.items():
        print(
            f"  {name:16s} primitives={stats['primitive_sequence']} "
            f"direct={stats['direct_mse']:.5f} greedy={stats['greedy_mse']:.5f} "
            f"search={stats['search_mse']:.5f} nmse={stats['search_nmse']:.4f} "
            f"improve={stats['search_improvement']:+.5f} probe={stats['probe_mse']:.5f}"
        )
        print("    greedy sequences:")
        for pattern, count, frac in stats["greedy_sequences"][:3]:
            print(f"      {pattern}  n={count:3d} frac={frac:.3f}")
        print("    searched sequences:")
        for pattern, count, frac in stats["search_sequences"][:3]:
            print(f"      {pattern}  n={count:3d} frac={frac:.3f}")


    # Keep the discrete search smoke test as a diagnostic only. HALT and the
    # discrete-state value distribution are still not properly trained here.
    model.eval()
    demo_x, demo_y, query_x, query_y, _ = tasks.sample(1, device)
    H0 = model.encode(demo_x, demo_y, query_x)
    action, probs = puct_search(
        model,
        H0,
        SearchConfig(simulations=24, max_depth=4, c_puct=1.5),
    )

    print("\nPUCT smoke test (diagnostic only; HALT/discrete-state value are not fully trained)")
    print("chosen action:", "HALT" if action == model.HALT_ACTION else action)
    print("top policy visits:", torch.topk(probs, k=5).indices.tolist())
    print("top visit probs:", torch.topk(probs, k=5).values.tolist())

    return model


if __name__ == "__main__":
    train_smoke_test()
