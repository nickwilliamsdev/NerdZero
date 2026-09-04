"""
yetirah_v0.py

Version 0 of a geometry-generated operator model inspired by the architecture
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
        nn.init.normal_(out.weight, std=0.02)
        nn.init.zeros_(out.bias)

        self.net = nn.Sequential(
            nn.Linear(coord_dim + code_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, node_dim * 2 + 1),
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
        gamma = 1.0 + 0.10 * torch.tanh(out[:, :d])
        beta = 0.10 * torch.tanh(out[:, d:2*d])
        alpha = 0.10 * torch.sigmoid(out[:, 2*d:])
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

        coords = make_hypercube_vertices(coord_dim)
        self.register_buffer("coords", coords)

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

        # Write an input vector into the geometric workspace.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_nodes * node_dim),
            nn.Tanh(),
        )

        # State summary.
        self.pool_query = nn.Parameter(torch.randn(node_dim) / math.sqrt(node_dim))

        # Compatibility-based policy rather than a fixed 22-way linear head.
        self.state_to_policy = nn.Linear(node_dim, operator_code_dim)
        self.halt_head = nn.Linear(node_dim, 1)

        self.value_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 1),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        n = self.core.coords.shape[0]
        return self.encoder(x).view(b, n, self.node_dim)

    def pool(self, H: torch.Tensor) -> torch.Tensor:
        # Learned query attention over the 32 substrate locations.
        score = torch.einsum("bnd,d->bn", H, self.pool_query)
        attn = torch.softmax(score / math.sqrt(self.node_dim), dim=-1)
        return torch.einsum("bn,bnd->bd", attn, H)

    def policy_value(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.pool(H)

        q = self.state_to_policy(z)  # [B, code_dim]
        op_logits = q @ self.core.operator_codes.t() / math.sqrt(q.shape[-1])
        halt_logit = self.halt_head(z)
        logits = torch.cat([op_logits, halt_logit], dim=-1)

        value = self.value_head(z).squeeze(-1)
        return logits, value

    def transition(self, H: torch.Tensor, action: int) -> torch.Tensor:
        if action == self.HALT_ACTION:
            return H
        return self.core.apply_operator(H, action)

    def decode(self, H: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.pool(H))


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
    """
    Generates a simple family of vector transformations.

    These are NOT meant as a benchmark. They only test that the machinery can
    learn reusable transformations before we connect it to ARC.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def sample(self, batch_size: int, device):
        x = torch.randn(batch_size, self.dim, device=device)

        task_ids = torch.randint(0, 4, (batch_size,), device=device)
        y = torch.empty_like(x)

        for i, t in enumerate(task_ids.tolist()):
            if t == 0:
                y[i] = torch.roll(x[i], shifts=1, dims=0)
            elif t == 1:
                y[i] = -x[i]
            elif t == 2:
                y[i] = x[i].flip(0)
            else:
                y[i] = 0.5 * x[i] + 0.5 * torch.roll(x[i], shifts=2, dims=0)

        return x, y, task_ids


def differentiable_rollout(
    model: TinyReasoner,
    x: torch.Tensor,
    steps: int = 3,
    temperature: float = 1.0,
):
    """
    Soft operator mixture for differentiable pretraining.

    MCTS remains discrete at inference/search time. During initial training,
    this soft mixture gives the operator policy and operator generator a dense
    gradient signal.
    """
    H = model.encode(x)
    A = model.core.adjacency()

    for _ in range(steps):
        logits, _ = model.policy_value(H)
        probs = torch.softmax(logits[:, :model.operator_count] / temperature, dim=-1)

        all_next = []
        for k in range(model.operator_count):
            all_next.append(model.core.apply_operator(H, k, A))
        stack = torch.stack(all_next, dim=1)  # [B,K,N,D]
        H = torch.einsum("bk,bknd->bnd", probs, stack)

    return H


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
def evaluate_loss(
    model: TinyReasoner,
    task_source: SyntheticTaskBatch,
    batch_size: int,
    device,
    rollout_steps: int = 3,
) -> float:
    x, y, _ = task_source.sample(batch_size, device)
    H = differentiable_rollout(model, x, steps=rollout_steps)
    pred = model.decode(H)
    return float(F.mse_loss(pred, y).item())


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
    noises = []
    losses_plus = []
    losses_minus = []

    # Use one fixed evaluation batch worth of RNG state implicitly close enough
    # for this V0; later, use common random numbers exactly.
    for _ in range(pairs):
        eps = torch.randn_like(base)
        noises.append(eps)

        set_genotype_vector_(model, base + sigma * eps)
        lp = evaluate_loss(model, task_source, eval_batch_size, device, rollout_steps)

        set_genotype_vector_(model, base - sigma * eps)
        lm = evaluate_loss(model, task_source, eval_batch_size, device, rollout_steps)

        losses_plus.append(lp)
        losses_minus.append(lm)

    set_genotype_vector_(model, base)

    # Gradient estimate for minimizing loss.
    g = torch.zeros_like(base)
    for eps, lp, lm in zip(noises, losses_plus, losses_minus):
        g += (lp - lm) * eps
    g /= (2.0 * pairs * sigma)

    # Normalize to make the first experiments much less sensitive to scale.
    g = g / (g.norm() + 1e-8)
    set_genotype_vector_(model, base - lr * g)

    return {
        "es_loss_plus": sum(losses_plus) / len(losses_plus),
        "es_loss_minus": sum(losses_minus) / len(losses_minus),
        "es_grad_norm": float(g.norm().item()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_smoke_test(
    steps: int = 500,
    batch_size: int = 32,
    inner_rollout_steps: int = 3,
    es_every: int = 50,
    device: Optional[str] = None,
):
    seed_all(0)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    model = TinyReasoner().to(device)
    tasks = SyntheticTaskBatch(dim=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"device={device}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"substrate_nodes={model.core.coords.shape[0]}")
    print(f"operators={model.operator_count} + HALT")

    for step in range(1, steps + 1):
        model.train()
        x, y, _ = tasks.sample(batch_size, device)

        H = differentiable_rollout(
            model,
            x,
            steps=inner_rollout_steps,
            temperature=max(0.4, 1.0 - step / max(steps, 1)),
        )
        pred = model.decode(H)

        recon = F.mse_loss(pred, y)

        # Keep operator codes spread out to discourage collapse.
        codes = F.normalize(model.core.operator_codes, dim=-1)
        gram = codes @ codes.t()
        eye = torch.eye(model.operator_count, device=device)
        diversity = ((gram - eye) ** 2).mean()

        # Encourage generated updates to remain small early on.
        A = model.core.adjacency()
        graph_reg = A.pow(2).mean()

        loss = recon + 0.01 * diversity + 1e-4 * graph_reg

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
                f"div={diversity.item():.5f}"
            )
            if es_info:
                msg += (
                    f" | es+={es_info['es_loss_plus']:.5f}"
                    f" es-={es_info['es_loss_minus']:.5f}"
                )
            print(msg)

    # Verify discrete search runs.
    model.eval()
    x, y, _ = tasks.sample(1, device)
    H0 = model.encode(x)
    action, probs = puct_search(
        model,
        H0,
        SearchConfig(simulations=24, max_depth=4, c_puct=1.5),
    )

    print("\nPUCT smoke test")
    print("chosen action:", "HALT" if action == model.HALT_ACTION else action)
    print("top policy visits:", torch.topk(probs, k=5).indices.tolist())
    print("top visit probs:", torch.topk(probs, k=5).values.tolist())

    return model


if __name__ == "__main__":
    train_smoke_test()
