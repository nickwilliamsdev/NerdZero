"""
train_cppn_arc.py

CPPN -> phenotype (few-shot ARC solver) -> weight-delta meta-training.

Outer loop (NEAT, structural evolution) evolves a *CPPN genome*.  Each genome's
CPPN is a HyperNEAT-style weight generator: queried at 2-D coordinate points it
emits the flat parameter vector of a small phenotype network (a tiny conv net by
default; the architecture is swappable for deltanet/srwm/attention later).

Per-genome fitness pipeline:

    1. Encode:  W_init = cppn(coords)                # phenotype params, grad -> cppn
    2. Adapt:   SGD the phenotype on one ARC task so it predicts the query output
                from the demo (input,output) pairs + the query input grid.
    3. Delta:   L = || W_after - W_init ||^2          # "how much adaptation was needed"
    4. Meta:    backprop L through W_init into the CPPN params and take a CPPN step.

    Repeat 1-4 for `cppn_iters` on the shared SGD task.  A CPPN that produces
    initialisations which need little SGD adaptation (small delta) is rewarded.

    5. Eval:    on a shared random sample of `n_eval` tasks, re-encode the now
                trained CPPN, adapt a little, and score prediction accuracy.
                Mean accuracy is the genome's fitness.

The SGD task and the eval task sample are identical for every member of the
population within a generation (fair comparison).

This is intentionally a compact, end-to-end scaffold to validate the idea; grid
size, phenotype architecture, and population size are all kept small so a full
generation runs in seconds-to-minutes on a single GPU.

Usage:
    python train_cppn_arc.py --smoke            # tiny 1-2 gen smoke test
    python train_cppn_arc.py --generations 20   # real (small) run
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import textwrap
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: make the local PyTorch-NEAT checkout importable (it is a sibling
# of this project directory).
# ---------------------------------------------------------------------------
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTORCH_NEAT_DIR = os.path.join(_PROJECT_DIR, "PyTorch-NEAT")
if _PYTORCH_NEAT_DIR not in sys.path:
    sys.path.insert(0, _PYTORCH_NEAT_DIR)

import neat  # type: ignore
from pytorch_neat.cppn import create_torch_cppn, TorchCPPN  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ARC / grid
    grid_size: int = 15          # pads/normalises all grids to S x S
    max_demos: int = 3           # max (input,output) demo pairs per episode
    in_channels: int = 1         # normalised single channel (value/9)
    out_channels: int = 10       # one-hot of the 10 ARC colours

    # Phenotype (the network the CPPN encodes) -- keep it small
    phenotype: str = "attn"       # "conv" | "attn" | "deltanet" | "pure_attn"
    channels: int = 8            # hidden channels of the conv phenotype
    attn_dim: int = 48           # sequence-model hidden dim
    attn_heads: int = 4          # attention heads
    attn_layers: int = 2         # attention layers

    # Meta-training (weight-delta) hyperparameters
    cppn_iters: int = 6          # inner CPPN update iterations on the SGD task
    sgd_steps: int = 12          # phenotype SGD steps per inner iteration
    sgd_lr: float = 5e-2         # phenotype SGD learning rate
    cppn_lr: float = 5e-3        # CPPN (meta) learning rate

    # Evaluation / fitness
    n_eval: int = 6              # number of eval tasks (5-10 as requested)
    eval_sgd_steps: int = 6      # little adaptation before scoring on eval tasks

    # NEAT outer loop
    pop_size: int = 12           # small population (intensive fitness fn)
    cppn_num_hidden: int = 14    # small CPPN
    cppn_num_outputs: int = 1    # CPPN output channels; 4 for pure_attn (k, q, v, rest)
    generations: int = 20

    # Dynamic substrate: condition the CPPN query substrate on the input grid
    dynamic_substrate: bool = True   # False -> fixed abstract substrate (original)
    substrate_dim: int = 5           # CPPN coordinate dimensionality (5D: layer-aware)
    substrate_radius: float = 0.25   # half-size of the per-task substrate shift window
    coord_schema: str = "auto"     # "auto" (>=4 layer types -> layer, else flat), "layer" (5D), "flat"
    gated_branches: int = 3         # 'gated' phenotype: 3 = dn/attn/conv (original); 4 adds a global "whole-grid" linear branch
    save_name: str = "cppn_arc_winner.pkl"   # where the winning genome is pickled
    fixed_eval: bool = False                 # sample eval tasks once, reuse every generation
    meta_loss: str = "delta"                 # "delta" (1st-order weight-delta) or "maml" (2nd-order FOMAML)
    meta_loss_weight: str = "mean"           # "mean" (per-coordinate mean) or "per_param" (equal weight per parameter tensor)
    gate_meta: str = "delta"                 # meta-signal for tiny selector tensors (gates): "delta" or "task" (post-adaptation task-loss gradient)
    gate_task_scale: float = 0.5             # with gate_meta="task", blend factor s in (1-s)*delta + s*task for the gate. 1.0=full task signal (over-commits), 0.0=delta; damp to ~0.25-0.5 to keep branch choice from overfitting

    seed: int = 0
    device: str = "auto"
    log_every: int = 1

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return DEVICE
        return torch.device(self.device)


def neat_config_text(num_inputs: int, num_outputs: int, num_hidden: int,
                     pop_size: int, seed: int = 0) -> str:
    """A complete NEAT (neat-python 2.x) config for the CPPN genome.

    The CPPN is a single-output network: 2 coordinate inputs (px, py) -> 1
    output (the phenotype parameter value at that coordinate).
    """
    return textwrap.dedent(f"""
    [NEAT]
    fitness_criterion = max
    fitness_threshold = 1e9
    pop_size = {pop_size}
    reset_on_extinction = False
    no_fitness_termination = True
    seed = {seed}

    [DefaultGenome]
    num_inputs = {num_inputs}
    num_outputs = {num_outputs}
    num_hidden = {num_hidden}
    feed_forward = True
    initial_connection = full_direct
    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient = 0.2
    conn_add_prob = 0.40
    conn_delete_prob = 0.10
    node_add_prob = 0.30
    node_delete_prob = 0.05

    activation_default = tanh
    activation_options = tanh sigmoid identity
    activation_mutate_rate = 0.10
    aggregation_default = sum
    aggregation_options = sum
    aggregation_mutate_rate = 0.0

    bias_init_mean = 0.0
    bias_init_stdev = 0.5
    bias_replace_rate = 0.1
    bias_mutate_rate = 0.7
    bias_mutate_power = 0.5
    bias_max_value = 5.0
    bias_min_value = -5.0

    response_init_mean = 1.0
    response_init_stdev = 0.0
    response_replace_rate = 0.0
    response_mutate_rate = 0.0
    response_mutate_power = 0.0
    response_max_value = 5.0
    response_min_value = -5.0

    time_constant_init_mean = 1.0
    time_constant_init_stdev = 0.0
    time_constant_replace_rate = 0.0
    time_constant_mutate_rate = 0.0
    time_constant_mutate_power = 0.0
    time_constant_max_value = 5.0
    time_constant_min_value = -5.0

    weight_init_mean = 0.0
    weight_init_stdev = 0.5
    weight_replace_rate = 0.1
    weight_mutate_rate = 0.8
    weight_mutate_power = 0.5
    weight_max_value = 5.0
    weight_min_value = -5.0

    enabled_default = True
    enabled_mutate_rate = 0.01

    [DefaultSpeciesSet]
    compatibility_threshold = 4.5

    [DefaultStagnation]
    species_fitness_func = max
    max_stagnation = 20
    species_elitism = 2

    [DefaultReproduction]
    elitism = 2
    survival_threshold = 0.25
    """).strip() + "\n"


def build_neat_config(cfg: Config) -> neat.Config:
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(neat_config_text(cfg.substrate_dim, cfg.cppn_num_outputs,
                                 cfg.cppn_num_hidden, cfg.pop_size, cfg.seed))
        name = f.name
    return neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, name,
    )


# ---------------------------------------------------------------------------
# Phenotype: the small network the CPPN encodes.
#
# It sees the demo (input,output) grids and the query input grid, and predicts
# the query output grid.  A conv backbone (shared weights => few parameters) is
# used by default; swap in a DeltaNet / SRWM / attention module here later --
# the only contract is a flat `parameters()` order that matches `ParamCoords`.
# ---------------------------------------------------------------------------
class ArcPhenotype(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.stem = nn.Conv2d(cfg.in_channels, cfg.channels, 3, padding=1)
        self.mid = nn.Conv2d(cfg.channels, cfg.channels, 3, padding=1)
        self.combine = nn.Conv2d(3 * cfg.channels, cfg.out_channels, 1)

    def _encode_grid(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, S, S]  ->  [B, k, S, S]
        return self.mid(self.stem(x))

    def forward(self, ep: Episode) -> torch.Tensor:
        # demo_in / demo_out: [D, C, S, S];  query_in: [C, S, S] (single query grid)
        # Returns: [1, out_channels, S, S]  (the predicted query output grid)
        demo_in, demo_out, query_in = ep.demo_in, ep.demo_out, ep.query_in
        d = demo_in.shape[0]
        qi = query_in.unsqueeze(0).expand(d, *query_in.shape)   # [D, C, S, S]
        di = self._encode_grid(demo_in)
        do = self._encode_grid(demo_out)
        qe = self._encode_grid(qi)
        x = torch.cat([di, do, qe], dim=1)          # [D, 3k, S, S]
        logits = self.combine(x)                    # [D, out_channels, S, S]
        return logits.mean(dim=0, keepdim=True)     # [1, out_channels, S, S]


def build_phenotype(cfg: Config) -> nn.Module:
    """Instantiate the phenotype of the requested architecture on `device`."""
    if cfg.phenotype == "conv":
        return ArcPhenotype(cfg).to(cfg.resolve_device())
    if cfg.phenotype == "attn":
        return ArcAttnPhenotype(cfg).to(cfg.resolve_device())
    if cfg.phenotype == "deltanet":
        return ArcDeltaNetPhenotype(cfg).to(cfg.resolve_device())
    if cfg.phenotype == "mixed":
        return ArcMixedPhenotype(cfg).to(cfg.resolve_device())
    if cfg.phenotype == "gated":
        return ArcGatedPhenotype(cfg).to(cfg.resolve_device())
    if cfg.phenotype == "pure_attn":
        return ArcPureAttnPhenotype(cfg).to(cfg.resolve_device())
    raise ValueError(f"unknown phenotype: {cfg.phenotype!r}")


# ---------------------------------------------------------------------------
# Sequence-based phenotypes (attention / DeltaNet).
#
# Grids are flattened into token sequences.  Each demo pair contributes its
# input cells (role 0) then output cells (role 1); the query input cells
# (role 2) come last.  The model reads the sequence and predicts, for each
# query cell, the 10-colour output distribution.  Every cell token is:
#
#     color_embed[color] + pos_row_embed[i] + pos_col_embed[j] + role_embed[r]
#
# where (i, j) is the cell's position within its (S x S) grid.
#
# Both sequence phenotypes accept an `Episode` and return [1, out, S, S]
# logits, exactly like the conv phenotype, so sgd_adapt / episode_accuracy /
# load_param_vector are uniform across all three architectures.  Being plain
# nn.Modules, the CPPN encodes their flat `parameters()` identically.
# ---------------------------------------------------------------------------
@dataclass
class SeqEpisode:
    """Token-level view of an Episode for the sequence phenotypes."""
    tokens: torch.Tensor        # [L] long  (colour ids, 0..9)
    row: torch.Tensor           # [L] long  (position within grid)
    col: torch.Tensor           # [L] long
    role: torch.Tensor          # [L] long  (0=demo-in, 1=demo-out, 2=query-in)
    seq_idx: torch.Tensor       # [L] long  (index within its own grid, 0..hw-1)
    query_start: int            # position of first query token
    n_query: int                # S*S
    target: torch.Tensor        # [S*S] long (query output colours)
    grid_size: int


def episode_to_sequence(ep: Episode, device) -> SeqEpisode:
    """Flatten an Episode into a token sequence, TRUNCATING each demo grid to
    its true (unpadded) size so the sequence models only process real cells.

    Order: [demo0_in][demo0_out][demo1_in][demo1_out]...[query_in].  Padded demo
    slots (shape (0,0)) are skipped.  The query-in grid is kept at full S x S so
    the returned prediction is a uniform [1, out, S, S] map.
    """
    s = ep.demo_in.shape[-1]                    # == grid_size
    n_demo = ep.demo_in.shape[0]
    tokens, rows, cols, roles, seqs = [], [], [], [], []

    def append(grid, shape, role):
        h, w = shape
        if h <= 0 or w <= 0:
            return
        col_idx = (grid[:h, :w] * 9.0).round().long().reshape(-1)
        r, c = torch.meshgrid(
            torch.arange(h, dtype=torch.long),
            torch.arange(w, dtype=torch.long), indexing="ij")
        tokens.append(col_idx)
        rows.append(r.reshape(-1))
        cols.append(c.reshape(-1))
        roles.append(torch.full((h * w,), role, dtype=torch.long))
        seqs.append(torch.arange(h * w, dtype=torch.long))   # index within this grid

    for d in range(n_demo):
        append(ep.demo_in[d].squeeze(0), ep.demo_in_shape[d], 0)
        append(ep.demo_out[d].squeeze(0), ep.demo_out_shape[d], 1)
    # query-in stays full S x S so the prediction is a uniform S x S map
    append(ep.query_in.squeeze(0), (s, s), 2)

    query_start = sum(t.numel() for t in tokens[:-1])
    target = (ep.target.squeeze(0) * 9.0).round().long().reshape(-1)

    return SeqEpisode(
        tokens=torch.cat(tokens, 0).to(device),
        row=torch.cat(rows, 0).to(device),
        col=torch.cat(cols, 0).to(device),
        role=torch.cat(roles, 0).to(device),
        seq_idx=torch.cat(seqs, 0).to(device),
        query_start=query_start,
        n_query=s * s,
        target=target.to(device),
        grid_size=s,
    )


class CellEmbedder(nn.Module):
    """color + 2-D position + role (+ optional sequence position) -> hidden token.

    The per-grid sequence position (normalised to [0,1]) is added through a small
    linear layer so sequence models can attend by relative cell order.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.attn_dim
        self.color = nn.Embedding(cfg.out_channels, d)
        self.pos_row = nn.Embedding(cfg.grid_size, d)
        self.pos_col = nn.Embedding(cfg.grid_size, d)
        self.role = nn.Embedding(3, d)
        self.seq = nn.Linear(1, d)

    def forward(self, tokens, row, col, role, seq_frac=None) -> torch.Tensor:
        x = (
            self.color(tokens) + self.pos_row(row) +
            self.pos_col(col) + self.role(role)
        )
        if seq_frac is not None:
            x = x + self.seq(seq_frac.to(x.dtype).unsqueeze(-1))
        return x


class ArcAttnPhenotype(nn.Module):
    """Few-shot ARC solver: self-attention over the cell-token sequence.

    Reads demo (in/out) and query-in cells, predicts query output colours.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d, H, L = cfg.attn_dim, cfg.attn_heads, cfg.attn_layers
        self.embed = CellEmbedder(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=H, dim_feedforward=4 * d,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=L)
        self.head = nn.Linear(d, cfg.out_channels)
        self.out_channels = cfg.out_channels
        self.grid_size = cfg.grid_size

    def forward(self, ep: Episode) -> torch.Tensor:
        seq = episode_to_sequence(ep, self.head.weight.device)
        s = seq.grid_size
        seq_frac = (seq.seq_idx.float() / max(1, s * s - 1))
        x = self.embed(seq.tokens, seq.row, seq.col, seq.role,
                       seq_frac).unsqueeze(0)      # [1,L,d]
        h = self.encoder(x)                                   # [1,L,d]
        q = h[0, seq.query_start:seq.query_start + seq.n_query]  # [Q,d]
        logits = self.head(q)                                 # [Q, out]
        return logits.reshape(1, self.out_channels, s, s)


class PureAttnBlock(nn.Module):
    """One plain multi-head self-attention block with SEPARATE key/query/value
    projections (q, k, v each get their own nn.Linear instead of one fused
    in_proj) -- so a 3-output CPPN can encode the three projection weight
    matrices independently, one output channel per K/Q/V.

    The FFN part and head biases still come from CPPN output channel 3
    ("rest"), so exactly three CPPN outputs carry the k/q/v weight matrices.
    """
    def __init__(self, d: int, H: int):
        super().__init__()
        assert d % H == 0
        self.d = d
        self.H = H
        self.head_dim = d // H
        self.k_proj = nn.Linear(d, d, bias=False)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, L, d]
        b, l, d = x.shape
        H, hd = self.H, self.head_dim
        q = self.q_proj(self.ln1(x)).view(b, l, H, hd).transpose(1, 2)
        k = self.k_proj(self.ln1(x)).view(b, l, H, hd).transpose(1, 2)
        v = self.v_proj(self.ln1(x)).view(b, l, H, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, l, d)
        x = x + self.out_proj(a)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class ArcPureAttnPhenotype(nn.Module):
    """Few-shot ARC solver: pure attention only (no conv, no DeltaNet).

    A stack of plain multi-head attention layers over the cell-token sequence
    plus a colour head.  Every layer exposes separate k_proj / q_proj /
    v_proj matrices; with `cppn_num_outputs=4` the CPPN emits a 4-channel
    output per substrate point -- channel 0 = key weights, 1 = query weights,
    2 = value weights, 3 = everything else (embeddings, FFN, norms, head,
    biases) -- so the HyperNEAT generator specialises its response surface
    for each of the three attention projections independently of the rest of
    the network.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d, H, L = cfg.attn_dim, cfg.attn_heads, cfg.attn_layers
        self.embed = CellEmbedder(cfg)
        self.blocks = nn.ModuleList([PureAttnBlock(d, H) for _ in range(L)])
        self.head = nn.Linear(d, cfg.out_channels)
        self.out_channels = cfg.out_channels
        self.grid_size = cfg.grid_size

    def forward(self, ep: Episode) -> torch.Tensor:
        seq = episode_to_sequence(ep, self.head.weight.device)
        s = seq.grid_size
        seq_frac = (seq.seq_idx.float() / max(1, s * s - 1))
        x = self.embed(seq.tokens, seq.row, seq.col, seq.role,
                       seq_frac).unsqueeze(0)        # [1,L,d]
        for blk in self.blocks:
            x = blk(x)
        q = x[0, seq.query_start:seq.query_start + seq.n_query]   # [Q,d]
        logits = self.head(q)                                    # [Q, out]
        return logits.reshape(1, self.out_channels, s, s)


class DeltaNetBlock(nn.Module):
    """A reusable DeltaNet layer that maps an embedded token sequence to
    per-position readouts.

    Takes ``x`` of shape [B, L, d] (embedded tokens) and returns [B, L, d]
    hidden states.  Each cell reads the fast-weight memory and writes an
    error-corrected key/value update:

        S_{t+1} = S_t P_t + A_t,   P_t = I - b_t k_t k_t^T,  A_t = b_t v_t k_t^T

    The affine recurrence is computed with a divide-and-conquer associative
    scan (numerically exact vs. the sequential loop).
    """
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.beta_proj = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        q = self.q_proj(x)
        # Unit-norm keys: P_t = I - b_t k_t k_t^T then has eigenvalues
        # {1 - b_t, 1} in (0, 1), so the fast-weight memory S stays bounded
        # over long sequences.  Unnormalised keys make ||k||^2 > 1 and S
        # grows exponentially (-> inf/NaN after a few hundred tokens).
        k = F.normalize(F.elu(self.k_proj(x)) + 1.0, dim=-1, eps=1e-6)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))                 # [B, L, 1]
        bbeta = beta[..., None]                                 # [B, L, 1, 1]

        eye = torch.eye(d, device=x.device, dtype=x.dtype)      # [d,d]
        kk = k[:, :, None, :] * k[:, :, :, None]               # [B,L,d,d]  (k_t k_t^T)
        vk = v[:, :, :, None] * k[:, :, None, :]               # [B,L,d,d]  (v_t k_t^T)
        P = eye[None] - bbeta * kk                             # [B,L,d,d]
        A = bbeta * vk                                         # [B,L,d,d]

        # Inclusive prefix scan: B-part at position t = memory S_{t+1}.
        _, S_after = _deltanet_prefix_scan(P, A)                 # [B,L,d,d]
        return torch.einsum("btdk,btk->btd", S_after, q)         # [B,L,d]


class ArcDeltaNetPhenotype(nn.Module):
    """Few-shot ARC solver: a single DeltaNet block over the cell-token sequence.

    Each cell reads the fast-weight memory and writes an error-corrected
    key/value update.  Query cells are read out after the recurrence.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.attn_dim
        self.embed = CellEmbedder(cfg)
        self.block = DeltaNetBlock(d)
        self.head = nn.Linear(d, cfg.out_channels)
        self.out_channels = cfg.out_channels
        self.grid_size = cfg.grid_size

    def forward(self, ep: Episode) -> torch.Tensor:
        seq = episode_to_sequence(ep, self.head.weight.device)
        s = seq.grid_size
        seq_frac = (seq.seq_idx.float() / max(1, s * s - 1))
        x = self.embed(seq.tokens, seq.row, seq.col, seq.role,
                       seq_frac).unsqueeze(0)                   # [1,L,d]
        H = self.block(x)                                       # [1,L,d]
        logits = self.head(H[0, seq.query_start:seq.query_start + seq.n_query])  # [Q,out]
        return logits.reshape(1, self.out_channels, s, s)


class ArcGatedPhenotype(nn.Module):
    """Few-shot ARC solver where the CPPN selects the active architecture.

    A pool of three distinct layer types runs in parallel, each producing its
    own full [1, out, S, S] logit map:
      1. DeltaNet fast-weight memory  (DeltaNetBlock + head)  -- contextual memory
      2. Self-attention                (TransformerEncoder + head) -- global reasoning
      3. Conv stem                     (Conv2d layers + 1x1 head) -- spatial local ops

    The final logits are a *gated* mixture:  logits = sum_i softmax(g)_i * logits_i
    where the gate vector g (3 values) is part of the encoded parameter vector.
    Because the CPPN substrate is task-shifted, the SAME genome can emit
    different gate values per task family -- e.g. gate ~ [1,0,0] for
    memory-style tasks, [0,0,1] for spatial tiling tasks.  The gate is also an
    ordinary phenotype parameter, so in-context SGD refines it per episode.

    This is the differentiable version of "the CPPN generates a different
    architecture per task": which layer type dominates is learned by the outer
    loop, not fixed.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d, H, L = cfg.attn_dim, cfg.attn_heads, cfg.attn_layers
        # shared input embedding for the sequence branches
        self.embed = CellEmbedder(cfg)
        # branch 1: DeltaNet
        self.deltanet = DeltaNetBlock(d)
        self.head_dn = nn.Linear(d, cfg.out_channels)
        # branch 2: attention
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=H, dim_feedforward=4 * d,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=L)
        self.head_attn = nn.Linear(d, cfg.out_channels)
        # branch 3: conv (operates on the grid images directly)
        self.conv_stem = nn.Conv2d(cfg.in_channels, cfg.channels, 3, padding=1)
        self.conv_mid = nn.Conv2d(cfg.channels, cfg.channels, 3, padding=1)
        self.conv_combine = nn.Conv2d(3 * cfg.channels, cfg.out_channels, 1)
        # branch 4 (optional, v2): per-cell demo-lookup.  The v1 branch (mean-
        # pool the whole grid -> one 32-d feature -> shared per-cell head)
        # scored LAST on unseen sets: a single global number per task is too
        # weak to carry real predictions.  v2 keeps the per-cell structure:
        # for each grid cell it gathers [query-colour, mean demo-in colour at
        # that cell, mean demo-out colour there, demo in->out change there,
        # row/S, col/S] and maps it through a small MLP to a 10-way colour
        # prediction.  This targets the dominant ARC rule family — "the demo
        # shows input cell -> output cell at the SAME position" (colour maps,
        # fills, replacements) — with a ~0.5k-param head the CPPN can encode
        # easily, and the conv/sequence branches cover the rest.
        self.lookup_fc1 = None
        self.lookup_fc2 = None
        if cfg.gated_branches >= 4:
            S = cfg.grid_size
            self.lookup_fc1 = nn.Linear(4 + 2, 24)
            self.lookup_fc2 = nn.Linear(24, cfg.out_channels)
            # row-major cell order: row value held for S cells (repeat_interleave),
            # column value cycles through the full range each row (repeat)
            rows = torch.linspace(0.0, 1.0, S).repeat_interleave(S)
            cols = torch.linspace(0.0, 1.0, S).repeat(S)
            self.register_buffer("_lookup_pos",
                                 torch.stack([rows, cols], dim=1))
        # per-architecture gate (encoded by the CPPN like any other weight)
        self.gate = nn.Parameter(torch.zeros(cfg.gated_branches))
        self.out_channels = cfg.out_channels
        self.grid_size = cfg.grid_size

    def _seq_logits(self, ep: Episode) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = episode_to_sequence(ep, self.head_dn.weight.device)
        s = seq.grid_size
        seq_frac = (seq.seq_idx.float() / max(1, s * s - 1))
        x = self.embed(seq.tokens, seq.row, seq.col, seq.role,
                       seq_frac).unsqueeze(0)                   # [1,L,d]
        qstart = seq.query_start
        h_dn = self.head_dn(self.deltanet(x)[0, qstart:qstart + seq.n_query])
        h_attn = self.head_attn(self.encoder(x)[0, qstart:qstart + seq.n_query])
        return (h_dn.reshape(1, self.out_channels, s, s),
                h_attn.reshape(1, self.out_channels, s, s))

    def _conv_logits(self, ep: Episode) -> torch.Tensor:
        demo_in, demo_out, query_in = ep.demo_in, ep.demo_out, ep.query_in
        d = demo_in.shape[0]
        qi = query_in.unsqueeze(0).expand(d, *query_in.shape)   # [D,C,S,S]
        def enc(x):
            return self.conv_mid(self.conv_stem(x))
        x = torch.cat([enc(demo_in), enc(demo_out), enc(qi)], dim=1)   # [D,3k,S,S]
        return self.conv_combine(x).mean(dim=0, keepdim=True)          # [1,out,S,S]

    def _masked_demo_mean(self, ep: Episode, which: str) -> torch.Tensor:
        """Mean of the demo grids, excluding padded (0,0) demo slots -> [1,C,S,S]."""
        t = ep.demo_in if which == "in" else ep.demo_out        # [D,C,S,S] (or [C,S,S])
        shapes = ep.demo_in_shape if which == "in" else ep.demo_out_shape
        if t.dim() == 3:                                        # single demo
            keep = 1.0 if (shapes and shapes[0] != (0, 0)) else 0.0
            return t.unsqueeze(0).unsqueeze(0) * keep           # [1,1,S,S]
        keep = torch.tensor([(h, w) != (0, 0) for (h, w) in shapes],
                           device=t.device, dtype=t.dtype).view(-1, 1, 1, 1)
        n = keep.sum().clamp(min=1.0)
        return (t * keep).sum(dim=0, keepdim=True) / n          # [1,C,S,S]

    def _lookup_logits(self, ep: Episode) -> torch.Tensor:
        assert self.lookup_fc1 is not None and self.lookup_fc2 is not None
        lookup_pos = self._lookup_pos
        assert isinstance(lookup_pos, torch.Tensor)
        S = self.grid_size
        din = self._masked_demo_mean(ep, "in")
        dout = self._masked_demo_mean(ep, "out")
        q = ep.query_in.unsqueeze(0)                           # [1,C,S,S]
        feats = torch.cat([q, din, dout, dout - din], dim=1)   # [1,4,S,S]
        x = feats.permute(0, 2, 3, 1).reshape(1, S * S, 4)     # [1,S*S,4]
        pos = lookup_pos.unsqueeze(0).expand(1, S * S, -1)
        x = torch.cat([x, pos], dim=2)                         # [1,S*S,6]
        h = torch.relu(self.lookup_fc1(x))                     # [1,S*S,24]
        logits = self.lookup_fc2(h)                            # [1,S*S,out]
        return logits.reshape(1, S, S, self.out_channels).permute(0, 3, 1, 2)

    def forward(self, ep: Episode) -> torch.Tensor:
        logit_dn, logit_attn = self._seq_logits(ep)
        logit_conv = self._conv_logits(ep)
        g = torch.softmax(self.gate, dim=0)
        out = g[0] * logit_dn + g[1] * logit_attn + g[2] * logit_conv
        if self.lookup_fc2 is not None:
            out = out + g[3] * self._lookup_logits(ep)
        return out


class ArcMixedPhenotype(nn.Module):
    """Few-shot ARC solver mixing multiple layer types.

    Architecture (three distinct layer types):
      1. DeltaNet fast-weight memory  (DeltaNetBlock)  -- local/contextual memory
      2. Self-attention                (TransformerEncoder) -- global reasoning
      3. Linear readout head           (nn.Linear)      -- colour prediction

    The same CPPN substrate encodes the flat weights of all three layers; the
    layer-aware coordinates (see ``build_param_coords``) let the CPPN specialise
    each layer's region.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d, H, L = cfg.attn_dim, cfg.attn_heads, cfg.attn_layers
        self.embed = CellEmbedder(cfg)
        self.deltanet = DeltaNetBlock(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=H, dim_feedforward=4 * d,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=L)
        self.head = nn.Linear(d, cfg.out_channels)
        self.out_channels = cfg.out_channels
        self.grid_size = cfg.grid_size

    def forward(self, ep: Episode) -> torch.Tensor:
        seq = episode_to_sequence(ep, self.head.weight.device)
        s = seq.grid_size
        seq_frac = (seq.seq_idx.float() / max(1, s * s - 1))
        x = self.embed(seq.tokens, seq.row, seq.col, seq.role,
                       seq_frac).unsqueeze(0)                   # [1,L,d]
        mem = self.deltanet(x)                                  # fast-weight memory
        h = self.encoder(mem)                                   # global attention
        q = h[0, seq.query_start:seq.query_start + seq.n_query]  # [Q,d]
        logits = self.head(q)                                    # [Q,out]
        return logits.reshape(1, self.out_channels, s, s)


def _deltanet_prefix_scan(R, B):
    """Inclusive prefix scan of affine transforms (R_t, B_t).

    ``R``, ``B`` are [B, L, d, d].  The combine operation
        (Ra, Ba) * (Rb, Bb) = (Ra@Rb, Ba@Rb + Bb)
    is associative with identity (I, 0), so the scan is a divide-and-conquer
    prefix sum.  Returns (prefix_R, prefix_B) where
        prefix_B[:, t] = B-part of (R_0@...@R_t, B_0@R_1@...@R_t + ... + B_t)
    which equals the fast-weight memory S_{t+1} after processing token t.
    """
    def rec(R, B):
        n = R.shape[1]
        if n == 1:
            return R, B
        m = n // 2
        LR, LB = rec(R[:, :m], B[:, :m])
        RR, RB = rec(R[:, m:], B[:, m:])
        tot_R = LR[:, -1]                                   # [B,d,d]  (left-half total transform)
        tot_B = LB[:, -1]                                   # [B,d,d]
        # right-half prefix = left-total o right-prefix = (tot_R @ Rr, tot_B @ Rr + Br)
        nR = tot_R.unsqueeze(1) @ RR                        # [B, n-m, d, d]
        nB = tot_B.unsqueeze(1) @ RR + RB                   # [B, n-m, d, d]
        return torch.cat([LR, nR], dim=1), torch.cat([LB, nB], dim=1)
    return rec(R, B)


# ---------------------------------------------------------------------------
# Parameter <-> CPPN substrate plumbing.
#
# The CPPN is queried at D-dimensional "substrate" points (4-D by default) to
# read off the phenotype's flat parameter vector.  There are two substrates:
#
#   * fixed substrate  -- a stable abstract hypercube, one point per parameter.
#   * dynamic substrate-- the same shape, but its CENTER is shifted by a
#     per-episode 4-D "task descriptor" derived from the query input grid.
#
# The shift lets the SAME genome encode different weights for different tasks:
# the CPPN sees the (shifted) coordinates and the descriptor's effect is
# baked into where on its response surface it is sampled.  The descriptor
# (query-grid geometry + demo count + colour statistics) is a frozen,
# hand-designed feature; a small learned net maps it to the 4-D shift.
# ---------------------------------------------------------------------------
@dataclass
class ParamCoords:
    """Stable flat parameter layout, D-D coords, and per-param init scale."""
    names: List[str]                 # flat param names, in parameters() order
    shapes: List[Tuple]              # per-parameter shape
    offsets: List[int]               # start index in the flat vector
    flat_size: int
    dim: int                         # substrate coordinate dimensionality (D)
    coords: torch.Tensor             # [N, D]  fixed substrate (abstract hypercube)
    scale: torch.Tensor              # [N]  per-param Xavier init scale


def _fan_in(shape, name: str) -> int:
    """Approximate fan-in for a layer parameter (0 for biases)."""
    if "bias" in name:
        return 0
    if len(shape) == 2:      # Linear (out, in)
        return shape[1]
    if len(shape) == 4:      # Conv2d (out_ch, in_ch, kh, kw)
        return shape[1] * shape[2] * shape[3]
    return 1


def build_param_scales(phen: nn.Module, device) -> torch.Tensor:
    """Per-element Xavier-uniform init scale, laid out flat like `parameters()`.

    The CPPN's tanh output is in [-1, 1]; multiplying it by this scale puts each
    encoded weight in a proper initialisation range (a few * std) so the
    phenotype starts at a sane init rather than near-constant.  This keeps the
    weight-delta loss well-conditioned.
    """
    flat = []
    for name, p in phen.named_parameters():
        if _fan_in(tuple(p.shape), name) == 0:
            s = 0.05
        else:
            s = math.sqrt(1.0 / max(1, _fan_in(tuple(p.shape), name)))
        flat.extend([s] * p.numel())
    return torch.tensor(flat, dtype=torch.float32, device=device)


def _element_positions(shape, flat_idx):
    """Fractional [0,1] position of a flat index along each tensor dim.

    For shape (a, b, c) and flat index i, returns [i//... fractions per dim],
    e.g. dim-0 fraction = row / (a-1).  1-D params get a single position.
    """
    nd = len(shape)
    rem = flat_idx
    pos = []
    for d in range(nd - 1, -1, -1):
        size = shape[d]
        digit = rem % size
        rem //= size
        pos.append(digit / max(1, size - 1) if size > 1 else 0.0)
    return list(reversed(pos))   # [frac_dim0, frac_dim1, ...]


_LAYER_TYPE_NAMES = ("bias", "linear", "embedding", "conv", "norm", "other")
_MAX_LAYER_TYPE = len(_LAYER_TYPE_NAMES) - 1


def _layer_type(name: str) -> int:
    """Map a parameter name to a discrete layer-type code in [0, _MAX_LAYER_TYPE]."""
    low = name.lower()
    if "bias" in low:
        return 0
    if "weight" in low and ("linear" in low or "proj" in low or "head" in low
                            or "fc" in low or "dense" in low or "gate" in low):
        return 1
    if "embedding" in low or "embed" in low:
        return 2
    if "conv" in low:
        return 3
    if "norm" in low or "gamma" in low or "weight" in low:
        return 4
    return _MAX_LAYER_TYPE


def build_param_coords(phen: nn.Module, dim: int, device,
                       schema: str = "layer") -> ParamCoords:
    """Assign each phenotype parameter a D-D coordinate in [0, 1]^D.

    ``schema`` selects the coordinate layout:

    * "layer" -- layer-aware 5-D schema: each coordinate encodes *which layer*
      a weight belongs to so the CPPN can specialise per layer type.
      5-D schema (``dim`` must be 5):
        axis 0  layer-type code   (bias/linear/embedding/conv/norm/other)
        axis 1  layer index       (position of the param within the network)
        axis 2  weight position along tensor dim 0 (row / out-channel)
        axis 3  weight position along tensor dim 1 (col / in-channel / height)
        axis 4  weight position along tensor dim 2 (conv depth; 0 for matrices)

    * "flat" -- the original abstract hypercube: a single mixed-radix grid
      over the flat parameter index.  A conv kernel and a DeltaNet beta
      projection are indistinguishable abstract points here.  Works for any
      ``dim``.

    * "kqv" -- kqv-aware 5-D schema for the pure_attn phenotype:
        axis 0  projection code   (k / q / v / other)
        axis 1  layer index
        axes 2-4  position within the tensor
      The key/query/value projection matrices get their own substrate
      regions, so a 4-output CPPN (one channel per k/q/v plus "rest") can
      specialise the generator per projection.

    Coordinates are normalised to [0, 1] per axis and built on the given device
    (CPU in practice, since the CPPN is kept on CPU).  The encoded vector is
    moved to the phenotype's device (differentiable) for the loss.  The CPPN is
    never moved with `.to()` because its node `children` lists would collide
    with `nn.Module.children()`.
    """
    names, shapes, offsets, flat = [], [], [], 0
    for name, p in phen.named_parameters():
        names.append(name)
        shapes.append(tuple(p.shape))
        offsets.append(flat)
        flat += p.numel()

    n = flat
    coords = torch.zeros(n, dim, dtype=torch.float32)

    if schema == "layer":
        if dim != 5:
            raise ValueError(
                f"layer-aware substrate requires dim=5, got {dim}")
        n_params = max(1, len(names))
        for idx, (name, shape, r0) in enumerate(zip(names, shapes, offsets)):
            ltype = _layer_type(name)
            ax0 = ltype / max(1, _MAX_LAYER_TYPE)             # layer type in [0,1]
            ax1 = idx / max(1, n_params - 1) if n_params > 1 else 0.0
            numel = 1
            for s in shape:
                numel *= s
            for i in range(numel):
                coords[r0 + i, 0] = ax0
                coords[r0 + i, 1] = ax1
                fracs = _element_positions(shape, i)
                for axis in range(2, dim):
                    if axis - 2 < len(fracs):
                        coords[r0 + i, axis] = fracs[axis - 2]
    elif schema == "flat":
        side = max(1, int(round(n ** (1.0 / dim))))
        for i in range(n):
            m = i
            for axis in range(dim - 1, -1, -1):
                base = side ** axis
                digit = (m // base) % side
                coords[i, axis] = digit / max(1, side - 1) if side > 1 else 0.0
    elif schema == "kqv":
        # kqv-aware schema for the pure_attn phenotype (``dim`` must be 5).
        #   axis 0  projection code:  k / q / v / other
        #   axis 1  layer index       (position of the param within the network)
        #   axes 2-4  position within the parameter tensor
        # The three projection matrices live in well-separated substrate
        # regions, so a (multi-output) CPPN can place the k/q/v response
        # surface structures independently of everything else.
        if dim != 5:
            raise ValueError(f"kqv substrate requires dim=5, got {dim}")
        n_params = max(1, len(names))
        for idx, (name, shape, r0) in enumerate(zip(names, shapes, offsets)):
            low = name.lower()
            if "k_proj" in low:
                ax0 = 0.0
            elif "q_proj" in low:
                ax0 = 1.0 / 3.0
            elif "v_proj" in low:
                ax0 = 2.0 / 3.0
            else:
                ax0 = 1.0
            ax1 = idx / max(1, n_params - 1) if n_params > 1 else 0.0
            numel = 1
            for s in shape:
                numel *= s
            for i in range(numel):
                coords[r0 + i, 0] = ax0
                coords[r0 + i, 1] = ax1
                fracs = _element_positions(shape, i)
                for axis in range(2, dim):
                    if axis - 2 < len(fracs):
                        coords[r0 + i, axis] = fracs[axis - 2]
    else:
        raise ValueError(f"unknown coordinate schema: {schema!r}")

    scale = build_param_scales(phen, device)
    return ParamCoords(names, shapes, offsets, flat, dim, coords, scale)


# ---------------------------------------------------------------------------
# Deterministic dynamic substrate: shift the query coordinates by a per-task
# 5-D vector derived from simple statistics of the query input grid.  No
# learned parameters -- the shift is a closed-form map from grid geometry.
# Each substrate axis is given a meaningful interpretation:
#
#   axis 0 (x)  -> vertical   centre of mass of coloured cells
#   axis 1 (y)  -> horizontal centre of mass of coloured cells
#   axis 2     -> colour "balance"  (fraction in the high half of colours -
#                 fraction in the low half, signed, so it spans [-1, 1])
#   axis 3     -> coloured-cell density  (fraction of cells that are non-zero)
#   axis 4     -> spatial extent  (bbox diagonal of coloured cells / grid size)
#
# The result is in [-1, 1]^D; the substrate is then
#     coords + radius * shift   (clamped to the CPPN's [0, 1]^D domain).
# This is task-specific and deterministic: the same grid always yields the same
# shift, and different grids yield different shifts.
# ---------------------------------------------------------------------------
def task_shift(ep: Episode, device, dim: int = 5) -> torch.Tensor:
    """Closed-form 5-D substrate shift from the query input grid's statistics."""
    q = ep.query_in.squeeze(0)                       # [S, S] in [0, 1]
    cols = (q * 9.0).round().long()                  # [S, S] colour ids
    s = q.shape[0]
    nz = cols > 0
    if nz.any():
        ys, xs = torch.nonzero(nz, as_tuple=True)
        cy = ys.float().mean() / max(1, s - 1)        # vertical centre   [0,1]
        cx = xs.float().mean() / max(1, s - 1)        # horizontal centre [0,1]
        h = (ys.max() - ys.min()) / max(1, s - 1)     # vertical extent   [0,1]
        w = (xs.max() - xs.min()) / max(1, s - 1)     # horizontal extent [0,1]
        extent = float(torch.sqrt((h * h + w * w).clamp(max=2.0)) / 2.0)  # [0,1]
    else:
        cy, cx, extent = 0.5, 0.5, 0.0               # grid is empty
    # colour balance: high colours (5..9) minus low colours (1..4), signed [-1,1]
    cfloat = cols.float()
    high = (cfloat >= 5).float()
    low = (cfloat <= 4) & (cfloat > 0)
    total = max(1.0, float((cfloat > 0).float().sum()))
    balance = (high.sum() - low.sum()) / total
    density = float(nz.float().mean())
    # 5 meaningful axes, each centred in [-1, 1]:
    #   0: vertical centre of mass   1: horizontal centre of mass
    #   2: colour balance            3: coloured-cell density
    #   4: spatial extent of coloured cells
    vals = [2.0 * cy - 1.0, 2.0 * cx - 1.0, balance,
            2.0 * density - 1.0, 2.0 * extent - 1.0]
    while len(vals) < dim:
        vals.append(0.0)
    shift = torch.tensor(vals[:dim], dtype=torch.float32, device=device)
    return shift.unsqueeze(0)     # [1, D]


def dynamic_substrate(coords: torch.Tensor, shift: torch.Tensor,
                      radius: float) -> torch.Tensor:
    """Shift the fixed substrate by `shift` (in CPPN-domain units), clamped to
    [0, 1]^D.  `coords` is [N, D], `shift` is [1, D].  Uses a non-in-place
    clamp so the result stays differentiable if `shift` is trained."""
    return (coords + radius * shift).clamp(0.0, 1.0)


def encode(cppn: TorchCPPN, pc: ParamCoords, substrate: torch.Tensor) -> torch.Tensor:
    """Differentiable encode of the phenotype's flat parameter vector.

    `substrate` is [N, D] -- the (fixed or dynamic) query points.  The CPPN's
    tanh output is scaled per-parameter to a proper init magnitude.

    Multi-output CPPNs (4 channels for pure_attn: k / q / v / rest) are
    dispatched per parameter: each parameter's coordinates are fed to every
    output channel, and the channel owning that parameter (by name prefix)
    supplies its encoded values -- the other channels are evaluated on the
    same points but never selected, so they carry no gradient.
    """
    if pc.dim == 5 and hasattr(cppn, "output_nodes") and len(cppn.output_nodes) > 1:
        kwargs = {f"c{i}": substrate[:, i] for i in range(pc.dim)}
        outs = cppn(**kwargs)                          # output_mode="tensor" -> [N, C]
        if outs.dim() == 1:
            outs = outs.unsqueeze(1)
        elif outs.dim() == 2:
            outs = outs.transpose(0, 1)                # -> [C, N]
        chunks = []
        for name, shape, off in zip(pc.names, pc.shapes, pc.offsets):
            numel = int(np.prod(shape))
            low = name.lower()
            if "k_proj" in low:
                ch = 0
            elif "q_proj" in low:
                ch = 1
            elif "v_proj" in low:
                ch = 2
            else:
                ch = outs.shape[0] - 1
            chunks.append(outs[ch, off:off + numel])
        out = torch.cat(chunks, dim=0)
    else:
        kwargs = {
            f"c{i}": substrate[:, i] for i in range(pc.dim)
        }
        out = cppn(**kwargs).reshape(-1)
    if out.shape[0] != pc.flat_size:
        raise RuntimeError(
            f"CPPN encoded {out.shape[0]} values but phenotype has "
            f"{pc.flat_size} parameters.")
    return out * pc.scale


def load_param_vector(phen: nn.Module, pc: ParamCoords, vec: torch.Tensor) -> None:
    """Copy a flat parameter vector into `phen` (no grad)."""
    vec = vec.detach().to(next(phen.parameters()).device)
    for name, p, off, shape in zip(pc.names, phen.parameters(), pc.offsets, pc.shapes):
        p.data.copy_(vec[off:off + int(np.prod(shape))].reshape(shape))


def param_vector(phen: nn.Module, pc: ParamCoords) -> torch.Tensor:
    """Flatten the phenotype's current parameters (detached)."""
    parts = [p.detach().reshape(-1) for p in phen.parameters()]
    return torch.cat(parts, dim=-1) if parts else torch.zeros(0, device=next(phen.parameters()).device)


# ---------------------------------------------------------------------------
# ARC task loading (via arckit) and episode construction.
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    task_id: str
    demo_in: torch.Tensor            # [D, C, S, S]
    demo_out: torch.Tensor           # [D, C, S, S]
    query_in: torch.Tensor           # [C, S, S]
    target: torch.Tensor             # [C, S, S]
    shape: Tuple[int, int]           # (h, w) of the true (unpadded) target grid
    n_demos: int
    # Per-grid TRUE (unpadded) shapes, for sequence truncation.
    demo_in_shape: List[Tuple[int, int]]   # [D] (h, w); padded demo slots are (0, 0)
    demo_out_shape: List[Tuple[int, int]]  # [D] (h, w); padded slots (0, 0)
    query_in_shape: Tuple[int, int]        # (h, w) of the query input grid


def _grid_to_tensor(grid, grid_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    arr = np.asarray(grid, dtype=np.float32)
    h, w = arr.shape
    out = np.zeros((grid_size, grid_size), dtype=np.float32)
    out[:h, :w] = arr / 9.0          # normalise colours to [0, 1]
    return torch.from_numpy(out).unsqueeze(0), (h, w)


@dataclass
class ARCStore:
    grid_size: int
    tasks: list
    eligible: list = field(default_factory=list)

    @classmethod
    def load(cls, grid_size: int, seed: int = 0):
        import arckit
        train, _test = arckit.load_data()
        tasks = list(train.tasks)
        rng = random.Random(seed)
        eligible = []
        for t in tasks:
            pairs = list(getattr(t, "train", []))
            if len(pairs) < 2:
                continue
            # filter to small grids so they fit grid_size and stay cheap
            ok = True
            for p in pairs:
                x, y = np.asarray(p[0]), np.asarray(p[1])
                if (x.shape[0] > grid_size or x.shape[1] > grid_size or
                        y.shape[0] > grid_size or y.shape[1] > grid_size):
                    ok = False
                    break
            if ok:
                eligible.append(t)
        rng.shuffle(eligible)
        return cls(grid_size, tasks, eligible)

    def make_episode(self, task, max_demos: int, rng: random.Random) -> Episode:
        pairs = list(getattr(task, "train", []))
        q = rng.randrange(len(pairs))
        qx, qy = pairs[q]
        demo_pairs = [p for i, p in enumerate(pairs) if i != q]
        rng.shuffle(demo_pairs)
        demo_pairs = demo_pairs[:max_demos]

        d_in, d_out, din_sh, dout_sh = [], [], [], []
        for p in demo_pairs:
            x, y = p[0], p[1]
            ti, sh_x = _grid_to_tensor(x, self.grid_size)
            to, sh_y = _grid_to_tensor(y, self.grid_size)
            d_in.append(ti)
            d_out.append(to)
            din_sh.append(sh_x)
            dout_sh.append(sh_y)
        # pad to exactly max_demos with zeros if a task has fewer
        while len(d_in) < max_demos:
            d_in.append(torch.zeros_like(d_in[0]))
            d_out.append(torch.zeros_like(d_out[0]))
            din_sh.append((0, 0))
            dout_sh.append((0, 0))

        query_in, sh_qx = _grid_to_tensor(qx, self.grid_size)
        target, shape = _grid_to_tensor(qy, self.grid_size)

        return Episode(
            task_id=str(getattr(task, "id", "unknown")),
            demo_in=torch.stack(d_in, dim=0),
            demo_out=torch.stack(d_out, dim=0),
            query_in=query_in,
            target=target,
            shape=shape,
            n_demos=len([p for p in pairs if p is not pairs[q]]),
            demo_in_shape=din_sh,
            demo_out_shape=dout_sh,
            query_in_shape=sh_qx,
        )


def sample_sgd_task(store: ARCStore, cfg: Config, rng: random.Random,
                    exclude_ids: set) -> Episode:
    """Sample a single SGD training task, avoiding tasks in `exclude_ids`."""
    eligible = [i for i, t in enumerate(store.eligible)
                if str(getattr(t, "id", f"task{i}")) not in exclude_ids]
    if not eligible:
        eligible = list(range(len(store.eligible)))
    i = rng.choice(eligible)
    return store.make_episode(store.eligible[i], cfg.max_demos, rng)


def sample_tasks(store: ARCStore, cfg: Config, gen: int,
                 rng: random.Random) -> Tuple[Episode, List[Episode]]:
    """Pick the shared SGD task and the shared eval task sample for a generation."""
    idx = rng.sample(range(len(store.eligible)), min(len(store.eligible), 1 + cfg.n_eval))
    sgd_ep = store.make_episode(store.eligible[idx[0]], cfg.max_demos, rng)
    eval_eps = [
        store.make_episode(store.eligible[i], cfg.max_demos, rng)
        for i in idx[1:1 + cfg.n_eval]
    ]
    return sgd_ep, eval_eps


def sample_fixed_eval(store: ARCStore, cfg: Config,
                      rng: random.Random) -> List[Episode]:
    """Sample the shared eval task set ONCE, reused across all generations.

    With a stable selection signal (same eval tasks every generation), the
    population can actually climb toward the best genome for *those* tasks
    instead of chasing a different target each generation.
    """
    n = min(len(store.eligible), cfg.n_eval)
    idx = rng.sample(range(len(store.eligible)), n)
    return [store.make_episode(store.eligible[i], cfg.max_demos, rng) for i in idx]


# ---------------------------------------------------------------------------
# Phenotype adaptation (SGD) and accuracy.
# ---------------------------------------------------------------------------
def _ep_on_device(ep: Episode, device) -> Episode:
    """Return a copy of the Episode with all grid tensors on `device`."""
    import dataclasses
    return dataclasses.replace(
        ep,
        demo_in=ep.demo_in.to(device),
        demo_out=ep.demo_out.to(device),
        query_in=ep.query_in.to(device),
        target=ep.target.to(device),
    )


def sgd_adapt(phen: nn.Module, ep: Episode, steps: int, lr: float,
              device) -> float:
    """Train the phenotype on one episode with plain SGD. Returns final task loss."""
    phen.train()
    opt = torch.optim.SGD(phen.parameters(), lr=lr)
    epd = _ep_on_device(ep, device)
    # Cross-entropy needs integer colour classes in [0, 9], shape [1, S, S].
    target = (epd.target.squeeze(0) * 9.0).round().long().unsqueeze(0)   # [1, S, S]
    last_loss = 0.0
    for _ in range(steps):
        opt.zero_grad()
        pred = phen(epd)                             # [1, out_channels, S, S]
        loss = F.cross_entropy(pred, target)
        loss.backward()
        # clip at 10 (NOT 1): the healthy gated mix trains with grad norms
        # ~10-80, so clip 1.0 throttles it (12-step accuracy 0.24 vs 0.90
        # unclipped, measured on the same genome/task). The plain-attention
        # line DIVERGES unclipped (loss 6.8 -> 1e10 -> NaN by step 8, grad
        # norm 260+ at step 4); clip 10 tames it (loss converges to 0.25,
        # same genome/task) without touching the gated line, because the
        # divergence's first runaway step already exceeds norm 10 while
        # healthy training stays below it. This is what keeps the delta
        # meta-loss finite so the CPPN meta-optimizer never NaNs its
        # weights (which would make every genome encode "all-background"
        # and freeze fitness at the background fraction — the 30-gen
        # 0.57814 tie observed on the unfixed pure_attn run).
        torch.nn.utils.clip_grad_norm_(phen.parameters(), 10.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    phen.eval()
    return last_loss


def task_ce_loss(phen: nn.Module, epd: Episode) -> torch.Tensor:
    """Cross-entropy of the phenotype's prediction on one episode (differentiable)."""
    pred = phen(epd)                                 # [1, out_channels, S, S]
    target = (epd.target.squeeze(0) * 9.0).round().long().unsqueeze(0)   # [1, S, S]
    return F.cross_entropy(pred, target)


def maml_meta_loss(phen: nn.Module, ep: Episode, W_init: torch.Tensor,
                   pc: ParamCoords, steps: int, lr: float,
                   device) -> torch.Tensor:
    """FOMAML-style second-order meta-loss, differentiable in ``W_init``.

    First-order (delta) loss:  L = ||W_after - W_init||^2, with W_after
    detached -- the CPPN only feels "where the weights landed".

    Second-order (MAML) loss:  L = task_loss(W_after(W_init)), backpropagated
    through the SGD steps -- the CPPN feels "how to initialise so that the
    post-adaptation loss is low".  This is the standard fix for flat
    first-order plateaus.

    Implementation (FOMAML approximation, cheap on memory): the inner SGD
    steps run with the phenotype's leaf parameters loaded from the current
    weight vector; the per-step gradient is taken w.r.t. the parameters and
    *detached* before updating the weight vector (so ``W`` stays a
    differentiable function of ``W_init`` while no per-step activation graph
    is retained).  At the end, ``g_final = grad_params L_task(W_final)`` is
    computed and detached, and the meta-loss is
        meta = dot( g_final, W_init )
    whose gradient w.r.t. ``W_init`` is exactly ``g_final`` -- the first-order
    MAML meta-gradient  ∇_W L_task(W_after).  Using ``W_init`` (not ``W``) in
    the dot product is what drops the second-order Jacobian term.
    """
    epd = _ep_on_device(ep, device)
    params = list(phen.parameters())
    W = W_init
    for _ in range(steps):
        load_param_vector(phen, pc, W.detach())
        loss = task_ce_loss(phen, epd)
        grads = torch.autograd.grad(loss, params)
        g = torch.cat([g.reshape(-1).detach() for g in grads])
        W = W - lr * g                              # differentiable in W_init
    # final task loss at the adapted weights
    load_param_vector(phen, pc, W.detach())
    loss_final = task_ce_loss(phen, epd)
    grads_final = torch.autograd.grad(loss_final, params)
    g_final = torch.cat([g.reshape(-1).detach() for g in grads_final])
    # dot(g_final, W_init) -> d(meta)/d(W_init) = g_final  (FOMAML meta-grad)
    return (g_final * W_init).sum()


def episode_accuracy(phen: nn.Module, ep: Episode, device) -> float:
    """Cell-wise match rate of the phenotype's query prediction on the target grid."""
    phen.eval()
    epd = _ep_on_device(ep, device)
    with torch.no_grad():
        pred = phen(epd)
        pred = pred.argmax(dim=1).squeeze(0)          # [S, S] colour indices
    # target was normalised by /9; recover integer colours for the comparison.
    target = (epd.target.squeeze(0) * 9.0).round().long()   # [S, S]
    h, w = ep.shape
    match = (pred[:h, :w] == target[:h, :w]).float().mean()
    return float(match.item())


# ---------------------------------------------------------------------------
# Per-genome meta-training + fitness.
# ---------------------------------------------------------------------------
def _substrate_for(ep: Episode, pc: ParamCoords, cfg: Config) -> torch.Tensor:
    """Return the [N, D] CPPN query substrate for this episode.

    Fixed mode -> the abstract hypercube.  Dynamic mode -> the hypercube shifted
    by a closed-form 4-D shift derived from the query grid's statistics (clamped
    to the CPPN domain).  Kept on CPU (the CPPN's device).
    """
    coords = pc.coords  # [N, D] on CPU
    if not cfg.dynamic_substrate:
        return coords
    shift = task_shift(ep, coords.device, pc.dim)     # [1, D]
    return dynamic_substrate(coords, shift, cfg.substrate_radius)


def evaluate_genome(genome, neat_cfg, pc: ParamCoords, sgd_ep: Episode,
                    eval_eps: List[Episode], cfg: Config, device) -> float:
    """Run the full per-genome pipeline and return fitness (mean eval accuracy).

    The CPPN stays on CPU (TorchCPPN nodes carry a plain `children` list that
    would break `nn.Module.to()`).  The encoded vector is moved to `device` for
    the delta loss; the move is differentiable so grad still flows into the CPPN.

    In dynamic-substrate mode the CPPN is queried at a per-task shifted 4-D
    substrate (a closed-form function of the query grid's statistics), so the
    same genome encodes task-dependent weights.
    """
    dim = pc.dim
    n_out = cfg.cppn_num_outputs
    cppn = create_torch_cppn(
        genome=genome,
        config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(dim)],
        node_names=["k", "q", "v", "rest"][:n_out],
        output_mode="tensor" if n_out > 1 else "auto",
    )  # CPU
    cppn_opt = torch.optim.SGD(cppn.parameters(), lr=cfg.cppn_lr)

    phen = build_phenotype(cfg)  # on `device`

    # --- meta-training over cppn_iters ---
    if cfg.meta_loss == "maml":
        # Second-order (FOMAML): minimise the post-adaptation task loss,
        # backpropagated through the inner SGD steps (detached inner grads).
        for _ in range(cfg.cppn_iters):
            substrate = _substrate_for(sgd_ep, pc, cfg)
            W_init = encode(cppn, pc, substrate).to(device)   # grad -> cppn
            loss = maml_meta_loss(phen, sgd_ep, W_init, pc,
                                  cfg.sgd_steps, cfg.sgd_lr, device)
            cppn_opt.zero_grad()
            loss.backward()
            cppn_opt.step()
    else:
        # First-order (delta): minimise ||W_after - W_init||^2 with W_after
        # detached -- the CPPN only feels where the weights landed.
        #
        # Two weightings (cfg.meta_loss_weight):
        #   "mean"      -- mean over all coordinates.  Tiny tensors (e.g. a
        #                  3-value gate) get ~N/3 * less gradient than a
        #                  24x24 projection, so they barely learn.
        #   "per_param" -- mean of per-tensor squared deltas.  Every parameter
        #                  tensor contributes equally, so small tensors like
        #                  architecture gates learn on the same timescale as
        #                  large weight matrices.
        #
        # cfg.gate_meta == "task": swap the gate's delta signal for the
        # post-adaptation task-loss gradient (see below in the loop).
        # cfg.gate_task_scale damps the swap: scale=1.0 is the full task
        # signal, 0.0 is pure delta; in between the gate feels
        #   (1-s) * delta_term + s * task_term
        # Pure task (s=1) over-commits the gate to one branch per task and
        # overfits the branch choice, so a damped blend is the default
        # experiment.
        gate_idx = None
        for _n, name in enumerate(pc.names):
            if name == "gate":
                gate_idx = _n
                break
        gate_slice = (pc.offsets[gate_idx], int(np.prod(pc.shapes[gate_idx]))) \
            if gate_idx is not None else None
        gate_scale = (float(cfg.gate_task_scale)
                      if cfg.gate_meta == "task" and gate_slice is not None
                      else 0.0)
        use_task_gate = gate_scale > 0.0
        for _ in range(cfg.cppn_iters):
            substrate = _substrate_for(sgd_ep, pc, cfg)
            W_init = encode(cppn, pc, substrate).to(device)   # [N], grad -> cppn (CPU)
            load_param_vector(phen, pc, W_init)
            sgd_adapt(phen, sgd_ep, cfg.sgd_steps, cfg.sgd_lr, device)
            if cfg.meta_loss_weight == "per_param":
                # per-tensor squared delta, equal weight per tensor
                wa_parts = [p.detach().reshape(-1) for p in phen.parameters()]
                wi_parts = [W_init[o:o + int(np.prod(s))]
                            for o, s in zip(pc.offsets, pc.shapes)]
                if use_task_gate:
                    off, size = gate_slice
                    gate_delta_mean = (((wi_parts[gate_idx] -
                                         wa_parts[gate_idx]) ** 2).mean())
                    epd = _ep_on_device(sgd_ep, device)
                    g_gate = torch.autograd.grad(
                        task_ce_loss(phen, epd), phen.gate)[0].detach()
                    # gate term inside the per-tensor mean: its weight is
                    # 1/T; the blend is (1-s)*delta + s*task, so multiply
                    # the replacement by 1/T too.
                    gate_term = ((1.0 - gate_scale) * gate_delta_mean
                                 + gate_scale
                                 * ((g_gate.reshape(-1)
                                     * W_init[off:off + size]).sum()
                                    / len(wa_parts)))
                    terms = [((w - i) ** 2).mean()
                             for _i, (w, i) in enumerate(zip(wa_parts, wi_parts))
                             if _i != gate_idx]
                    loss = torch.stack([*terms, gate_term]).mean()
                else:
                    loss = torch.stack([((w - i) ** 2).mean()
                                        for w, i in zip(wa_parts, wi_parts)]).mean()
            else:
                W_after = param_vector(phen, pc)       # on `device`, detached
                delta = W_after - W_init
                if use_task_gate:
                    # Blend between the two endpoints:
                    #   s=0 -> mean_all(delta^2)                        (pure delta)
                    #   s=1 -> mean_{non-gate}(delta^2) + dot(g, W_gate) (pure task,
                    #          as shipped in the first gate_meta=task run)
                    #   loss = (1-s)*mean_all + s*(mean_non_gate + dot)
                    off, size = gate_slice
                    epd = _ep_on_device(sgd_ep, device)
                    g_gate = torch.autograd.grad(
                        task_ce_loss(phen, epd), phen.gate)[0].detach()
                    n_all = delta.numel()
                    n_ng = n_all - size
                    part_all = (delta * delta).sum() / n_all
                    ng = torch.cat([delta[:off], delta[off + size:]])
                    part_ng = (ng * ng).sum() / n_ng
                    dot = (g_gate.reshape(-1) * W_init[off:off + size]).sum()
                    loss = ((1.0 - gate_scale) * part_all
                            + gate_scale * (part_ng + dot))
                else:
                    loss = (delta * delta).mean()
            # NaN guard: if the inner adaptation diverged (loss/W_after non-
            # finite), the delta loss is NaN and stepping the CPPN with it
            # NaNs the CPPN's own weights, which then makes EVERY genome
            # encode the "all-background" prediction and freezes the whole
            # run's fitness at the background fraction. Skip the meta-update
            # (leave the CPPN at its current weights) instead.
            if not torch.isfinite(loss):
                cppn_opt.zero_grad()
                continue
            cppn_opt.zero_grad()
            loss.backward()
            cppn_opt.step()

    # --- evaluation on the shared eval task sample ---
    phen_eval = build_phenotype(cfg)
    accs = []
    for ep in eval_eps:
        substrate = _substrate_for(ep, pc, cfg)
        with torch.no_grad():
            W_init = encode(cppn, pc, substrate).to(device)
        load_param_vector(phen_eval, pc, W_init)
        sgd_adapt(phen_eval, ep, cfg.eval_sgd_steps, cfg.sgd_lr, device)
        accs.append(episode_accuracy(phen_eval, ep, device))

    return float(np.mean(accs)) if accs else 0.0


# ---------------------------------------------------------------------------
# Outer NEAT loop driver.
# ---------------------------------------------------------------------------
def make_evaluator(cfg: Config):
    neat_cfg = build_neat_config(cfg)
    device = cfg.resolve_device()

    # The CPPN lives on CPU, so its substrate is built on CPU too.
    # The encoded vector is moved to `device` (differentiable) for the loss.
    template = build_phenotype(cfg)
    n_types = len({_layer_type(n) for n, _ in template.named_parameters()})
    schema = cfg.coord_schema
    if schema == "auto":
        # pure_attn: kqv-aware schema, so the key/query/value projection
        # matrices get their own substrate regions (matches the 4-output
        # CPPN: one channel per k/q/v plus "rest").
        if cfg.phenotype == "pure_attn":
            schema = "kqv"
        else:
            # A/B evidence (5 gens, seed 0, grid 10, dim 24): layer-aware clearly
            # wins for the mixed phenotype (4 types: .65 vs .43 winner) and
            # clearly LOSES for deltanet (3 types: .35 vs .65), because two of the
            # five substrate axes are near-constant when few types exist.  The
            # threshold is a heuristic calibrated on that single-seed comparison;
            # raise it (always-flat) if more seeds confirm the 3-type result.
            schema = "layer" if n_types >= 4 else "flat"
    if cfg.phenotype == "pure_attn" and schema != "kqv":
        raise ValueError(
            "pure_attn requires the kqv substrate schema "
            f"(got {schema!r})")
    pc = build_param_coords(template, cfg.substrate_dim, torch.device("cpu"),
                            schema=schema)

    store = ARCStore.load(cfg.grid_size, seed=cfg.seed)
    if len(store.eligible) < 1 + cfg.n_eval:
        raise RuntimeError(
            f"Only {len(store.eligible)} eligible ARC tasks (grid<= {cfg.grid_size}); "
            f"need at least {1 + cfg.n_eval}.")

    print(f"Phenotype: {cfg.phenotype}  |  params: {pc.flat_size}  "
          f"(channels={cfg.channels}, attn_dim={cfg.attn_dim}, "
          f"S={cfg.grid_size}, out={cfg.out_channels})")
    resolved = f" (auto, {n_types} layer types)" if cfg.coord_schema == "auto" else ""
    print(f"Substrate: dim={cfg.substrate_dim}  schema={schema}{resolved}  "
          f"mode={'dynamic (task stats)' if cfg.dynamic_substrate else 'fixed'}  "
          f"radius={cfg.substrate_radius}")
    print(f"Eligible ARC tasks: {len(store.eligible)}  |  pop={cfg.pop_size}  "
          f"cppn_hidden={cfg.cppn_num_hidden}  device={device}")

    state = {"gen": 0}

    # Fixed eval set: sampled once, reused for every generation so the
    # selection signal is stable.  The SGD task still varies per generation
    # but never collides with an eval task.
    fixed_eval = None
    fixed_eval_ids = set()
    if cfg.fixed_eval:
        fixed_rng = random.Random(cfg.seed + 99991)
        fixed_eval = sample_fixed_eval(store, cfg, fixed_rng)
        fixed_eval_ids = {e.task_id for e in fixed_eval}
        print(f"Fixed eval set ({len(fixed_eval)} tasks): "
              f"{[e.task_id for e in fixed_eval]}")

    def eval_genomes(genomes, config):
        state["gen"] += 1
        gen = state["gen"]
        rng = random.Random(cfg.seed + gen * 7919)
        if fixed_eval is not None:
            sgd_ep = sample_sgd_task(store, cfg, rng, fixed_eval_ids)
            eval_eps = fixed_eval
        else:
            sgd_ep, eval_eps = sample_tasks(store, cfg, gen, rng)
        t0 = time.time()
        for genome_id, genome in genomes:
            try:
                fit = evaluate_genome(genome, neat_cfg, pc, sgd_ep,
                                      eval_eps, cfg, device)
            except Exception as e:  # keep the population alive on any genome error
                fit = 0.0
                print(f"  [gen {gen}] genome {genome_id} failed: {e}")
            genome.fitness = fit
        dt = time.time() - t0
        fits = [g.fitness for _, g in genomes]
        best = max(fits) if fits else 0.0
        print(f"[gen {gen}] sgd_task={sgd_ep.task_id} eval={[e.task_id for e in eval_eps]} "
              f"best={best:.4f} mean={np.mean(fits):.4f} ({dt:.1f}s)")
        if gen % cfg.log_every == 0 and best > 0:
            print(f"    tasks solved (acc>0.99): "
                  f"{sum(1 for a in fits if a > 0.99)}/{len(fits)}")

    population = neat.Population(neat_cfg)
    return population, eval_genomes, neat_cfg


def run(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = cfg.resolve_device()
    print(f"Using device: {device}")

    population, eval_genomes, neat_cfg = make_evaluator(cfg)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    n = max(1, cfg.generations)
    winner = population.run(eval_genomes, n=n)

    if winner is None:
        print("No winner returned.")
        return None
    print("\n=== Done ===")
    print("Winner fitness:", winner.fitness)
    print("Winner genome:")
    print(winner)

    # Persist the winner for downstream inspection.
    import pickle
    out = os.path.join(_PROJECT_DIR, cfg.save_name)
    with open(out, "wb") as f:
        pickle.dump(winner, f)
    print(f"Saved winner genome -> {out}")
    return winner


def smoke(cfg: Config):
    """Tiny end-to-end check: 1 generation, few genomes, few steps."""
    import dataclasses
    cfg2 = dataclasses.replace(
        cfg,
        cppn_iters=2, sgd_steps=6,
        n_eval=2, eval_sgd_steps=4,
        pop_size=3, cppn_num_hidden=6, generations=1,
        log_every=1,
    )
    print("Smoke test config:", cfg2)
    run(cfg2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="CPPN -> phenotype weight-delta ARC meta-training")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-gen smoke test")
    ap.add_argument("--generations", type=int, default=0, help="outer generations (0 = smoke default)")
    ap.add_argument("--pop-size", type=int, default=12)
    ap.add_argument("--cppn-hidden", type=int, default=14)
    ap.add_argument("--cppn-outputs", type=int, default=0,
                    help="CPPN output channels (0 = auto: 4 for pure_attn "
                         "(k/q/v/rest), 1 otherwise)")
    ap.add_argument("--grid-size", type=int, default=15)
    ap.add_argument("--phenotype", type=str, default="mixed",
                    choices=["conv", "attn", "deltanet", "mixed", "gated",
                             "pure_attn"],
                    help="phenotype architecture the CPPN encodes; 'gated' = "
                         "parallel deltanet+attention+conv branches with a "
                         "CPPN-encoded per-task architecture gate; "
                         "'pure_attn' = plain attention stack with separate "
                         "k/q/v projections, encoded by a 4-output CPPN "
                         "(one channel each for k, q, v, and the rest)")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--attn-dim", type=int, default=48)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--attn-layers", type=int, default=2)
    ap.add_argument("--max-demos", type=int, default=3)
    ap.add_argument("--cppn-iters", type=int, default=6)
    ap.add_argument("--sgd-steps", type=int, default=12)
    ap.add_argument("--eval-sgd-steps", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--dynamic-substrate", dest="dynamic_substrate",
                    action="store_true", default=None,
                    help="condition the CPPN query substrate on the task")
    ap.add_argument("--no-dynamic-substrate", dest="dynamic_substrate",
                    action="store_false",
                    help="use the fixed abstract substrate (original behaviour)")
    ap.add_argument("--substrate-dim", type=int, default=5,
                    help="CPPN coordinate dimensionality (5D layer-aware points)")
    ap.add_argument("--substrate-radius", type=float, default=0.25,
                    help="half-size of the per-task substrate shift window")
    ap.add_argument("--coord-schema", type=str, default="auto",
                    choices=["auto", "layer", "flat", "kqv"],
                    help="coordinate layout: auto (pure_attn -> kqv; layer-aware "
                         "if >=4 layer types, else flat), layer, flat, or kqv "
                         "(k/q/v projection regions for pure_attn)")
    ap.add_argument("--sgd-lr", type=float, default=5e-2)
    ap.add_argument("--cppn-lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-name", type=str, default="cppn_arc_winner.pkl",
                    help="filename for the saved winner genome")
    ap.add_argument("--fixed-eval", dest="fixed_eval", action="store_true",
                    help="sample the eval task set once and reuse it every "
                         "generation (stable selection signal)")
    ap.add_argument("--meta-loss", type=str, default="delta",
                    choices=["delta", "maml"],
                    help="outer meta-loss: 1st-order weight delta or 2nd-order "
                         "FOMAML (backprop through the inner SGD steps)")
    ap.add_argument("--meta-loss-weight", type=str, default="mean",
                    choices=["mean", "per_param"],
                    help="delta-loss weighting: mean over all coordinates (default) "
                         "or equal weight per parameter tensor (lets tiny tensors "
                         "like architecture gates learn on equal footing)")
    ap.add_argument("--gate-meta", type=str, default="delta",
                    choices=["delta", "task"],
                    help="meta-signal for the gated phenotype's architecture gate: "
                         "delta (default) keeps it in the weight-delta loss; task "
                         "swaps it for the post-adaptation task-loss gradient")
    ap.add_argument("--gate-task-scale", type=float, default=0.5,
                    help="with --gate-meta task, blend factor s in (1-s)*delta + "
                         "s*task for the gate (1.0 = full task signal, 0.0 = pure "
                         "delta). Full task over-commits the branch choice; damp "
                         "toward delta if that happens.")
    ap.add_argument("--gated-branches", type=int, default=3, choices=[3, 4],
                    help="gated phenotype: 3 = memory/attention/conv (default, "
                         "original); 4 adds a global whole-grid linear branch "
                         "(counting/global-statistics aimed)")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args(argv)
    dyn = Config().dynamic_substrate if args.dynamic_substrate is None else args.dynamic_substrate
    cppn_outputs = (args.cppn_outputs if args.cppn_outputs > 0
                    else (4 if args.phenotype == "pure_attn" else 1))

    if args.smoke:
        cfg = Config(grid_size=args.grid_size, max_demos=args.max_demos,
                     phenotype=args.phenotype, attn_dim=args.attn_dim,
                     attn_heads=args.attn_heads, attn_layers=args.attn_layers,
                     channels=args.channels, seed=args.seed, device=args.device,
                     dynamic_substrate=dyn, substrate_dim=args.substrate_dim,
                     substrate_radius=args.substrate_radius,
                     coord_schema=args.coord_schema,
                     gated_branches=args.gated_branches,
                     cppn_num_outputs=cppn_outputs,
                     save_name=args.save_name)
        smoke(cfg)
        return

    cfg = Config(
        grid_size=args.grid_size, max_demos=args.max_demos,
        phenotype=args.phenotype, channels=args.channels,
        attn_dim=args.attn_dim, attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        cppn_iters=args.cppn_iters, sgd_steps=args.sgd_steps,
        sgd_lr=args.sgd_lr, cppn_lr=args.cppn_lr,
        n_eval=args.n_eval, eval_sgd_steps=args.eval_sgd_steps,
        pop_size=args.pop_size, cppn_num_hidden=args.cppn_hidden,
        cppn_num_outputs=cppn_outputs,
        generations=args.generations, seed=args.seed, device=args.device,
        dynamic_substrate=dyn, substrate_dim=args.substrate_dim,
        substrate_radius=args.substrate_radius,
        coord_schema=args.coord_schema,
        gated_branches=args.gated_branches,
        save_name=args.save_name,
        fixed_eval=args.fixed_eval,
        meta_loss=args.meta_loss,
        meta_loss_weight=args.meta_loss_weight,
        gate_meta=args.gate_meta,
        gate_task_scale=args.gate_task_scale,
    )
    run(cfg)


if __name__ == "__main__":
    main()
