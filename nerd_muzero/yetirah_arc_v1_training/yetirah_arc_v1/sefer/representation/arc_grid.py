from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def shape_mask(shapes: torch.Tensor, max_size: int = 30) -> torch.Tensor:
    """Boolean [B,H,W] mask for unpadded cells."""
    rows = torch.arange(max_size, device=shapes.device)[None, :, None]
    cols = torch.arange(max_size, device=shapes.device)[None, None, :]
    return (rows < shapes[:, 0, None, None]) & (cols < shapes[:, 1, None, None])


class _LatentBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        a, _ = self.attn(z, z, z, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class ARCGridEncoder(nn.Module):
    """Variable-size ARC grid -> 32 abstract substrate slots.

    v1.2 keeps the 32-node Yetirah reasoning substrate but strengthens the
    information bottleneck: cells first receive local convolutional context,
    the learned slot queries cross-attend to all valid cells, then a small
    latent transformer refines relations between slots before projection to the
    32-dimensional operator space.
    """

    def __init__(
        self,
        max_size: int = 30,
        colors: int = 10,
        cell_dim: int = 96,
        slot_dim: int = 32,
        n_slots: int = 32,
        heads: int = 4,
        dropout: float = 0.0,
        latent_layers: int = 2,
    ):
        super().__init__()
        self.max_size = max_size
        self.colors = colors
        self.cell_dim = cell_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots

        self.color_emb = nn.Embedding(colors, cell_dim)
        self.row_emb = nn.Embedding(max_size, cell_dim)
        self.col_emb = nn.Embedding(max_size, cell_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(cell_dim, cell_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cell_dim, cell_dim, 3, padding=1),
            nn.GELU(),
        )
        self.cell_norm = nn.LayerNorm(cell_dim)
        self.slot_queries = nn.Parameter(torch.randn(n_slots, cell_dim) / math.sqrt(cell_dim))
        self.cross_attn = nn.MultiheadAttention(
            cell_dim, heads, dropout=dropout, batch_first=True
        )
        self.latent_blocks = nn.ModuleList(
            [_LatentBlock(cell_dim, heads, dropout) for _ in range(latent_layers)]
        )
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(cell_dim),
            nn.Linear(cell_dim, slot_dim * 4),
            nn.GELU(),
            nn.Linear(slot_dim * 4, slot_dim),
            nn.LayerNorm(slot_dim),
        )

    def _cell_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        b, h, w = grid.shape
        rows = torch.arange(h, device=grid.device)
        cols = torch.arange(w, device=grid.device)
        x = self.color_emb(grid.long().clamp(0, self.colors - 1))
        x = x + self.row_emb(rows)[None, :, None, :] + self.col_emb(cols)[None, None, :, :]
        conv = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.cell_norm(x + conv)

    def forward(self, grid: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        b, h, w = grid.shape
        if h != self.max_size or w != self.max_size:
            raise ValueError(f"ARCGridEncoder expects padded {self.max_size}x{self.max_size} grids")
        cells = self._cell_tokens(grid).reshape(b, h * w, self.cell_dim)
        valid = shape_mask(shapes, self.max_size).reshape(b, h * w)
        q = self.slot_queries[None].expand(b, -1, -1)
        attended, _ = self.cross_attn(q, cells, cells, key_padding_mask=~valid, need_weights=False)
        slots = q + attended
        for block in self.latent_blocks:
            slots = block(slots)
        return self.slot_proj(slots)


class ARCGridDecoder(nn.Module):
    """32 substrate slots -> ARC color logits + output height/width."""

    def __init__(
        self,
        max_size: int = 30,
        colors: int = 10,
        cell_dim: int = 96,
        slot_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_size = max_size
        self.colors = colors
        self.cell_dim = cell_dim
        self.row_query = nn.Embedding(max_size, cell_dim)
        self.col_query = nn.Embedding(max_size, cell_dim)
        self.slot_proj = nn.Linear(slot_dim, cell_dim)
        self.cross_attn = nn.MultiheadAttention(
            cell_dim, heads, dropout=dropout, batch_first=True
        )
        # Local refinement is much cheaper than 900-token self-attention and
        # restores a useful spatial inductive bias after latent decoding.
        self.local_refine = nn.Sequential(
            nn.Conv2d(cell_dim, cell_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cell_dim, cell_dim, 3, padding=1),
        )
        self.cell_head = nn.Sequential(
            nn.LayerNorm(cell_dim),
            nn.Linear(cell_dim, cell_dim),
            nn.GELU(),
            nn.Linear(cell_dim, colors),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
        )
        self.height_head = nn.Linear(slot_dim * 2, max_size)
        self.width_head = nn.Linear(slot_dim * 2, max_size)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = slots.shape[0]
        rows = self.row_query(torch.arange(self.max_size, device=slots.device))
        cols = self.col_query(torch.arange(self.max_size, device=slots.device))
        q = (rows[:, None, :] + cols[None, :, :]).reshape(1, self.max_size ** 2, self.cell_dim)
        q = q.expand(b, -1, -1)
        kv = self.slot_proj(slots)
        cells, _ = self.cross_attn(q, kv, kv, need_weights=False)
        cells = cells + q
        spatial = cells.transpose(1, 2).reshape(b, self.cell_dim, self.max_size, self.max_size)
        spatial = spatial + self.local_refine(spatial)
        cells = spatial.flatten(2).transpose(1, 2)
        logits = self.cell_head(cells).transpose(1, 2).reshape(
            b, self.colors, self.max_size, self.max_size
        )
        pooled = slots.mean(dim=1)
        shape_feat = self.shape_head(pooled)
        return logits, self.height_head(shape_feat), self.width_head(shape_feat)


def arc_grid_loss(
    color_logits: torch.Tensor,
    height_logits: torch.Tensor,
    width_logits: torch.Tensor,
    target: torch.Tensor,
    target_shapes: torch.Tensor,
    foreground_boost: float = 1.5,
) -> Tuple[torch.Tensor, dict]:
    """Frequency-balanced masked cell CE + output-shape CE.

    ARC grids are frequently dominated by color 0. A plain mean CE can obtain a
    deceptively good pixel score by focusing on background, so v1.2 balances
    colors by inverse-sqrt batch frequency and gives non-zero cells a modest
    additional weight. Raw/foreground/background accuracies are all reported.
    """
    max_size = target.shape[-1]
    mask = shape_mask(target_shapes, max_size)
    per_cell = F.cross_entropy(color_logits, target.long(), reduction="none")

    valid_target = target[mask].long()
    counts = torch.bincount(valid_target, minlength=color_logits.shape[1]).float()
    present = counts > 0
    class_w = torch.ones_like(counts)
    if present.any():
        inv = counts[present].clamp_min(1.0).rsqrt()
        inv = inv / inv.mean().clamp_min(1e-8)
        class_w[present] = inv.clamp(0.25, 4.0)
    cell_w = class_w[target.long().clamp(0, color_logits.shape[1] - 1)]
    if foreground_boost != 1.0:
        cell_w = cell_w * torch.where(
            target != 0,
            torch.as_tensor(foreground_boost, device=target.device, dtype=cell_w.dtype),
            torch.ones((), device=target.device, dtype=cell_w.dtype),
        )
    weighted_mask = mask.float() * cell_w
    color_loss = (per_cell * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

    h_loss = F.cross_entropy(height_logits, (target_shapes[:, 0] - 1).long())
    w_loss = F.cross_entropy(width_logits, (target_shapes[:, 1] - 1).long())
    shape_loss = 0.5 * (h_loss + w_loss)
    with torch.no_grad():
        pred = color_logits.argmax(dim=1)
        correct = pred == target
        pixel_acc = (correct & mask).sum().float() / mask.sum().clamp_min(1)
        fg = mask & (target != 0)
        bg = mask & (target == 0)
        foreground_acc = (correct & fg).sum().float() / fg.sum().clamp_min(1)
        background_acc = (correct & bg).sum().float() / bg.sum().clamp_min(1)
        shape_acc = 0.5 * (
            (height_logits.argmax(dim=-1) + 1 == target_shapes[:, 0]).float().mean()
            + (width_logits.argmax(dim=-1) + 1 == target_shapes[:, 1]).float().mean()
        )
    return color_loss, {
        "color_loss": color_loss.detach(),
        "shape_loss": shape_loss,
        "pixel_acc": pixel_acc,
        "foreground_acc": foreground_acc,
        "background_acc": background_acc,
        "shape_acc": shape_acc,
    }
