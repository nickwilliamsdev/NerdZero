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


class ARCGridEncoder(nn.Module):
    """Variable-size ARC grid -> 32 substrate slots.

    Cells receive color and x/y embeddings, a small convolutional context stem,
    then 32 learned substrate queries cross-attend to valid cells.  This keeps
    the existing 32-node Yetirah algebra while accepting any ARC grid <=30x30.
    """

    def __init__(
        self,
        max_size: int = 30,
        colors: int = 10,
        cell_dim: int = 64,
        slot_dim: int = 32,
        n_slots: int = 32,
        heads: int = 4,
        dropout: float = 0.0,
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
        self.slot_queries = nn.Parameter(torch.randn(n_slots, cell_dim) / math.sqrt(cell_dim))
        self.cross_attn = nn.MultiheadAttention(
            cell_dim, heads, dropout=dropout, batch_first=True
        )
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(cell_dim),
            nn.Linear(cell_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
            nn.LayerNorm(slot_dim),
        )

    def _cell_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        b, h, w = grid.shape
        rows = torch.arange(h, device=grid.device)
        cols = torch.arange(w, device=grid.device)
        x = self.color_emb(grid.long().clamp(0, self.colors - 1))
        x = x + self.row_emb(rows)[None, :, None, :] + self.col_emb(cols)[None, None, :, :]
        conv = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x + conv

    def forward(self, grid: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        b, h, w = grid.shape
        if h != self.max_size or w != self.max_size:
            raise ValueError(f"ARCGridEncoder expects padded {self.max_size}x{self.max_size} grids")
        cells = self._cell_tokens(grid).reshape(b, h * w, self.cell_dim)
        valid = shape_mask(shapes, self.max_size).reshape(b, h * w)
        q = self.slot_queries[None].expand(b, -1, -1)
        slots, _ = self.cross_attn(q, cells, cells, key_padding_mask=~valid, need_weights=False)
        return self.slot_proj(slots)


class ARCGridDecoder(nn.Module):
    """32 substrate slots -> ARC color logits + output height/width."""

    def __init__(
        self,
        max_size: int = 30,
        colors: int = 10,
        cell_dim: int = 64,
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
        logits = self.cell_head(cells).transpose(1, 2).reshape(
            b, self.colors, self.max_size, self.max_size
        )
        pooled = slots.mean(dim=1)
        shape_feat = self.shape_head(pooled)
        # Class 0 means size 1, class 29 means size 30.
        return logits, self.height_head(shape_feat), self.width_head(shape_feat)


def arc_grid_loss(
    color_logits: torch.Tensor,
    height_logits: torch.Tensor,
    width_logits: torch.Tensor,
    target: torch.Tensor,
    target_shapes: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Masked cell CE + output shape CE."""
    max_size = target.shape[-1]
    mask = shape_mask(target_shapes, max_size)
    per_cell = F.cross_entropy(color_logits, target.long(), reduction="none")
    color_loss = (per_cell * mask.float()).sum() / mask.sum().clamp_min(1)
    h_loss = F.cross_entropy(height_logits, (target_shapes[:, 0] - 1).long())
    w_loss = F.cross_entropy(width_logits, (target_shapes[:, 1] - 1).long())
    shape_loss = 0.5 * (h_loss + w_loss)
    with torch.no_grad():
        pred = color_logits.argmax(dim=1)
        pixel_acc = ((pred == target) & mask).sum().float() / mask.sum().clamp_min(1)
        shape_acc = 0.5 * (
            (height_logits.argmax(dim=-1) + 1 == target_shapes[:, 0]).float().mean()
            + (width_logits.argmax(dim=-1) + 1 == target_shapes[:, 1]).float().mean()
        )
    return color_loss, {
        "color_loss": color_loss.detach(),
        "shape_loss": shape_loss,
        "pixel_acc": pixel_acc,
        "shape_acc": shape_acc,
    }
