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


def seed_all(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def flatten_params(module: nn.Module) -> torch.Tensor:
    """Flatten trainable parameters into one vector."""
    return torch.cat([p.detach().reshape(-1) for p in module.parameters() if p.requires_grad])

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
