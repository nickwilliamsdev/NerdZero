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
