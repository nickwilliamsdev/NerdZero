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


def make_hypercube_vertices(dim: int = 5, device=None) -> torch.Tensor:
    """Return all 2^dim vertices in {-1,+1}^dim."""
    n = 2 ** dim
    vals = []
    for i in range(n):
        bits = [(1.0 if ((i >> b) & 1) else -1.0) for b in range(dim)]
        vals.append(bits)
    return torch.tensor(vals, dtype=torch.float32, device=device)
