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

from sefer.controllers.reasoner import TinyReasoner
from sefer.tasks.synthetic_algebra import primitive_transport_sources

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
