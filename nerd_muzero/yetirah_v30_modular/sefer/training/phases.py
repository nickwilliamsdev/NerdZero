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

def set_query_codec_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze the task-independent query coordinate system.

    Once warmup has learned E_Q(x) <-> x, operator learning should happen
    inside that fixed coordinate system rather than moving the coordinate
    system to accommodate each primitive.
    """
    for module in (model.query_encoder, model.query_decoder):
        for param in module.parameters():
            param.requires_grad_(trainable)

def set_algebra_trainable(model: TinyReasoner, trainable: bool):
    """Freeze/unfreeze the learned operator algebra while training program inference."""
    for p in model.core.parameters():
        p.requires_grad_(trainable)

def freeze_for_program_phase(model: TinyReasoner):
    """Preserve the validated algebra; train only goal policy + goal value."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.delta_state, model.program_controller, model.program_value):
        for p in module.parameters():
            p.requires_grad_(True)
