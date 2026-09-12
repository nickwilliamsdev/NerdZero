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


class SyntheticTaskBatch:
    """Primitive training tasks plus strictly held-out compositions."""

    TRAIN_NAMES = ("roll+1", "negate", "flip", "identity")
    # Variable-length held-out programs (length 1..4). STOP is action 4 and
    # is not part of the ground-truth primitive tuple.
    COMPOSITIONS = {
        "roll2": (0, 0),
        "flip_after_roll": (0, 2),
        "roll_after_flip": (2, 0),
        "neg_roll_flip": (1, 0, 2),
        "roll_flip_roll": (0, 2, 0),
        "flip_roll_neg": (2, 0, 1),
        "roll_roll_flip_neg": (0, 0, 2, 1),
        "flip_roll_flip_roll": (2, 0, 2, 0),
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

    def program_training_sequences(self):
        """Programs of length 1..4 excluding exact held-out tuples.

        Identity primitive (3) is allowed inside programs. The separate STOP
        action is used only by the controller to terminate before max depth.
        """
        import itertools
        held_out = set(self.COMPOSITIONS.values())
        seqs = []
        for length in range(1, 5):
            for seq in itertools.product(range(4), repeat=length):
                if seq not in held_out:
                    seqs.append(seq)
        return tuple(seqs)

    def sample_program_batch(self, batch_size: int, device):
        # Balance lengths explicitly. Uniform sampling over all tuples would make
        # length-4 programs dominate (256 of 340 possible tuples), starving STOP.
        seqs = self.program_training_sequences()
        by_len = {L: [seq for seq in seqs if len(seq) == L] for L in range(1, 5)}
        sampled_lengths = torch.randint(1, 5, (batch_size,), device=device)
        chosen = []
        for L in sampled_lengths.cpu().tolist():
            pool = by_len[int(L)]
            idx = int(torch.randint(0, len(pool), (1,)).item())
            chosen.append(pool[idx])
        program = torch.full((batch_size, 4), 4, device=device, dtype=torch.long)
        lengths = sampled_lengths.long()
        demo_x = torch.randn(batch_size, self.dim, device=device)
        query_x = torch.randn(batch_size, self.dim, device=device)
        demo_y = torch.empty_like(demo_x)
        query_y = torch.empty_like(query_x)
        for i, seq in enumerate(chosen):
            program[i, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
            demo_y[i:i+1] = self.apply_composition(demo_x[i:i+1], seq)
            query_y[i:i+1] = self.apply_composition(query_x[i:i+1], seq)
        return demo_x, demo_y, query_x, query_y, program, lengths

def primitive_transport_sources(primitive_id: int, n: int, device) -> torch.Tensor:
    dst = torch.arange(n, device=device)
    if primitive_id == 0:  # torch.roll(x, +1): y[dst] = x[dst-1]
        return (dst - 1) % n
    if primitive_id == 1:  # negate keeps position
        return dst
    if primitive_id == 2:  # flip: y[dst] = x[n-1-dst]
        return (n - 1) - dst
    if primitive_id == 3:  # identity
        return dst
    raise ValueError(primitive_id)

def enumerate_primitive_programs(max_depth: int = 4):
    """All primitive programs of length 0..max_depth; STOP is implicit at the end."""
    import itertools
    programs = [tuple()]
    for length in range(1, max_depth + 1):
        programs.extend(itertools.product(range(4), repeat=length))
    return tuple(programs)
