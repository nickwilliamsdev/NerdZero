from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def _pair_arrays(pair: Any) -> Tuple[np.ndarray, np.ndarray | None]:
    """Accept arckit tuple pairs and common object/dict representations."""
    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
        return np.asarray(pair[0], dtype=np.int64), np.asarray(pair[1], dtype=np.int64)
    if isinstance(pair, dict):
        x = pair.get("input")
        y = pair.get("output")
        return np.asarray(x, dtype=np.int64), None if y is None else np.asarray(y, dtype=np.int64)
    x = getattr(pair, "input", getattr(pair, "x", None))
    y = getattr(pair, "output", getattr(pair, "y", None))
    if x is None:
        raise TypeError(f"unsupported ARC pair type: {type(pair)!r}")
    return np.asarray(x, dtype=np.int64), None if y is None else np.asarray(y, dtype=np.int64)


def _tasks_from_split(split: str):
    try:
        import arckit
    except ImportError as exc:
        raise ImportError(
            "ARC-v1 requires arckit. Install/use the same environment where your existing arckit dependency lives."
        ) from exc
    train_set, eval_set = arckit.load_data()
    key = split.lower()
    dataset = train_set if key in {"train", "training"} else eval_set
    return list(dataset.tasks)


def pad_grid(grid: np.ndarray, max_size: int = 30) -> Tuple[np.ndarray, Tuple[int, int]]:
    grid = np.asarray(grid, dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError(f"ARC grid must be rank-2, got {grid.shape}")
    h, w = grid.shape
    if h > max_size or w > max_size:
        raise ValueError(f"grid {grid.shape} exceeds configured {max_size}x{max_size}")
    out = np.zeros((max_size, max_size), dtype=np.int64)
    out[:h, :w] = grid
    return out, (h, w)


@dataclass
class ARCBatch:
    demos_x: torch.Tensor          # [B,D,30,30]
    demos_y: torch.Tensor          # [B,D,30,30]
    demos_x_shapes: torch.Tensor   # [B,D,2]
    demos_y_shapes: torch.Tensor   # [B,D,2]
    demo_mask: torch.Tensor        # [B,D]
    query_x: torch.Tensor          # [B,30,30]
    query_shape: torch.Tensor      # [B,2]
    target_y: torch.Tensor         # [B,30,30]
    target_shape: torch.Tensor     # [B,2]
    task_ids: List[str]


class ARCMetaDataset:
    """In-memory ARC task sampler for leave-one-example-out meta-learning.

    Each training episode chooses one task, withholds one training pair as the
    query/target, and uses the remaining pairs as demonstrations.  This matches
    ARC inference much better than treating every (input,output) pair as an
    unrelated supervised example.
    """

    def __init__(
        self,
        split: str = "train",
        max_size: int = 30,
        max_demos: int = 4,
        seed: int = 0,
        limit_tasks: int | None = None,
        tasks: Sequence[Any] | None = None,
    ):
        self.max_size = max_size
        self.max_demos = max_demos
        self.rng = random.Random(seed)
        self.tasks = list(tasks) if tasks is not None else _tasks_from_split(split)
        if limit_tasks is not None:
            self.tasks = self.tasks[: int(limit_tasks)]
        # ARC training tasks normally have >=2 examples. Keep a graceful fallback.
        self.eligible = [t for t in self.tasks if len(getattr(t, "train")) >= 1]
        if not self.eligible:
            raise RuntimeError("no ARC tasks with training examples found")

    @property
    def task_count(self) -> int:
        return len(self.eligible)

    def split_train_validation(self, fraction: float = 0.1, seed: int = 0):
        idx = list(range(len(self.eligible)))
        random.Random(seed).shuffle(idx)
        n_val = max(1, int(round(len(idx) * fraction))) if len(idx) > 1 else 0
        val_idx = set(idx[:n_val])
        train_tasks = [t for i, t in enumerate(self.eligible) if i not in val_idx]
        val_tasks = [t for i, t in enumerate(self.eligible) if i in val_idx]
        if not train_tasks:
            train_tasks = val_tasks
        return (
            ARCMetaDataset(max_size=self.max_size, max_demos=self.max_demos, seed=seed, tasks=train_tasks),
            ARCMetaDataset(max_size=self.max_size, max_demos=self.max_demos, seed=seed + 1, tasks=val_tasks or train_tasks),
        )

    def _episode(self, task):
        pairs = list(task.train)
        q_idx = self.rng.randrange(len(pairs))
        qx, qy = _pair_arrays(pairs[q_idx])
        if qy is None:
            raise RuntimeError("training ARC pair has no output")
        demo_pairs = [p for i, p in enumerate(pairs) if i != q_idx]
        # Single-example fallback: use the same pair as context. This is rare but
        # avoids making the data loader brittle.
        if not demo_pairs:
            demo_pairs = [pairs[q_idx]]
        self.rng.shuffle(demo_pairs)
        demo_pairs = demo_pairs[: self.max_demos]
        return demo_pairs, qx, qy

    def sample_batch(self, batch_size: int, device) -> ARCBatch:
        B, D, S = batch_size, self.max_demos, self.max_size
        dx = np.zeros((B, D, S, S), dtype=np.int64)
        dy = np.zeros((B, D, S, S), dtype=np.int64)
        dxs = np.ones((B, D, 2), dtype=np.int64)
        dys = np.ones((B, D, 2), dtype=np.int64)
        dm = np.zeros((B, D), dtype=np.bool_)
        q = np.zeros((B, S, S), dtype=np.int64)
        y = np.zeros((B, S, S), dtype=np.int64)
        qs = np.ones((B, 2), dtype=np.int64)
        ys = np.ones((B, 2), dtype=np.int64)
        task_ids: List[str] = []

        for b in range(B):
            task = self.rng.choice(self.eligible)
            demos, qx, qy = self._episode(task)
            for d, pair in enumerate(demos):
                x0, y0 = _pair_arrays(pair)
                if y0 is None:
                    continue
                dx[b, d], dxs[b, d] = pad_grid(x0, S)
                dy[b, d], dys[b, d] = pad_grid(y0, S)
                dm[b, d] = True
            q[b], qs[b] = pad_grid(qx, S)
            y[b], ys[b] = pad_grid(qy, S)
            task_ids.append(str(getattr(task, "id", b)))

        def t(a, dtype=None):
            return torch.as_tensor(a, device=device, dtype=dtype)

        return ARCBatch(
            t(dx, torch.long), t(dy, torch.long),
            t(dxs, torch.long), t(dys, torch.long), t(dm, torch.bool),
            t(q, torch.long), t(qs, torch.long), t(y, torch.long), t(ys, torch.long),
            task_ids,
        )

    def sample_grid_batch(self, batch_size: int, device):
        """Random individual grids for codec autoencoding."""
        grids, shapes = [], []
        for _ in range(batch_size):
            task = self.rng.choice(self.eligible)
            pair = self.rng.choice(list(task.train))
            x, y = _pair_arrays(pair)
            grid = x if (y is None or self.rng.random() < 0.5) else y
            pg, sh = pad_grid(grid, self.max_size)
            grids.append(pg)
            shapes.append(sh)
        return (
            torch.as_tensor(np.stack(grids), device=device, dtype=torch.long),
            torch.as_tensor(np.asarray(shapes), device=device, dtype=torch.long),
        )

    def evaluation_episodes(self, limit: int | None = None):
        """Yield (task_id, demos, query_x, target_y) from task.test when labels exist.

        If a task's test output is unavailable, fall back to one leave-one-out
        train episode so the evaluator still works with hidden-label datasets.
        """
        tasks = self.eligible[: limit or len(self.eligible)]
        for task in tasks:
            demos = list(task.train)[: self.max_demos]
            test_pairs = list(getattr(task, "test", []))
            emitted = False
            for pair in test_pairs:
                x, y = _pair_arrays(pair)
                if y is not None:
                    emitted = True
                    yield str(getattr(task, "id", "unknown")), demos, x, y
            if not emitted:
                demo_pairs, qx, qy = self._episode(task)
                yield str(getattr(task, "id", "unknown")), demo_pairs, qx, qy
