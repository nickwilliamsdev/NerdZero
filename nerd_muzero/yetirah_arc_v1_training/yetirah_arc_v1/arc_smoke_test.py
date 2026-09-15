from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from sefer.config import ARCConfig
from sefer.tasks.arc_dataset import ARCMetaDataset
from sefer.training.arc_trainer import train_arc_v1


@dataclass
class FakeTask:
    id: str
    train: list
    test: list


def make_tasks():
    rng = np.random.default_rng(0)
    tasks = []
    for tid in range(4):
        pairs = []
        for _ in range(3):
            h, w = 4 + tid % 2, 5
            x = rng.integers(0, 4, size=(h, w), dtype=np.int64)
            # Simple task-local transformation: identity for even, horizontal flip for odd.
            y = x.copy() if tid % 2 == 0 else np.fliplr(x).copy()
            pairs.append((x, y))
        tx = rng.integers(0, 4, size=(4 + tid % 2, 5), dtype=np.int64)
        ty = tx.copy() if tid % 2 == 0 else np.fliplr(tx).copy()
        tasks.append(FakeTask(f"fake-{tid}", pairs, [(tx, ty)]))
    return tasks


def main():
    cfg = ARCConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_grid_size=30,
        batch_size=2,
        codec_steps=1,
        direct_steps=1,
        program_steps=1,
        diagnostic_every=1,
        eval_every=0,
        eval_tasks=2,
        initialize_from_v30=False,
        arc_checkpoint_path="/tmp/yetirah_arc_v1_smoke.pt",
        arc_best_checkpoint_path="/tmp/yetirah_arc_v1_smoke_best.pt",
    )
    data = ARCMetaDataset(max_size=30, max_demos=2, tasks=make_tasks())
    train, val = data.split_train_validation(0.25, seed=0)
    model, metrics = train_arc_v1(cfg, train, val)
    print("ARC smoke metrics:", metrics)
    print("ARC smoke params:", sum(p.numel() for p in model.parameters()))
    print("ARC smoke test passed")


if __name__ == "__main__":
    main()
