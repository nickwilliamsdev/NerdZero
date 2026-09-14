from __future__ import annotations

from yetirah_arc_v1_training.yetirah_arc_v1.sefer.config import ARCConfig
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.tasks.arc_dataset import ARCMetaDataset
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.training.arc_trainer import train_arc_v1


def run(cfg: ARCConfig | None = None):
    cfg = cfg or ARCConfig()
    data = ARCMetaDataset(
        split=cfg.split,
        max_size=cfg.max_grid_size,
        max_demos=cfg.max_demos,
        seed=cfg.seed,
        limit_tasks=cfg.limit_tasks,
    )
    train_data, val_data = data.split_train_validation(cfg.validation_fraction, cfg.seed)
    print(f"ARC-v1 task split: train={train_data.task_count} val={val_data.task_count}")
    return train_arc_v1(cfg, train_data, val_data)


if __name__ == "__main__":
    run()
