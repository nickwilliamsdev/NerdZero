from __future__ import annotations

import argparse

from yetirah_arc_v1_training.yetirah_arc_v1.sefer.config import ARCConfig
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.tasks.arc_dataset import ARCMetaDataset
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.training.arc_trainer import load_arc_v1_checkpoint
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.evaluation.arc import evaluate_arc


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained Yetirah ARC-v1 checkpoint")
    p.add_argument("--checkpoint", default="yetirah_arc_v1.pt")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = ARCConfig(split=args.split)
    if args.device:
        cfg.device = args.device
    model, payload = load_arc_v1_checkpoint(cfg, args.checkpoint, cfg.device)
    data = ARCMetaDataset(
        split=args.split,
        max_size=cfg.max_grid_size,
        max_demos=cfg.max_demos,
        seed=cfg.seed,
    )
    metrics = evaluate_arc(model, data, limit=args.limit, device=cfg.device)
    print(metrics)


if __name__ == "__main__":
    main()
