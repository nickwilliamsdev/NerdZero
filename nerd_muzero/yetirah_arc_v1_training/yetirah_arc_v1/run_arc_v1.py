from __future__ import annotations

import argparse

from sefer.config import ARCConfig
from sefer.experiments.arc_v1 import run


def main():
    p = argparse.ArgumentParser(description="Train Yetirah ARC-v1.1 on arckit tasks")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--codec-steps", type=int, default=None)
    p.add_argument("--direct-steps", type=int, default=None)
    p.add_argument("--program-steps", type=int, default=None)
    p.add_argument("--limit-tasks", type=int, default=None)
    p.add_argument("--eval-tasks", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--no-v30-init",
        action="store_true",
        help="Intentionally disable v30 NEAT/algebra initialization (bootstrap ARC core).",
    )
    p.add_argument("--neat-winner", type=str, default=None)
    p.add_argument("--algebra-checkpoint", type=str, default=None)
    p.add_argument("--checkpoint-search-depth", type=int, default=None)
    args = p.parse_args()

    cfg = ARCConfig()
    for name in (
        "batch_size", "codec_steps", "direct_steps", "program_steps",
        "limit_tasks", "eval_tasks", "device", "checkpoint_search_depth",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.no_v30_init:
        cfg.initialize_from_v30 = False
        cfg.require_v30_init = False
    if args.neat_winner:
        cfg.neat_winner_path = args.neat_winner
    if args.algebra_checkpoint:
        cfg.algebra_checkpoint_path = args.algebra_checkpoint
    run(cfg)


if __name__ == "__main__":
    main()
