from __future__ import annotations

import argparse
import inspect
from pathlib import Path


import sefer.config as _arc_config_module
from sefer.config import ARCConfig
from sefer.experiments.arc_v1 import run
import sefer.training.arc_trainer as _arc_trainer_module
import sefer.controllers.arc_reasoner as _arc_reasoner_module
import sefer.evaluation.arc as _arc_eval_module


def main():
    p = argparse.ArgumentParser(description="Train Yetirah ARC-v1 on arckit tasks")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--codec-steps", type=int, default=None)
    p.add_argument("--direct-steps", type=int, default=None)
    p.add_argument("--operator-discovery-steps", type=int, default=None)
    p.add_argument("--program-steps", type=int, default=None)
    p.add_argument("--limit-tasks", type=int, default=None)
    p.add_argument("--eval-tasks", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-v30-init", action="store_true")
    p.add_argument("--neat-winner", type=str, default=None)
    p.add_argument("--algebra-checkpoint", type=str, default=None)
    p.add_argument("--no-arc-neat", action="store_true")
    p.add_argument("--arc-neat-generations", type=int, default=None)
    p.add_argument("--arc-neat-population", type=int, default=None)
    p.add_argument("--arc-neat-eval-tasks", type=int, default=None)
    p.add_argument("--arc-neat-config", type=str, default=None)
    args = p.parse_args()

    expected_patch = "v1.8-arc-neat-outer-loop"
    actual_patch = getattr(_arc_config_module, "ARC_PATCH_ID", None)
    trainer_patch = getattr(_arc_trainer_module, "ARC_TRAINER_PATCH_ID", None)
    reasoner_patch = getattr(_arc_reasoner_module, "ARC_REASONER_PATCH_ID", None)
    eval_patch = getattr(_arc_eval_module, "ARC_EVAL_PATCH_ID", None)
    print(f"ARC launcher expected patch: {expected_patch}")
    print(f"ARC config source: {Path(inspect.getfile(_arc_config_module)).resolve()} patch={actual_patch}")
    print(f"ARC trainer source: {Path(inspect.getfile(_arc_trainer_module)).resolve()} patch={trainer_patch}")
    print(f"ARC reasoner source: {Path(inspect.getfile(_arc_reasoner_module)).resolve()} patch={reasoner_patch}")
    print(f"ARC eval source: {Path(inspect.getfile(_arc_eval_module)).resolve()} patch={eval_patch}")
    if actual_patch != expected_patch or trainer_patch != "v1.8-arc-neat-outer-loop" or reasoner_patch != "v1.8-refreshable-cppn-base" or eval_patch != "v1.7-demo-consistency-search":
        raise RuntimeError(
            "ARC-v1.8 patch verification failed. Python is importing one or more old files. "
            "Replace the files listed in the patch and run this launcher from the ARC project root."
        )

    cfg = ARCConfig()
    for name in (
        "batch_size", "codec_steps", "direct_steps", "operator_discovery_steps",
        "program_steps", "limit_tasks", "eval_tasks", "device"
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.no_v30_init:
        cfg.initialize_from_v30 = False
    if args.neat_winner:
        cfg.neat_winner_path = args.neat_winner
    if args.algebra_checkpoint:
        cfg.algebra_checkpoint_path = args.algebra_checkpoint
    if args.no_arc_neat:
        cfg.arc_neat_enabled = False
    if args.arc_neat_generations is not None:
        cfg.arc_neat_generations = args.arc_neat_generations
    if args.arc_neat_population is not None:
        cfg.arc_neat_population = args.arc_neat_population
    if args.arc_neat_eval_tasks is not None:
        cfg.arc_neat_eval_tasks = args.arc_neat_eval_tasks
    if args.arc_neat_config:
        cfg.arc_neat_config_path = args.arc_neat_config
    run(cfg)


if __name__ == "__main__":
    main()
