from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import sefer.config as _arc_config_module
from sefer.config import ARCConfig
from sefer.experiments.arc_v1 import run
from sefer.tasks.arc_dataset import ARCMetaDataset
from sefer.training.arc_trainer import load_arc_v1_checkpoint
from sefer.evolution.arc_neat_outer import evolve_arc_cppn
import sefer.training.arc_trainer as _arc_trainer_module
import sefer.controllers.arc_reasoner as _arc_reasoner_module
import sefer.evaluation.arc as _arc_eval_module


# ---------------------------------------------------------------------------
# ARC component patch versions
# ---------------------------------------------------------------------------

EXPECTED_CONFIG_PATCH = "v1.9.1-arc-facing-neat-fitness"
EXPECTED_TRAINER_PATCH = "v1.8.1-resumable-arc-neat"
EXPECTED_REASONER_PATCH = "v1.8-refreshable-cppn-base"
EXPECTED_EVAL_PATCH = "v1.7-demo-consistency-search"


def _verify_arc_patches() -> None:
    """Verify that Python imported the expected ARC implementation files."""

    actual_config_patch = getattr(
        _arc_config_module,
        "ARC_PATCH_ID",
        None,
    )
    actual_trainer_patch = getattr(
        _arc_trainer_module,
        "ARC_TRAINER_PATCH_ID",
        None,
    )
    actual_reasoner_patch = getattr(
        _arc_reasoner_module,
        "ARC_REASONER_PATCH_ID",
        None,
    )
    actual_eval_patch = getattr(
        _arc_eval_module,
        "ARC_EVAL_PATCH_ID",
        None,
    )

    print("ARC patch verification:")
    print(
        f"  config   expected={EXPECTED_CONFIG_PATCH} "
        f"actual={actual_config_patch}"
    )
    print(
        f"           source="
        f"{Path(inspect.getfile(_arc_config_module)).resolve()}"
    )

    print(
        f"  trainer  expected={EXPECTED_TRAINER_PATCH} "
        f"actual={actual_trainer_patch}"
    )
    print(
        f"           source="
        f"{Path(inspect.getfile(_arc_trainer_module)).resolve()}"
    )

    print(
        f"  reasoner expected={EXPECTED_REASONER_PATCH} "
        f"actual={actual_reasoner_patch}"
    )
    print(
        f"           source="
        f"{Path(inspect.getfile(_arc_reasoner_module)).resolve()}"
    )

    print(
        f"  eval     expected={EXPECTED_EVAL_PATCH} "
        f"actual={actual_eval_patch}"
    )
    print(
        f"           source="
        f"{Path(inspect.getfile(_arc_eval_module)).resolve()}"
    )

    mismatches = []

    if actual_config_patch != EXPECTED_CONFIG_PATCH:
        mismatches.append(
            f"config: expected {EXPECTED_CONFIG_PATCH!r}, "
            f"got {actual_config_patch!r}"
        )

    if actual_trainer_patch != EXPECTED_TRAINER_PATCH:
        mismatches.append(
            f"trainer: expected {EXPECTED_TRAINER_PATCH!r}, "
            f"got {actual_trainer_patch!r}"
        )

    if actual_reasoner_patch != EXPECTED_REASONER_PATCH:
        mismatches.append(
            f"reasoner: expected {EXPECTED_REASONER_PATCH!r}, "
            f"got {actual_reasoner_patch!r}"
        )

    if actual_eval_patch != EXPECTED_EVAL_PATCH:
        mismatches.append(
            f"eval: expected {EXPECTED_EVAL_PATCH!r}, "
            f"got {actual_eval_patch!r}"
        )

    if mismatches:
        raise RuntimeError(
            "ARC patch verification failed.\n"
            "Python is importing one or more unexpected ARC components:\n  - "
            + "\n  - ".join(mismatches)
            + "\n\nRun this launcher from the ARC project root and make sure "
              "the expected patch files are installed."
        )

    print("ARC patch verification passed.")


def main():
    p = argparse.ArgumentParser(
        description="Train Yetirah ARC-v1 on arckit tasks"
    )

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

    p.add_argument(
        "--arc-neat-only",
        action="store_true",
        help="Run only the NEAT outer loop from an existing ARC checkpoint",
    )

    p.add_argument(
        "--arc-neat-resume",
        type=str,
        default=None,
        help=(
            "Checkpoint for --arc-neat-only; defaults to pre-NEAT, "
            "then best, then final"
        ),
    )

    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Verify ARC implementation before constructing/training anything.
    # -----------------------------------------------------------------------

    _verify_arc_patches()

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    cfg = ARCConfig()

    for name in (
        "batch_size",
        "codec_steps",
        "direct_steps",
        "operator_discovery_steps",
        "program_steps",
        "limit_tasks",
        "eval_tasks",
        "device",
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

    # -----------------------------------------------------------------------
    # ARC-NEAT-only mode
    # -----------------------------------------------------------------------

    if args.arc_neat_only:
        import torch

        device = torch.device(cfg.device)

        root = ARCMetaDataset(
            split=cfg.split,
            max_size=cfg.max_grid_size,
            max_demos=cfg.max_demos,
            seed=cfg.seed,
            limit_tasks=cfg.limit_tasks,
        )

        _, val_data = root.split_train_validation(
            cfg.validation_fraction,
            cfg.seed,
        )

        choices = []

        if args.arc_neat_resume:
            choices.append(Path(args.arc_neat_resume))

        choices += [
            Path(cfg.arc_pre_neat_checkpoint_path),
            Path(cfg.arc_best_checkpoint_path),
            Path(cfg.arc_checkpoint_path),
        ]

        ckpt = next(
            (candidate for candidate in choices if candidate.is_file()),
            None,
        )

        if ckpt is None:
            raise FileNotFoundError(
                "--arc-neat-only could not find an ARC checkpoint. Tried: "
                + ", ".join(str(candidate) for candidate in choices)
            )

        print(
            f"ARC-NEAT-only loading ARC checkpoint: "
            f"{ckpt.resolve()}"
        )

        model, _ = load_arc_v1_checkpoint(
            cfg,
            str(ckpt),
            device=device,
        )

        result = evolve_arc_cppn(
            model,
            val_data,
            cfg,
            device,
        )

        if result is not None:
            metrics = result["metrics"]

            print(
                "ARC-NEAT-only complete "
                f"fitness={result['fitness']:.4f} "
                f"pixel={metrics['pixel_acc']:.3f} "
                f"demoFit={metrics.get('search_demo_fit', 0.0):.3f}"
            )

        return

    # -----------------------------------------------------------------------
    # Normal ARC training
    # -----------------------------------------------------------------------

    run(cfg)


if __name__ == "__main__":
    main()