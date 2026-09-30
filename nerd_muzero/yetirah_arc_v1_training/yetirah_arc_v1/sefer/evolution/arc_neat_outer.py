
    
  
from __future__ import annotations

ARC_NEAT_PATCH_ID = "arc-scratch-v1"

import copy
import itertools
import pickle
import random
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from sefer.evaluation.arc import evaluate_arc
from sefer.tasks.arc_dataset import ARCMetaDataset
from sefer.evolution.evolve_transport import require_pytorch_neat  # ARC-local runtime dependency; see README
try:
    from sefer.evolution.neat_runtime import NEAT_INPUT_NAMES
except Exception:
    NEAT_INPUT_NAMES = None


def _load_neat_config(cfg):
    """Load this ARC project's explicit NEAT INI, without searching old projects."""
    import neat
    path = Path(cfg.arc_neat_config_path or "arc_neat_config.ini").expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ARC NEAT config missing: {path}")
    neat_cfg = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path),
    )
    if NEAT_INPUT_NAMES is not None and neat_cfg.genome_config.num_inputs != len(NEAT_INPUT_NAMES):
        raise ValueError(
            f"CPPN input mismatch: INI defines {neat_cfg.genome_config.num_inputs}, "
            f"runtime expects {len(NEAT_INPUT_NAMES)}"
        )
    neat_cfg.pop_size = max(2, int(cfg.arc_neat_population))
    print(f"ARC-NEAT config: {path}")
    return neat_cfg, path


def initialize_arc_cppn_from_scratch(model, cfg):
    """Create and persist a new ARC-only CPPN before ARC training begins."""
    import neat
    neat_cfg, _ = _load_neat_config(cfg)
    genome = neat.DefaultGenome(0)
    genome.configure_new(neat_cfg.genome_config)
    _install_genome(model, genome, neat_cfg)
    target = Path(cfg.arc_neat_initial_seed_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump({"winner": genome, "source": ARC_NEAT_PATCH_ID}, stream)
    print(f"ARC initialized from a newly generated CPPN: {target.resolve()}")
    return genome


def restore_arc_cppn(model, cfg, *, winner_path=None):
    """Restore this project's CPPN genome needed to reconstruct an ARC checkpoint."""
    path = Path(winner_path or cfg.arc_neat_initial_seed_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"ARC-owned CPPN seed missing: {path}. "
            "Retrain from scratch or restore this project's seed alongside the checkpoint."
        )
    with path.open("rb") as stream:
        genome, _ = _unwrap_seed_genome(pickle.load(stream))
    if genome is None:
        raise ValueError(f"ARC-owned CPPN seed is invalid: {path}")
    neat_cfg, _ = _load_neat_config(cfg)
    _install_genome(model, genome, neat_cfg)
    return genome


def _install_genome(model, genome, neat_cfg):
    """Install an in-memory neat-python genome directly into the frozen core.

    Candidate genomes already exist in memory and share the active neat.Config.
    ARC CPPN genomes can be installed directly; no external checkpoint loader
    or inherited model is involved.
    """
    if not _is_neat_genome(genome):
        raise TypeError(
            "ARC-NEAT candidate is not a neat-python genome: "
            f"type={type(genome).__name__}"
        )

    _, create_cppn_fn = require_pytorch_neat()
    model.core.op_hyper.install_evolved_cppn(genome, neat_cfg, create_cppn_fn)
    model.core.operator_codes.requires_grad_(False)

    # The ARC fast-operator bank materializes the CPPN-derived base geometry.
    # Refresh only that base; learned static/task-conditioned fast weights remain.
    model.refresh_operator_base_from_core()

