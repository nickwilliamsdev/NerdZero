from __future__ import annotations

ARC_NEAT_PATCH_ID = "arc-scratch-v3-innovation-fix"
ARC_NEAT_USES_EXTERNAL_SEED = False

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
from sefer.evolution.evolve_transport import require_pytorch_neat

try:
    from sefer.evolution.neat_runtime import NEAT_INPUT_NAMES
except Exception:
    NEAT_INPUT_NAMES = None


def _load_neat_config(cfg):
    """Load only the explicit ARC-local neat-python configuration."""
    import neat
    path = Path(cfg.arc_neat_config_path or "arc_neat_config.ini").expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ARC-NEAT config not found: {path}")
    neat_cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )
    if NEAT_INPUT_NAMES is not None:
        expected = len(NEAT_INPUT_NAMES)
        if neat_cfg.genome_config.num_inputs != expected:
            raise ValueError(
                f"CPPN input mismatch: config has {neat_cfg.genome_config.num_inputs}, "
                f"runtime expects {expected}"
            )
    neat_cfg.pop_size = max(2, int(cfg.arc_neat_population))
    print(f"ARC-NEAT config: {path}")
    return neat_cfg, path


def _is_neat_genome(obj) -> bool:
    return (
        obj is not None
        and hasattr(obj, "nodes")
        and hasattr(obj, "connections")
        and hasattr(obj, "mutate")
    )


def _unwrap_seed_genome(obj):
    if _is_neat_genome(obj):
        return obj, "root"
    if isinstance(obj, dict):
        for key in ("winner", "genome", "seed_genome"):
            if key in obj and _is_neat_genome(obj[key]):
                return obj[key], key
        for key, value in obj.items():
            if _is_neat_genome(value):
                return value, str(key)
    return None, None


def initialize_arc_cppn_from_scratch(model, cfg):
    """Create a brand-new ARC-owned CPPN using neat-python's normal population bootstrap.

    Recent neat-python versions require the innovation tracker to be initialized
    before initial connections are created. Constructing a Population performs
    that initialization correctly; we then take one freshly generated genome as
    the ARC-owned seed.
    """
    import neat

    neat_cfg, _ = _load_neat_config(cfg)

    # IMPORTANT: do not call DefaultGenome.configure_new() directly here.
    # neat.Population initializes the innovation tracker used by connection genes.
    bootstrap = neat.Population(neat_cfg)
    if not bootstrap.population:
        raise RuntimeError("neat-python created an empty bootstrap population")

    first_key = sorted(bootstrap.population.keys())[0]
    genome = copy.deepcopy(bootstrap.population[first_key])
    genome.key = 0
    genome.fitness = None

    _install_genome(model, genome, neat_cfg)

    path = Path(cfg.arc_neat_initial_seed_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {"winner": genome, "source": ARC_NEAT_PATCH_ID},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"ARC initialized new CPPN: {path.resolve()} "
        f"nodes={len(genome.nodes)} connections={len(genome.connections)}"
    )
    return genome


def restore_arc_cppn(model, cfg, *, winner_path=None):
    """Restore this ARC project's own CPPN so a saved ARC checkpoint is reconstructible."""
    path = Path(winner_path or cfg.arc_neat_initial_seed_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"ARC-owned CPPN seed not found: {path}. "
            "Train from scratch first or restore the seed saved with this ARC run."
        )
    with path.open("rb") as f:
        genome, _ = _unwrap_seed_genome(pickle.load(f))
    if genome is None:
        raise ValueError(f"Invalid ARC-owned CPPN seed: {path}")
    neat_cfg, _ = _load_neat_config(cfg)
    _install_genome(model, genome, neat_cfg)
    return genome


def _install_genome(model, genome, neat_cfg):
    """Install an in-memory neat-python genome directly into the frozen core.

    Candidate genomes already exist in memory and share the active neat.Config.
    Routing them through the legacy legacy pickle loader is both unnecessary and
    fragile because that loader expects a saved payload containing both
    ``winner`` and ``config_text``.
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


def _split_neat_datasets(val_data, cfg):
    """Create deterministic, disjoint fitness and holdout ARC task sets."""
    tasks = list(getattr(val_data, "eligible", []))
    if not tasks:
        raise RuntimeError("ARC-NEAT validation dataset contains no eligible tasks")

    rng = random.Random(int(getattr(cfg, "seed", 0)) + 1909)
    rng.shuffle(tasks)

    requested_fit = max(int(getattr(cfg, "arc_neat_eval_tasks", 50)), 1)
    requested_holdout = max(int(getattr(cfg, "arc_neat_holdout_tasks", 50)), 0)

    if len(tasks) >= requested_fit + requested_holdout:
        n_fit = requested_fit
        n_holdout = requested_holdout
    else:
        # Preserve a genuinely disjoint holdout whenever there is more than one task.
        n_fit = min(requested_fit, max(1, len(tasks) // 2))
        n_holdout = min(requested_holdout, len(tasks) - n_fit)

    fit_tasks = tasks[:n_fit]
    holdout_tasks = tasks[n_fit:n_fit + n_holdout]

    fitness_data = ARCMetaDataset(
        max_size=val_data.max_size, max_demos=val_data.max_demos,
        seed=int(getattr(cfg, "seed", 0)) + 1910, tasks=fit_tasks,
    )
    holdout_data = None
    if holdout_tasks:
        holdout_data = ARCMetaDataset(
            max_size=val_data.max_size, max_demos=val_data.max_demos,
            seed=int(getattr(cfg, "seed", 0)) + 1911, tasks=holdout_tasks,
        )

    fit_ids = {str(getattr(t, "id", id(t))) for t in fit_tasks}
    hold_ids = {str(getattr(t, "id", id(t))) for t in holdout_tasks}
    overlap = fit_ids.intersection(hold_ids)
    if overlap:
        raise RuntimeError(f"ARC-NEAT fitness/holdout task leakage detected: {sorted(overlap)[:5]}")

    print(
        f"ARC-NEAT task split: fitness={len(fit_tasks)} holdout={len(holdout_tasks)} "
        f"totalVal={len(tasks)} seed={int(getattr(cfg, 'seed', 0)) + 1909}"
    )
    return fitness_data, holdout_data


def _fixed_meta_batches(data, cfg, device):
    batches = []
    for _ in range(max(int(cfg.arc_neat_eval_batches), 1)):
        batches.append(data.sample_batch(min(int(cfg.batch_size), 64), device))
    return batches


@torch.no_grad()
def _one_step_improvement(model, batches) -> float:
    vals = []
    for batch in batches:
        rule = model.encode_rule(
            batch.demos_x, batch.demos_y,
            batch.demos_x_shapes, batch.demos_y_shapes, batch.demo_mask,
        )
        Hq = model.encode_grid(batch.query_x, batch.query_shape)
        rule = model.condition_rule_on_query(rule, Hq)
        Ht = model.encode_grid(batch.target_y, batch.target_shape)
        all_states = model.operator_bank.apply_all(Hq, rule)
        best = (all_states - Ht[:, None]).pow(2).mean(dim=(2, 3)).min(dim=1).values
        base = (Hq - Ht).pow(2).mean(dim=(1, 2))
        vals.append((best < base).float().mean())
    return float(torch.stack(vals).mean()) if vals else 0.0


def _genome_complexity(genome) -> float:
    nodes = len(getattr(genome, "nodes", {}))
    conns = getattr(genome, "connections", {})
    enabled = 0
    for c in conns.values():
        if getattr(c, "enabled", True):
            enabled += 1
    return float(nodes + enabled)


def _metric_triplet(metrics, improve: float):
    return (
        float(metrics.get("pixel_acc", 0.0)),
        float(metrics.get("search_demo_fit", 0.0)),
        float(improve),
    )


def _relative_fitness(metrics, improve: float, complexity: float, baseline, cfg) -> tuple[float, dict]:
    """Score a genome by ARC-facing improvement over the original legacy seed.

    Query pixel transfer and demonstration consistency drive selection. The
    one-step operator-improvement metric is retained in ``deltas`` for
    diagnostics, but ARCConfig sets its fitness weight to zero in v1.9.1.
    """
    pixel, demo, imp = _metric_triplet(metrics, improve)
    bpixel, bdemo, bimp = _metric_triplet(baseline["metrics"], baseline["improve"])
    d_pixel = pixel - bpixel
    d_demo = demo - bdemo
    d_improve = imp - bimp
    d_complexity = float(complexity) - float(baseline["complexity"])

    fitness = (
        float(cfg.arc_neat_query_weight) * d_pixel
        + float(cfg.arc_neat_demo_fit_weight) * d_demo
        + float(cfg.arc_neat_improve_weight) * d_improve
        - float(cfg.arc_neat_complexity_weight) * d_complexity
    )
    deltas = {
        "pixel": d_pixel, "demo": d_demo, "improve": d_improve,
        "complexity": d_complexity,
    }
    return float(fitness), deltas


def _seed_population(population, seed_genome, config, mutations: int):
    """Seed population from the ARC-owned CPPN while keeping NEAT node IDs monotonic."""
    if not _is_neat_genome(seed_genome):
        raise TypeError(
            "ARC-NEAT seed is not a neat-python genome: "
            f"type={type(seed_genome).__name__}"
        )

    existing_ids = list(getattr(seed_genome, "nodes", {}).keys())
    for g in population.population.values():
        existing_ids.extend(getattr(g, "nodes", {}).keys())
    next_node = (max(existing_ids) + 1) if existing_ids else 0
    config.genome_config.node_indexer = itertools.count(next_node)

    keys = list(population.population.keys())
    for i, key in enumerate(keys):
        g = copy.deepcopy(seed_genome)
        g.key = key
        if i > 0:
            for _ in range(max(int(mutations), 1)):
                g.mutate(config.genome_config)
        g.fitness = None
        population.population[key] = g
    population.species.speciate(config, population.population, population.generation)


def evolve_arc_cppn(model, val_data, cfg, device):
    """Baldwinian NEAT outer loop over ARC-owned CPPN geometry."""
    if not getattr(cfg, "arc_neat_enabled", False) or int(cfg.arc_neat_generations) <= 0:
        return None
    if val_data is None:
        print("ARC-NEAT skipped: no validation/meta-evaluation split available")
        return None

    try:
        import neat
    except Exception as exc:
        raise RuntimeError("ARC-NEAT requires neat-python in the active environment") from exc

    neat_cfg, neat_cfg_path = _load_neat_config(cfg)

    seed_path = Path(cfg.arc_neat_initial_seed_path).expanduser()
    if not seed_path.is_file():
        raise FileNotFoundError(
            f"ARC-NEAT seed missing: {seed_path}. "
            "Run normal ARC scratch training before --arc-neat-only."
        )
    with seed_path.open("rb") as f:
        seed_genome, seed_key = _unwrap_seed_genome(pickle.load(f))
    if seed_genome is None:
        raise RuntimeError(f"ARC-NEAT could not recover a genome from {seed_path}")

    print(
        "ARC-NEAT recovered ARC-owned seed: "
        f"path={seed_path.resolve()} key={seed_key} "
        f"nodes={len(getattr(seed_genome, 'nodes', {}))} "
        f"connections={len(getattr(seed_genome, 'connections', {}))}"
    )

    # Keep this first experiment intentionally small. Population size is a
    # runtime override so the ARC-local config does not force 256 candidates.
    neat_cfg.pop_size = max(int(cfg.arc_neat_population), 2)
    population = neat.Population(neat_cfg)
    _seed_population(population, seed_genome, neat_cfg, cfg.arc_neat_seed_mutations)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    fitness_data, holdout_data = _split_neat_datasets(val_data, cfg)
    fixed_batches = _fixed_meta_batches(fitness_data, cfg, device)
    model.eval()

    # Establish the initial ARC CPPN as a zero-point. Every candidate is scored
    # on exactly the same fitness tasks and fixed one-step batches.
    _install_genome(model, seed_genome, neat_cfg)
    seed_improve = _one_step_improvement(model, fixed_batches)
    seed_metrics = evaluate_arc(model, fitness_data, limit=fitness_data.task_count, device=device)
    seed_complexity = _genome_complexity(seed_genome)
    baseline = {
        "metrics": dict(seed_metrics),
        "improve": float(seed_improve),
        "complexity": float(seed_complexity),
    }
    print(
        f"ARC-NEAT seed baseline fitness=0.0000 "
        f"pixel={seed_metrics.get('pixel_acc', 0.0):.3f} "
        f"demoFit={seed_metrics.get('search_demo_fit', 0.0):.3f} "
        f"improve={seed_improve:.3f} complexity={seed_complexity:.0f}"
    )
    print(
        "ARC-NEAT objective: "
        f"{float(cfg.arc_neat_query_weight):.2f}*dPixel + "
        f"{float(cfg.arc_neat_demo_fit_weight):.2f}*dDemo + "
        f"{float(cfg.arc_neat_improve_weight):.2f}*dImprove - "
        f"{float(cfg.arc_neat_complexity_weight):.6f}*dComplexity"
    )

    best_seen = {"fitness": float("-inf"), "metrics": None, "key": None}
    generation_counter = {"value": 0}

    with tempfile.TemporaryDirectory(prefix="arc_neat_") as td:
        temp_dir = Path(td)

        def evaluate_genomes(genomes: Iterable, config):
            gen = generation_counter["value"]
            print(f"ARC-NEAT generation {gen + 1}/{cfg.arc_neat_generations} candidates={len(genomes)}")
            for idx, (gid, genome) in enumerate(genomes):
                # The existing loader replaces the CPPN used by the frozen core.
                # No gradient step occurs, and ARC trainable parameters are untouched.
                try:
                    _install_genome(model, genome, neat_cfg)
                    improve = _one_step_improvement(model, fixed_batches)
                    metrics = evaluate_arc(
                        model, fitness_data, limit=fitness_data.task_count, device=device
                    )
                    complexity = _genome_complexity(genome)
                    fitness, deltas = _relative_fitness(
                        metrics, improve, complexity, baseline, cfg
                    )
                    genome._arc_neat_eval_ok = True
                except Exception as exc:
                    fitness = -1.0
                    metrics = {"pixel_acc": 0.0, "search_demo_fit": 0.0}
                    improve = 0.0
                    complexity = _genome_complexity(genome)
                    deltas = {"pixel": 0.0, "demo": 0.0, "improve": 0.0, "complexity": 0.0}
                    genome._arc_neat_eval_ok = False
                    print(f"ARC-NEAT candidate {gid} failed: {type(exc).__name__}: {exc}")
                genome.fitness = float(fitness)
                if fitness > best_seen["fitness"]:
                    best_seen.update(fitness=float(fitness), metrics=dict(metrics), key=gid)
                print(
                    f"  cand={idx + 1:02d}/{len(genomes):02d} key={gid} fit={fitness:.4f} "
                    f"pixel={metrics.get('pixel_acc', 0.0):.3f} "
                    f"demoFit={metrics.get('search_demo_fit', 0.0):.3f} "
                    f"improve={improve:.3f} complexity={complexity:.0f} "
                    f"dPixel={deltas['pixel']:+.3f} dDemo={deltas['demo']:+.3f} "
                    f"dImprove={deltas['improve']:+.3f}"
                )
            successful = [
                genome for _, genome in genomes
                if getattr(genome, "_arc_neat_eval_ok", False)
            ]
            if not successful:
                raise RuntimeError(
                    "ARC-NEAT: every candidate in this generation failed evaluation. "
                    "Aborting instead of evolving or saving a failure-only population."
                )
            generation_counter["value"] += 1

        winner = population.run(evaluate_genomes, int(cfg.arc_neat_generations))
        if (winner is None or winner.fitness is None
                or not getattr(winner, "_arc_neat_eval_ok", False)):
            raise RuntimeError(
                "ARC-NEAT did not produce a successfully evaluated winner; refusing to save it."
            )

        winner_path = Path(cfg.arc_neat_winner_path)
        winner_path.parent.mkdir(parents=True, exist_ok=True)
        config_text = Path(neat_cfg_path).read_text()
        with winner_path.open("wb") as f:
            pickle.dump(
                {
                    "winner": winner,
                    "config_text": config_text,
                    "source": ARC_NEAT_PATCH_ID,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"ARC-NEAT saved winner: {winner_path} fitness={winner.fitness:.4f}")

        # Install the winning geometry while preserving all trained ARC weights.
        _install_genome(model, winner, neat_cfg)
        fitness_final = evaluate_arc(
            model, fitness_data, limit=fitness_data.task_count, device=device
        )
        if holdout_data is not None:
            holdout_final = evaluate_arc(
                model, holdout_data, limit=holdout_data.task_count, device=device
            )
        else:
            holdout_final = {}

        print(
            f"ARC-NEAT winner fitness-split pixel={fitness_final.get('pixel_acc', 0.0):.3f} "
            f"greedyPixel={fitness_final.get('greedy_pixel_acc', 0.0):.3f} "
            f"demoFit={fitness_final.get('search_demo_fit', 0.0):.3f} "
            f"directPixel={fitness_final.get('direct_pixel_acc', 0.0):.3f}"
        )
        if holdout_final:
            print(
                f"ARC-NEAT winner HOLDOUT pixel={holdout_final.get('pixel_acc', 0.0):.3f} "
                f"greedyPixel={holdout_final.get('greedy_pixel_acc', 0.0):.3f} "
                f"demoFit={holdout_final.get('search_demo_fit', 0.0):.3f} "
                f"directPixel={holdout_final.get('direct_pixel_acc', 0.0):.3f}"
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": vars(cfg),
                "metrics": holdout_final or fitness_final,
                "fitness_split_metrics": fitness_final,
                "holdout_metrics": holdout_final,
                "arc_neat_seed_baseline": baseline,
                "arc_neat_winner_path": str(winner_path),
                "arc_neat_fitness": float(winner.fitness),
            },
            cfg.arc_neat_checkpoint_path,
        )
        print(f"ARC-NEAT saved evolved ARC checkpoint: {cfg.arc_neat_checkpoint_path}")
        return {
            "winner": winner,
            "metrics": holdout_final or fitness_final,
            "fitness_metrics": fitness_final,
            "holdout_metrics": holdout_final,
            "fitness": float(winner.fitness),
        }
