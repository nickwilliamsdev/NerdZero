from __future__ import annotations

ARC_NEAT_PATCH_ID = "v1.9.1-arc-facing-fitness"

import copy
import configparser
import pickle
import random
import re
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


def _looks_like_neat_config(text: str) -> bool:
    return '[NEAT]' in text and '[DefaultGenome]' in text


def _is_plain_neat_config(text: str) -> bool:
    if not _looks_like_neat_config(text):
        return False
    first_meaningful = None
    for raw in text.lstrip('\ufeff').splitlines():
        line = raw.strip()
        if not line or line.startswith(('#', ';')):
            continue
        first_meaningful = line
        break
    if first_meaningful is None or not first_meaningful.startswith('['):
        return False
    parser = configparser.ConfigParser()
    try:
        parser.read_string(text)
    except configparser.Error:
        return False
    return parser.has_section('NEAT') and parser.has_section('DefaultGenome')


def _extract_embedded_neat_config(text: str) -> str | None:
    if not _looks_like_neat_config(text):
        return None
    if _is_plain_neat_config(text):
        return text.strip() + '\n'
    start = text.find('[NEAT]')
    block = text[start:]
    # Search for the longest parseable INI prefix beginning at [NEAT].
    lines = block.splitlines()
    for stop in range(len(lines), 1, -1):
        candidate = '\n'.join(lines[:stop]).strip() + '\n'
        if _is_plain_neat_config(candidate):
            return candidate
    return None


def _render_neat_template(path: Path, cfg) -> Path:
    """Render placeholders left by v30's Python-generated NEAT config.

    The original v30 helper neat_config_text(num_inputs, pop_size, seed) used
    an f-string. Extracting its literal INI block therefore leaves placeholders
    such as {num_inputs}, {pop_size}, and {seed}. Resolve them here before
    neat-python parses the file.
    """
    text = path.read_text(errors="ignore")
    if "{" not in text:
        return path

    if NEAT_INPUT_NAMES is not None:
        num_inputs = len(NEAT_INPUT_NAMES)
    else:
        # v30's evolved CPPN feature vector is fixed at 40 inputs:
        # 5 dst + 5 src + 5 diff + 5 prod + 12 relational/phase + 8 op code.
        num_inputs = 40

    values = {
        "pop_size": max(int(getattr(cfg, "arc_neat_population", 16)), 2),
        "num_inputs": int(num_inputs),
        "seed": int(getattr(cfg, "seed", 0)),
    }
    rendered = text
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", str(value))

    unresolved = sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rendered)))
    if unresolved:
        raise ValueError(
            "ARC-NEAT extracted config still contains unresolved template fields: "
            + ", ".join(unresolved)
            + ". Pass --arc-neat-config pointing to a fully rendered neat-python INI "
              "or extend the template mapping."
        )

    parser = configparser.ConfigParser()
    parser.read_string(rendered)
    if not parser.has_section("NEAT") or not parser.has_section("DefaultGenome"):
        raise ValueError(f"ARC-NEAT rendered config is missing required sections: {path}")
    # Catch the exact failure from v1.8.2 before neat-python gets involved.
    parser.getint("NEAT", "pop_size")
    parser.getint("DefaultGenome", "num_inputs")

    out = path.parent / ".arc_neat_rendered_config.ini"
    out.write_text(rendered)
    print(
        f"ARC-NEAT rendered config placeholders: "
        f"num_inputs={values['num_inputs']} pop_size={values['pop_size']} seed={values['seed']}"
    )
    print(f"ARC-NEAT rendered config: {out.resolve()}")
    return out.resolve()

def _candidate_roots(winner: Path) -> list[Path]:
    roots = [winner.parent, Path.cwd(), Path(__file__).resolve().parents[3]]
    roots.extend(list(winner.parents)[:5])
    out, seen = [], set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen and root.exists():
            seen.add(key)
            out.append(root)
    return out


def _find_neat_config(requested: str | None, winner_path: str) -> Path:
    if requested:
        p = Path(requested).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'ARC-NEAT config not found: {p}')
        text = p.read_text(errors='ignore')
        if _is_plain_neat_config(text):
            return p
        block = _extract_embedded_neat_config(text)
        if block:
            generated = Path(winner_path).expanduser().resolve().parent / '.arc_neat_resolved_config.ini'
            try:
                generated.write_text(block)
            except OSError:
                generated = Path.cwd() / '.arc_neat_resolved_config.ini'
                generated.write_text(block)
            print(f'ARC-NEAT extracted embedded config from explicit source: {p}')
            print(f'ARC-NEAT generated config: {generated.resolve()}')
            return generated.resolve()
        raise ValueError(f'ARC-NEAT config/source contains no parseable neat-python config: {p}')

    winner = Path(winner_path).expanduser().resolve()
    roots = _candidate_roots(winner)
    exact_names = {
        'neat_config.ini', 'neat-config.ini', 'config-neat', 'config-feedforward',
        'neat_config.txt', 'config.txt', 'neat.cfg', 'neat.ini',
    }
    skip_dirs = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.cache'}

    candidates = []
    seen = set()
    for root in roots:
        try:
            for p in root.rglob('*'):
                if not p.is_file() or any(part in skip_dirs for part in p.parts):
                    continue
                name = p.name.lower()
                if name in exact_names or ('neat' in name and 'config' in name) or name.startswith('config-'):
                    key = str(p.resolve(strict=False))
                    if key not in seen:
                        seen.add(key)
                        candidates.append(p)
        except (OSError, PermissionError):
            continue

    for p in candidates:
        try:
            text = p.read_text(errors='ignore')
        except OSError:
            continue
        if _is_plain_neat_config(text):
            print(f'ARC-NEAT auto-located plain config: {p.resolve()}')
            return p.resolve()
        block = _extract_embedded_neat_config(text)
        if block:
            generated = winner.parent / '.arc_neat_resolved_config.ini'
            try:
                generated.write_text(block)
            except OSError:
                generated = Path.cwd() / '.arc_neat_resolved_config.ini'
                generated.write_text(block)
            print(f'ARC-NEAT extracted embedded config from: {p.resolve()}')
            print(f'ARC-NEAT generated config: {generated.resolve()}')
            return generated.resolve()

    source_exts = {'.py', '.txt', '.md', '.ini', '.cfg', '.conf'}
    source_seen = set()
    for root in roots:
        try:
            for p in root.rglob('*'):
                if not p.is_file() or p.suffix.lower() not in source_exts:
                    continue
                if any(part in skip_dirs for part in p.parts):
                    continue
                key = str(p.resolve(strict=False))
                if key in source_seen:
                    continue
                source_seen.add(key)
                try:
                    if p.stat().st_size > 2_000_000:
                        continue
                    text = p.read_text(errors='ignore')
                except OSError:
                    continue
                if not _looks_like_neat_config(text):
                    continue
                block = _extract_embedded_neat_config(text)
                if block:
                    generated = winner.parent / '.arc_neat_resolved_config.ini'
                    try:
                        generated.write_text(block)
                    except OSError:
                        generated = Path.cwd() / '.arc_neat_resolved_config.ini'
                        generated.write_text(block)
                    print(f'ARC-NEAT extracted embedded config from: {p}')
                    print(f'ARC-NEAT generated config: {generated.resolve()}')
                    return generated.resolve()
        except (OSError, PermissionError):
            continue

    searched = '\n  - '.join(str(r) for r in roots)
    raise FileNotFoundError(
        'Could not auto-locate or extract the neat-python config used by the v30 winner. '
        'Searched recursively under:\n  - ' + searched +
        '\nPass --arc-neat-config /path/to/config explicitly if the original config lives elsewhere.'
    )


def _install_genome(model, genome, neat_cfg):
    """Install an in-memory neat-python genome directly into the frozen core.

    Candidate genomes already exist in memory and share the active neat.Config.
    Routing them through the legacy v30 pickle loader is both unnecessary and
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
    """Score a genome by ARC-facing improvement over the original v30 seed.

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


def _is_neat_genome(obj) -> bool:
    return (
        obj is not None
        and not isinstance(obj, dict)
        and hasattr(obj, "nodes")
        and hasattr(obj, "connections")
        and hasattr(obj, "mutate")
    )


def _unwrap_seed_genome(obj):
    """Recover a neat-python genome from legacy v30 winner pickle wrappers.

    Older v30 artifacts may pickle a metadata dict rather than the raw genome.
    Prefer conventional winner keys, then recursively inspect nested values.
    """
    if _is_neat_genome(obj):
        return obj, "root"

    if isinstance(obj, dict):
        preferred = (
            "genome", "winner", "best_genome", "winner_genome",
            "neat_winner", "cppn_genome", "best",
        )
        for key in preferred:
            if key in obj:
                found, path = _unwrap_seed_genome(obj[key])
                if found is not None:
                    return found, f"{key}.{path}"

        # Fall back to recursively examining every value.  Only accept a
        # unique genome-like object so metadata dictionaries cannot be
        # silently misinterpreted.
        candidates = []
        for key, value in obj.items():
            found, path = _unwrap_seed_genome(value)
            if found is not None:
                candidates.append((found, f"{key}.{path}"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(path for _, path in candidates[:8])
            raise RuntimeError(
                "ARC-NEAT winner pickle contains multiple genome-like objects; "
                f"cannot choose safely: {names}"
            )

    # Some legacy containers may store the genome as an attribute.
    for attr in ("genome", "winner", "best_genome"):
        if hasattr(obj, attr):
            found, path = _unwrap_seed_genome(getattr(obj, attr))
            if found is not None:
                return found, f"{attr}.{path}"

    return None, None


def _seed_population(population, seed_genome, config, mutations: int):
    if not _is_neat_genome(seed_genome):
        raise TypeError(
            "ARC-NEAT seed is not a neat-python genome after unwrapping: "
            f"type={type(seed_genome).__name__}"
        )
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
    """Slow Baldwinian NEAT outer loop over the CPPN geometry only.

    All learned ARC parameters are held fixed. Each genome is installed directly
    into the existing PyTorch-NEAT CPPN runtime and then the frozen base operator
    geometry is refreshed, preserving ARC-wide/static and task-conditioned fast
    network parameters.
    """
    if not getattr(cfg, "arc_neat_enabled", False) or int(cfg.arc_neat_generations) <= 0:
        return None
    if val_data is None:
        print("ARC-NEAT skipped: no validation/meta-evaluation split available")
        return None

    try:
        import neat
    except Exception as exc:
        raise RuntimeError("ARC-NEAT requires neat-python in the active environment") from exc

    neat_cfg_path = _find_neat_config(cfg.arc_neat_config_path, cfg.neat_winner_path)
    neat_cfg_path = _render_neat_template(neat_cfg_path, cfg)
    print(f"ARC-NEAT config: {neat_cfg_path}")
    neat_cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_cfg_path),
    )

    with open(cfg.neat_winner_path, "rb") as f:
        seed_payload = pickle.load(f)
    seed_genome, seed_path = _unwrap_seed_genome(seed_payload)
    if seed_genome is None:
        if isinstance(seed_payload, dict):
            keys = list(seed_payload.keys())
            detail = f"dict keys={keys[:20]}"
        else:
            detail = f"type={type(seed_payload).__name__}"
        raise RuntimeError(
            "ARC-NEAT could not recover a neat-python genome from the v30 winner pickle; "
            + detail
        )
    print(
        "ARC-NEAT recovered seed genome: "
        f"path={seed_path} type={type(seed_genome).__name__} "
        f"nodes={len(getattr(seed_genome, 'nodes', {}))} "
        f"connections={len(getattr(seed_genome, 'connections', {}))}"
    )

    # Keep this first experiment intentionally small. Population size is a
    # runtime override so the original v30 config does not force 256 candidates.
    neat_cfg.pop_size = max(int(cfg.arc_neat_population), 2)
    population = neat.Population(neat_cfg)
    _seed_population(population, seed_genome, neat_cfg, cfg.arc_neat_seed_mutations)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    fitness_data, holdout_data = _split_neat_datasets(val_data, cfg)
    fixed_batches = _fixed_meta_batches(fitness_data, cfg, device)
    model.eval()

    # Establish the original v30 CPPN as a zero-point. Every candidate is scored
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
