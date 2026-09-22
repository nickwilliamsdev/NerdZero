from __future__ import annotations

ARC_NEAT_PATCH_ID = "v1.8.3-render-neat-template"

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
from sefer.evolution.evolve_transport import load_evolved_transport_cppn
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


def _install_genome(model, genome, temp_dir: Path):
    path = temp_dir / f"candidate_{int(getattr(genome, 'key', 0))}.pkl"
    with path.open("wb") as f:
        pickle.dump(genome, f, protocol=pickle.HIGHEST_PROTOCOL)
    load_evolved_transport_cppn(model, str(path))
    model.refresh_operator_base_from_core()


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


def _fitness_from_metrics(metrics, improve: float, complexity: float, cfg) -> float:
    # Query pixel accuracy is the strongest term. Demo-fit measures whether one
    # shared program actually explains the known examples; one-step improvement
    # rewards useful local geometry. Complexity is deliberately only a tie-breaker.
    return (
        float(cfg.arc_neat_query_weight) * float(metrics["pixel_acc"])
        + float(cfg.arc_neat_demo_fit_weight) * float(metrics.get("search_demo_fit", 0.0))
        + float(cfg.arc_neat_improve_weight) * float(improve)
        - float(cfg.arc_neat_complexity_weight) * float(complexity)
    )


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

    All learned ARC parameters are held fixed. A genome is installed through the
    existing v30 loader, which regenerates the frozen base operator geometry while
    preserving ARC-wide/static and task-conditioned fast-network parameters.
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

    fixed_batches = _fixed_meta_batches(val_data, cfg, device)
    eval_tasks = max(int(cfg.arc_neat_eval_tasks), 1)
    model.eval()

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
                    _install_genome(model, genome, temp_dir)
                    improve = _one_step_improvement(model, fixed_batches)
                    metrics = evaluate_arc(model, val_data, limit=eval_tasks, device=device)
                    complexity = _genome_complexity(genome)
                    fitness = _fitness_from_metrics(metrics, improve, complexity, cfg)
                except Exception as exc:
                    fitness = -1e6
                    metrics = {"pixel_acc": 0.0, "search_demo_fit": 0.0}
                    improve = 0.0
                    complexity = _genome_complexity(genome)
                    print(f"ARC-NEAT candidate {gid} failed: {type(exc).__name__}: {exc}")
                genome.fitness = float(fitness)
                if fitness > best_seen["fitness"]:
                    best_seen.update(fitness=float(fitness), metrics=dict(metrics), key=gid)
                print(
                    f"  cand={idx + 1:02d}/{len(genomes):02d} key={gid} fit={fitness:.4f} "
                    f"pixel={metrics.get('pixel_acc', 0.0):.3f} "
                    f"demoFit={metrics.get('search_demo_fit', 0.0):.3f} "
                    f"improve={improve:.3f} complexity={complexity:.0f}"
                )
            generation_counter["value"] += 1

        winner = population.run(evaluate_genomes, int(cfg.arc_neat_generations))

        winner_path = Path(cfg.arc_neat_winner_path)
        with winner_path.open("wb") as f:
            pickle.dump(winner, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"ARC-NEAT saved winner: {winner_path} fitness={winner.fitness:.4f}")

        # Install the winning geometry while preserving all trained ARC weights.
        _install_genome(model, winner, temp_dir)
        final = evaluate_arc(model, val_data, limit=max(eval_tasks, int(cfg.eval_tasks)), device=device)
        print(
            f"ARC-NEAT winner eval pixel={final['pixel_acc']:.3f} "
            f"greedyPixel={final.get('greedy_pixel_acc', 0.0):.3f} "
            f"demoFit={final.get('search_demo_fit', 0.0):.3f} "
            f"directPixel={final.get('direct_pixel_acc', 0.0):.3f}"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": vars(cfg),
                "metrics": final,
                "arc_neat_winner_path": str(winner_path),
                "arc_neat_fitness": float(winner.fitness),
            },
            cfg.arc_neat_checkpoint_path,
        )
        print(f"ARC-NEAT saved evolved ARC checkpoint: {cfg.arc_neat_checkpoint_path}")
        return {"winner": winner, "metrics": final, "fitness": float(winner.fitness)}
