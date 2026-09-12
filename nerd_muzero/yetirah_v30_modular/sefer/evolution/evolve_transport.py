from __future__ import annotations

import math
import random
import os
import pickle
import tempfile
import textwrap
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sefer.algebra.rollout import differentiable_rollout
from sefer.controllers.reasoner import TinyReasoner
from sefer.evolution.neat_config import neat_config_text
from sefer.evolution.neat_runtime import DifferentiableGenomeCPPN, EvolvedTorchCPPN, NEAT_INPUT_NAMES, NEAT_OUTPUT_NAMES, require_pytorch_neat
from sefer.tasks.synthetic_algebra import SyntheticTaskBatch, primitive_transport_sources
from sefer.utils import flatten_params, set_flat_params_

def genotype_modules(model: TinyReasoner) -> nn.Module:
    """
    For V0 we treat the CPPN + operator hypernetwork + operator codes as the
    continuous 'genotype-like' parameters.

    To keep this easy to flatten, wrap references in a lightweight container.
    """
    holder = nn.Module()
    holder.add_module("cppn", model.core.cppn)
    holder.add_module("op_hyper", model.core.op_hyper)

    # operator_codes is not a module, so flatten ES manually elsewhere.
    return holder

def get_genotype_vector(model: TinyReasoner) -> torch.Tensor:
    chunks = [
        flatten_params(model.core.cppn),
        flatten_params(model.core.op_hyper),
        model.core.operator_codes.detach().reshape(-1),
    ]
    return torch.cat(chunks)

def set_genotype_vector_(model: TinyReasoner, flat: torch.Tensor):
    n_cppn = sum(p.numel() for p in model.core.cppn.parameters() if p.requires_grad)
    n_op = sum(p.numel() for p in model.core.op_hyper.parameters() if p.requires_grad)

    set_flat_params_(model.core.cppn, flat[:n_cppn])
    set_flat_params_(model.core.op_hyper, flat[n_cppn:n_cppn+n_op])

    code_flat = flat[n_cppn+n_op:]
    model.core.operator_codes.copy_(code_flat.view_as(model.core.operator_codes))

def evaluate_fixed_batch(
    model: TinyReasoner,
    demo_x: torch.Tensor,
    demo_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    rollout_steps: int = 1,
) -> float:
    rule = model.encode_rule(demo_x, demo_y)
    H0 = model.encode_query(query_x)
    H, _ = differentiable_rollout(model, rule, H0, steps=rollout_steps, hard=True)
    pred = model.decode_query(H)
    return float(F.mse_loss(pred, query_y).item())

def antithetic_es_step_(
    model: TinyReasoner,
    task_source: SyntheticTaskBatch,
    device,
    pairs: int = 4,
    sigma: float = 0.01,
    lr: float = 0.02,
    eval_batch_size: int = 32,
    rollout_steps: int = 3,
):
    """
    Minimize loss using a tiny antithetic ES neighborhood.

    Important: perturb the compact generative parameter vector, not the
    generated adjacency/output tensors.
    """
    base = get_genotype_vector(model)
    (
        demo_x,
        demo_y,
        query_x,
        query_y,
        _,
    ) = task_source.sample(
        eval_batch_size,
        device,
    )
    noises = []
    losses_plus = []
    losses_minus = []

    # Use the exact same batch for +epsilon and -epsilon so the
    # antithetic difference isolates the parameter perturbation.
    for _ in range(pairs):
        eps = torch.randn_like(base)
        noises.append(eps)

        set_genotype_vector_(model, base + sigma * eps)
        lp = evaluate_fixed_batch(
            model,
            demo_x,
            demo_y,
            query_x,
            query_y,
            rollout_steps,
        )
        set_genotype_vector_(model, base - sigma * eps)
        lm = evaluate_fixed_batch(
            model,
            demo_x,
            demo_y,
            query_x,
            query_y,
            rollout_steps,
        )

        losses_plus.append(lp)
        losses_minus.append(lm)

    set_genotype_vector_(model, base)

    # Gradient estimate for minimizing loss.
    g = torch.zeros_like(base)
    for eps, lp, lm in zip(noises, losses_plus, losses_minus):
        g += (lp - lm) * eps
    g /= (2.0 * pairs * sigma)

    # Only normalize/apply ES when the antithetic signal is meaningfully nonzero.
    raw_grad_norm = g.norm()
    if raw_grad_norm > 1e-6:
        g = g / raw_grad_norm
        set_genotype_vector_(model, base - lr * g)
    else:
        set_genotype_vector_(model, base)

    return {
        "es_loss_plus": sum(losses_plus) / len(losses_plus),
        "es_loss_minus": sum(losses_minus) / len(losses_minus),
        "es_grad_norm": float(raw_grad_norm.item()),
    }

def _neat_transport_metrics_from_runtime(runtime: EvolvedTorchCPPN, coords: torch.Tensor,
                                         operator_codes: torch.Tensor):
    n = coords.shape[0]
    losses, accs, ents = [], [], []
    for pid in range(4):
        logits = runtime.edge_logits(
            _standalone_neat_edge_features(coords, operator_codes[pid])
        )
        if logits.ndim > 2:
            logits = logits.squeeze(-1)
        P = torch.softmax(logits, dim=-1)
        target_src = primitive_transport_sources(pid, n, coords.device)
        chosen = P[torch.arange(n, device=coords.device), target_src].clamp_min(1e-8)
        losses.append(-chosen.log().mean())
        accs.append((P.argmax(dim=-1) == target_src).float().mean())
        ents.append(-(P * torch.log(P.clamp_min(1e-8))).sum(dim=-1).mean() / math.log(n))
    return torch.stack(losses).mean(), torch.stack(accs), torch.stack(ents)

def _standalone_neat_edge_features(coords: torch.Tensor, code: torch.Tensor) -> Dict[str, torch.Tensor]:
    n, d = coords.shape
    dst = coords[:, None, :].expand(n, n, -1)
    src = coords[None, :, :].expand(n, n, -1)
    diff = dst - src
    prod = dst * src
    dist = diff.pow(2).sum(dim=-1).sqrt()
    idx = torch.linspace(-1.0, 1.0, n, device=coords.device, dtype=coords.dtype)
    dst_idx = idx[:, None].expand(n, n)
    src_idx = idx[None, :].expand(n, n)
    feats: Dict[str, torch.Tensor] = {}
    for i in range(d):
        feats[f"dst_{i}"] = dst[..., i]
        feats[f"src_{i}"] = src[..., i]
        feats[f"diff_{i}"] = diff[..., i]
        feats[f"prod_{i}"] = prod[..., i]
    feats["dist"] = dist
    feats["dst_idx"] = dst_idx
    feats["src_idx"] = src_idx
    feats["idx_diff"] = dst_idx - src_idx
    feats["idx_sum"] = dst_idx + src_idx
    feats["is_diag"] = torch.eye(n, device=coords.device, dtype=coords.dtype)
    phase = torch.arange(n, device=coords.device, dtype=coords.dtype) * (2.0 * math.pi / n)
    dst_phase = phase[:, None].expand(n, n)
    src_phase = phase[None, :].expand(n, n)
    rel_phase = src_phase - dst_phase
    feats["dst_phase_sin"] = torch.sin(dst_phase)
    feats["dst_phase_cos"] = torch.cos(dst_phase)
    feats["src_phase_sin"] = torch.sin(src_phase)
    feats["src_phase_cos"] = torch.cos(src_phase)
    feats["rel_phase_sin"] = torch.sin(rel_phase)
    feats["rel_phase_cos"] = torch.cos(rel_phase)
    for i in range(code.numel()):
        feats[f"op_{i}"] = torch.ones_like(dist) * code[i].detach()
    return feats

def load_evolved_transport_cppn(model: TinyReasoner, winner_path: str = "yetirah_v30_neat_winner.pkl"):
    """Reload an evolved NEAT winner and install its PyTorch-NEAT CPPN graph."""
    neat, create_cppn_fn = require_pytorch_neat()
    with open(winner_path, "rb") as f:
        payload = pickle.load(f)
    genome = payload["winner"]
    cfg_txt = payload["config_text"]
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
        f.write(cfg_txt)
        cfg_path = f.name
    try:
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path
        )
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
    model.core.op_hyper.install_evolved_cppn(genome, config, create_cppn_fn)
    model.core.operator_codes.requires_grad_(False)
    return genome, config

def evolve_transport_cppn(model: TinyReasoner, generations: int = 100, pop_size: int = 32,
                          seed: int = 0, save_path: str = "yetirah_v30_neat_winner.pkl",
                          workers: int = 12, inner_steps: int = 10, inner_lr: float = 1e-2):
    """Evolve one shared CPPN that generates all four anchored transport laws.

    Fitness rewards low transport CE, correct argmax permutation rows, low row
    entropy, and compact genomes. The first four operator codes are frozen
    during/after evolution so the evolved CPPN's semantics cannot drift under Adam.
    """
    neat, create_cppn_fn = require_pytorch_neat()
    cfg_txt = neat_config_text(len(NEAT_INPUT_NAMES), pop_size, seed)
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
        f.write(cfg_txt)
        cfg_path = f.name
    try:
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path
        )
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass

    coords = model.core.coords.detach().cpu()
    codes = model.core.operator_codes.detach().cpu().clone()

    best_seen = {"fitness": -1e30, "ce": None, "acc": None, "ent": None, "c2": None, "c3": None, "loss0": None, "lossT": None, "learn": None, "auc": None}

    def _transport_matrices(runtime):
        mats = []
        for pid in range(4):
            logits = runtime.edge_logits(_standalone_neat_edge_features(coords, codes[pid]))
            if logits.ndim > 2:
                logits = logits.squeeze(-1)
            mats.append(torch.softmax(logits, dim=-1))
        return mats

    def _target_perm(pid):
        n = coords.shape[0]
        src = primitive_transport_sources(pid, n, coords.device)
        P = torch.zeros(n, n, dtype=coords.dtype)
        P[torch.arange(n), src] = 1.0
        return P

    target_mats = [_target_perm(pid) for pid in range(4)]

    def _compose(mats, seq):
        out = torch.eye(coords.shape[0], dtype=coords.dtype)
        for a in seq:
            out = mats[a] @ out
        return out

    depth2 = [(a,b) for a in range(4) for b in range(4)]
    depth3 = [(a,b,c) for a in range(4) for b in range(4) for c in range(4)]

    def _diff_transport_matrices(runtime):
        mats = []
        for pid in range(4):
            logits = runtime(_standalone_neat_edge_features(coords, codes[pid]))
            mats.append(torch.softmax(logits, dim=-1))
        return mats

    def _diff_composition_ce(mats, seqs):
        n = coords.shape[0]
        rows = torch.arange(n)
        ces = []
        for seq in seqs:
            pred = _compose(mats, seq)
            tgt = _compose(target_mats, seq)
            src = tgt.argmax(dim=-1)
            chosen = pred[rows, src].clamp_min(1e-8)
            ces.append(-chosen.log().mean())
        return torch.stack(ces).mean()

    def _inner_loss(runtime):
        mats = _diff_transport_matrices(runtime)
        n = coords.shape[0]
        rows = torch.arange(n)
        primitive_ces, target_probs = [], []
        for pid, P in enumerate(mats):
            src = primitive_transport_sources(pid, n, coords.device)
            chosen = P[rows, src].clamp_min(1e-8)
            primitive_ces.append(-chosen.log().mean())
            target_probs.append(chosen.mean())
        primitive_ces_t = torch.stack(primitive_ces)
        probs = torch.stack(target_probs)

        # Adaptive balancing: weak primitives receive more of the gradient budget.
        # Detaching the weights prevents the weighting rule itself from becoming
        # another optimization path; gradients still flow through each CE term.
        inv = 1.0 / probs.detach().clamp_min(0.02)
        weights = inv / inv.mean()
        pce = (weights * primitive_ces_t).mean()

        # Strong smooth bottleneck pressure. This approximates max-loss/min-quality
        # without the discontinuity of argmax accuracy, so the inner loop directly
        # attacks whichever primitive is currently weakest.
        tau = 0.08
        soft_min_prob = -tau * torch.logsumexp(-probs / tau, dim=0)
        c2 = _diff_composition_ce(mats, depth2)
        c3 = _diff_composition_ce(mats, depth3)
        return pce + 0.35 * c2 + 0.15 * c3 + 2.00 * (1.0 - soft_min_prob)

    def _composition_score(mats, seqs):
        accs, ces = [], []
        n = coords.shape[0]
        rows = torch.arange(n)
        for seq in seqs:
            pred = _compose(mats, seq)
            tgt = _compose(target_mats, seq)
            src = tgt.argmax(dim=-1)
            chosen = pred[rows, src].clamp_min(1e-8)
            ces.append(-chosen.log().mean())
            accs.append((pred.argmax(dim=-1) == src).float().mean())
        return torch.stack(ces).mean(), torch.stack(accs).mean()

    def score_one(item, neat_config, adapt_steps: int):
        gid, genome = item
        try:
            # Tiered memetic inner loop. Every genome gets a cheap adaptation pass;
            # promising genomes are revisited below for additional inherited steps.
            # Because trained parameters are written back into the actual genome,
            # later tiers continue from the weights learned by earlier tiers.
            diff_runtime = DifferentiableGenomeCPPN(genome, neat_config)
            opt = torch.optim.Adam(diff_runtime.parameters(), lr=inner_lr)
            with torch.no_grad():
                loss0 = float(_inner_loss(diff_runtime).item())
            qualities = []
            for _ in range(max(0, int(adapt_steps))):
                loss = _inner_loss(diff_runtime)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diff_runtime.parameters(), 5.0)
                opt.step()
                qualities.append(float(torch.exp(-loss.detach()).item()))
            with torch.no_grad():
                lossT = float(_inner_loss(diff_runtime).item())
            diff_runtime.write_back_()

            # Re-materialize through Uber PyTorch-NEAT so fitness reflects the
            # exact graph used downstream, not only the differentiable mirror.
            runtime = EvolvedTorchCPPN(genome, neat_config, create_cppn_fn)
            with torch.no_grad():
                ce, acc, ent = _neat_transport_metrics_from_runtime(runtime, coords, codes)
                mats = _transport_matrices(runtime)
                c2_ce, c2_acc = _composition_score(mats, depth2)
                c3_ce, c3_acc = _composition_score(mats, depth3)
            ce_v = float(ce.item())
            acc_list = [float(v) for v in acc.tolist()]
            acc_v = float(acc.mean().item())
            ent_v = float(ent.mean().item())
            c2_ce_v, c2_acc_v = float(c2_ce.item()), float(c2_acc.item())
            c3_ce_v, c3_acc_v = float(c3_ce.item()), float(c3_acc.item())
            complexity = len(genome.nodes) + len(genome.connections)
            min_acc_v = min(acc_list)
            acc_spread_v = max(acc_list) - min_acc_v
            learnability = max(-1.0, min(1.0, (loss0 - lossT) / (abs(loss0) + 1e-8)))
            auc_quality = sum(qualities) / max(1, len(qualities)) if qualities else math.exp(-lossT)
            fitness = (
                3.0 * acc_v + 6.0 * min_acc_v + 3.0 * math.exp(-ce_v)
                + 2.0 * c2_acc_v + 1.5 * math.exp(-c2_ce_v)
                + 1.0 * c3_acc_v + 0.75 * math.exp(-c3_ce_v)
                + 1.25 * math.exp(-lossT)
                + 0.75 * learnability + 0.50 * auc_quality
                - 0.25 * acc_spread_v
                - 0.05 * ent_v - 0.0005 * complexity
            )
            return (gid, fitness, ce_v, acc_v, ent_v, c2_ce_v, c2_acc_v,
                    c3_ce_v, c3_acc_v, loss0, lossT, learnability, auc_quality,
                    acc_list, int(adapt_steps))
        except Exception:
            return (gid, -1e9, 99.0, 0.0, 1.0, 99.0, 0.0, 99.0, 0.0,
                    99.0, 99.0, -1.0, 0.0, [0.0, 0.0, 0.0, 0.0], int(adapt_steps))

    # Semantic stopping target: don't terminate merely because a scalar fitness
    # happens to cross a threshold. The transport algebra itself must be strong.
    semantic_target = dict(min_primitive=0.95, depth2=0.95, depth3=0.90)
    semantic_winner = {"genome": None, "metrics": None}

    class _SemanticStop(RuntimeError):
        pass

    def _run_items(items, neat_config, steps):
        if not items:
            return []
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(lambda item: score_one(item, neat_config, steps), items))
        return [score_one(item, neat_config, steps) for item in items]

    def eval_genomes(genomes, neat_config):
        genomes = list(genomes)
        by_id = {gid: genome for gid, genome in genomes}

        # Successive-halving style adaptation budget. For inner_steps=32 this is
        # 8 steps for everyone, +8 for the top quartile, +16 for the top decile.
        base_steps = max(1, inner_steps // 4)
        mid_steps = max(0, inner_steps // 4)
        elite_steps = max(0, inner_steps - base_steps - mid_steps)

        results = _run_items(genomes, neat_config, base_steps)
        latest = {r[0]: r for r in results}

        ranked = sorted(results, key=lambda r: r[1], reverse=True)
        mid_n = max(1, math.ceil(len(genomes) * 0.25))
        mid_items = [(gid, by_id[gid]) for gid, *_ in ranked[:mid_n]]
        if mid_steps > 0:
            for r in _run_items(mid_items, neat_config, mid_steps):
                latest[r[0]] = r

        reranked = sorted(latest.values(), key=lambda r: r[1], reverse=True)
        elite_n = max(1, math.ceil(len(genomes) * 0.10))
        elite_items = [(gid, by_id[gid]) for gid, *_ in reranked[:elite_n]]
        if elite_steps > 0:
            for r in _run_items(elite_items, neat_config, elite_steps):
                latest[r[0]] = r

        final_results = list(latest.values())
        for r in final_results:
            (gid, fitness, ce_v, acc_v, ent_v, c2_ce_v, c2_acc_v, c3_ce_v,
             c3_acc_v, loss0_v, lossT_v, learn_v, auc_v, acc_list, used_steps) = r
            genome = by_id[gid]
            genome.fitness = fitness
            if fitness > best_seen["fitness"]:
                best_seen.update(fitness=fitness, ce=ce_v, acc=acc_v, ent=ent_v,
                                 c2=c2_acc_v, c3=c3_acc_v, loss0=loss0_v,
                                 lossT=lossT_v, learn=learn_v, auc=auc_v)

        best_r = max(final_results, key=lambda r: r[1])
        best_gid = best_r[0]
        best_accs = best_r[13]
        best_c2 = best_r[6]
        best_c3 = best_r[8]
        print(
            f"  tiered-inner base={base_steps} top25+={mid_steps} top10+={elite_steps} "
            f"bestMinAcc={min(best_accs):.3f} depth2={best_c2:.3f} depth3={best_c3:.3f}"
        )

        if (min(best_accs) >= semantic_target["min_primitive"]
                and best_c2 >= semantic_target["depth2"]
                and best_c3 >= semantic_target["depth3"]):
            import copy
            semantic_winner["genome"] = copy.deepcopy(by_id[best_gid])
            semantic_winner["metrics"] = best_r
            raise _SemanticStop(
                f"semantic transport target met: minAcc={min(best_accs):.3f}, "
                f"depth2={best_c2:.3f}, depth3={best_c3:.3f}"
            )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    try:
        winner = pop.run(eval_genomes, generations)
    except _SemanticStop as exc:
        winner = semantic_winner["genome"]
        print(f"\nSemantic NEAT stop: {exc}")
        if winner is None:
            raise

    runtime = EvolvedTorchCPPN(winner, config, create_cppn_fn)
    with torch.no_grad():
        ce, acc, ent = _neat_transport_metrics_from_runtime(runtime, coords, codes)
    print("\nNEAT transport winner")
    print(f"  innerLoop steps={inner_steps} lr={inner_lr:g} bestSeen L0={best_seen['loss0']:.4f} LT={best_seen['lossT']:.4f} learn={best_seen['learn']:.3f} aucQ={best_seen['auc']:.3f}")
    with torch.no_grad():
        mats = _transport_matrices(runtime)
        c2_ce, c2_acc = _composition_score(mats, depth2)
        c3_ce, c3_acc = _composition_score(mats, depth3)
    print(f"  fitness={winner.fitness:.6f} CE={ce.item():.6f} meanAcc={acc.mean().item():.4f} meanH={ent.mean().item():.4f}")
    print(f"  depth2Acc={c2_acc.item():.4f} depth2CE={c2_ce.item():.4f} depth3Acc={c3_acc.item():.4f} depth3CE={c3_ce.item():.4f}")
    for pid in range(4):
        print(f"  T{pid}Acc={acc[pid].item():.3f}/H={ent[pid].item():.3f}")
    print(f"  nodes={len(winner.nodes)} connections={len(winner.connections)}")

    # Install the actual PyTorch-NEAT graph and freeze the operator-code coordinate
    # system it was evolved against. Feature-affine weights remain trainable.
    model.core.op_hyper.install_evolved_cppn(winner, config, create_cppn_fn)
    model.core.operator_codes.requires_grad_(False)
    with open(save_path, "wb") as f:
        pickle.dump({"winner": winner, "config_text": cfg_txt}, f)
    print(f"  saved NEAT winner: {save_path}")
    return winner, config
