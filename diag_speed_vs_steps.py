"""Where does the CPPN's value live? (2026-11-21)

The 300-step unseen scores are all ~0.78-0.83 for three very different
genomes (baseline 49-node, gated96 97-node, random 14-node) — and a
random-init gated model was reported (commit 228cc3f) to solve 5 probe
tasks at 300 steps. If even random init reaches ~0.78 at 300 steps, the
CPPN is not buying anything at that budget: the in-context SGD itself
does the job. The CPPN's value, if any, must show up as FASTER
convergence — i.e. a gap at LOW step counts.

This script measures, on the common unseen 16-task set (seed 13), the
accuracy-vs-step-count curve for three "starting weight" providers:
  1. baseline genome   (49 nodes, the 0.7956-line winner)
  2. gated96 genome    (97 nodes, this session's 300-step-trained winner)
  3. random genome     (a fresh random CPPN, same default size)
at step counts 0, 5, 10, 20, 30, 50, 100, 300 (sgd lr 0.05, clipped, the
current code path — all genomes under the same protocol).

Reading guide:
  - if random ~= genomes at 300 and random < genomes at low steps:
    the CPPN buys speed, so shorter eval budgets (where the init matters)
    are the right fitness yardstick, not 300.
  - if random ~= genomes at EVERY step count: the meta-training is
    encoding nothing the phenotype cannot learn in a few steps, and the
    whole line's signal has to change (not more of the same knobs).
"""
import pickle
import random

import numpy as np

import train_cppn_arc as T
import eval_set13_300steps as E  # reuse store/pc/episode plumbing

STEPS = [0, 5, 10, 20, 50]
LR = 0.05


def make_random_genome(neat_cfg, seed):
    # Use the exact same seeding path as the training runs: a fresh
    # population of one genome, seeded by neat 2.0's own code.
    pop = T.neat.Population(neat_cfg)
    g = list(pop.population.values())[0]
    return g


def curve(genome, cfg, eps, pc, neat_cfg, device):
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(pc.dim)], node_names=["w"],
    )
    out = {}
    for steps in STEPS:
        accs = []
        for ep in eps:
            phen = T.build_phenotype(cfg)
            substrate = T._substrate_for(ep, pc, cfg)
            with torch_no_grad():
                W = T.encode(cppn, pc, substrate).to(device)
            T.load_param_vector(phen, pc, W)
            if steps > 0:
                T.sgd_adapt(phen, ep, steps, LR, device)
            accs.append(T.episode_accuracy(phen, ep, device))
        out[steps] = float(np.mean(accs))
    return out


def torch_no_grad():
    import torch
    return torch.no_grad()


def main():
    import torch
    store = T.ARCStore.load(E.GRID, seed=0)
    template_cfg = E.make_cfg(LR)
    template = T.build_phenotype(template_cfg)
    pc = T.build_param_coords(template, template_cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(template_cfg)
    device = template_cfg.resolve_device()

    ids, eps = E.sample_unseen(store, template_cfg, E.N_TASKS, seed=E.SEED)
    print(f"Unseen {E.N_TASKS}-task set (seed {E.SEED}), sgd_lr={LR}, clipped:")
    print(f"tasks: {ids}")
    print()

    with open("sweep_ctrl_delta48_e6.pkl", "rb") as f:
        base = pickle.load(f)
    with open("cppn_arc_gated96_pop4_e300.pkl", "rb") as f:
        g96 = pickle.load(f)
    rgen = make_random_genome(neat_cfg, seed=123)

    curves = {}
    for label, genome in [("baseline49", base),
                          ("gated96", g96),
                          ("random14", rgen)]:
        curves[label] = curve(genome, template_cfg, eps, pc, neat_cfg, device)
        line = "  ".join(f"{s}:{v:.4f}" for s, v in curves[label].items())
        print(f"{label:12s} {line}")

    print()
    print("step      " + "  ".join(f"{s:>6d}" for s in STEPS))
    for label in curves:
        print(f"{label:9s} " +
              "  ".join(f"{curves[label][s]:6.4f}" for s in STEPS))
    print()
    print("gated96 - random:")
    for s in STEPS:
        print(f"  {s:>4d}: {curves['gated96'][s] - curves['random14'][s]:+.4f}")
    print("baseline49 - random:")
    for s in STEPS:
        print(f"  {s:>4d}: {curves['baseline49'][s] - curves['random14'][s]:+.4f}")


if __name__ == "__main__":
    main()
