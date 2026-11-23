"""Adam-300-step cross-eval on the common unseen 16-task set (seed 13).

The successor yardstick to eval_set13_300steps.py (SGD 300, lr 0.1, clip 10):
same unseen set, same task ordering, but in-context adaptation is Adam lr 0.01
unclipped for 300 steps — the protocol established by diag_adam_ceiling.txt
(2026-11-23), under which the old best genome scores 0.9837.

Genomes:
  - adam100 pop16 e12 s0   cppn_arc_adam100_pop16_e12_s0.pkl  (new Adam run)
  - old best pop16 e12 s0  sweep_lr10_e12_pop16_s0.pkl        (SGD-trained control)
  - random genome         (fresh random NEAT genome, same architecture)

Read: does meta-selection under the Adam protocol beat the fixed old best's
0.9837? And by how much does it beat random (the selection-signal check)?
"""
import pickle
import random

import numpy as np

import train_cppn_arc as T
import eval_set13_300steps as E

ADAM_LR = 1e-2
SGD_STEPS = 300

RUNS = [
    ("adam100 pop16 e12 s0", "cppn_arc_adam100_pop16_e12_s0.pkl"),
    ("old best pop16 e12 s0", "sweep_lr10_e12_pop16_s0.pkl"),
    ("random", None),
]


def make_cfg():
    return T.Config(
        phenotype='gated', channels=8, attn_dim=48, attn_heads=4, attn_layers=2,
        grid_size=E.GRID, pop_size=16, n_eval=12, seed=E.SEED,
        fixed_eval=True, cppn_num_hidden=48, gated_branches=3,
        eval_sgd_steps=SGD_STEPS, sgd_lr=0.1,
    )


def pure_eval(genome, cfg, eps, pc, neat_cfg, device):
    import torch
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(pc.dim)], node_names=["w"],
    )
    accs = []
    for ep in eps:
        phen = T.build_phenotype(cfg)
        substrate = T._substrate_for(ep, pc, cfg)
        with torch.no_grad():
            W = T.encode(cppn, pc, substrate).to(device)
        T.load_param_vector(phen, pc, W)
        T.sgd_adapt(phen, ep, SGD_STEPS, ADAM_LR, device,
                    optimizer="adam", adam_lr=ADAM_LR)
        accs.append(T.episode_accuracy(phen, ep, device))
    return accs


def main():
    import sys, torch
    store = T.ARCStore.load(E.GRID, seed=0)
    template_cfg = make_cfg()
    template = T.build_phenotype(template_cfg)
    pc = T.build_param_coords(template, template_cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(template_cfg)
    device = template_cfg.resolve_device()

    ids, eps = E.sample_unseen(store, template_cfg, E.N_TASKS, seed=E.SEED)
    print(f"Common unseen {E.N_TASKS}-task set (seed {E.SEED}, Adam lr={ADAM_LR} "
          f"{SGD_STEPS}-step in-context): {ids}")
    print()

    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]
    rows = []
    for label, pkl in RUNS:
        if pkl:
            with open(pkl, "rb") as f:
                genome = pickle.load(f)
        else:
            genome = random_g
        cfg = make_cfg()
        a = pure_eval(genome, cfg, eps, pc, neat_cfg, device)
        per = [round(x, 4) for x in a]
        print(f"{label}: mean={np.mean(a):.4f}  per-task={per}", flush=True)
        print(f"  {label} done", flush=True)
        rows.append((label, float(np.mean(a))))
    print()
    rows.sort(key=lambda r: -r[1])
    print("Ranking on common unseen set (Adam 300-step in-context):")
    for i, (l, m) in enumerate(rows, 1):
        print(f"  {i}. {l}  {m:.4f}")
    if len(rows) >= 3:
        best = max(m for l, m in rows if l != "random")
        rnd = [m for l, m in rows if l == "random"][0]
        print(f"gap best-evolved - random = {best - rnd:+.4f}")


if __name__ == "__main__":
    main()
