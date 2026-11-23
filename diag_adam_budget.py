"""Minimal in-context budget that discriminates init quality under Adam.

diag_adam_ceiling.py showed Adam(lr 0.01) at 300 steps lifts the current best
genome to 0.9837 on set-13 (vs 0.8544 SGD) and the random genome's Adam score
is the control that says whether the CPPN's init still matters under Adam.

If meta-selecting with Adam, the eval fitness needs an eval budget that
(a) discriminates good init from random init, and (b) stays cheap enough that
pop16/e12 runs are tractable. This probes the best genome AND a random genome
at Adam budgets 10/20/50/100/300 steps on the full 16-task set-13.

Read: for each budget B, the gap best(B) - random(B). The smallest B where the
gap is large AND best(B) is near best(300) is the eval budget to use for
Adam meta-selection. If random catches up at short budgets, the gap only opens
at long budgets -> use 300.
"""
import pickle
import time

import numpy as np
import torch

import train_cppn_arc as T
import eval_set13_300steps as E

BUDGETS = [10, 20, 50, 100, 300]
ADAM_LR = 1e-2


def per_task_budgets(genome, cfg, eps, pc, neat_cfg, device):
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(pc.dim)], node_names=["w"],
    )
    out = {}
    for steps in BUDGETS:
        accs = []
        for ep in eps:
            phen = T.build_phenotype(cfg)
            substrate = T._substrate_for(ep, pc, cfg)
            with torch.no_grad():
                W = T.encode(cppn, pc, substrate).to(device)
            T.load_param_vector(phen, pc, W)
            T.sgd_adapt(phen, ep, steps, ADAM_LR, device,
                        optimizer="adam", adam_lr=ADAM_LR)
            accs.append(T.episode_accuracy(phen, ep, device))
        out[steps] = accs
        print(f"  steps={steps:3d} mean={np.mean(accs):.4f}  "
              f"per-task={['%.3f' % x for x in accs]}", flush=True)
    return out


def main():
    store = T.ARCStore.load(E.GRID, seed=0)
    cfg = T.Config(
        phenotype='gated', channels=8, attn_dim=48, attn_heads=4, attn_layers=2,
        grid_size=E.GRID, pop_size=16, n_eval=12, seed=E.SEED,
        fixed_eval=True, cppn_num_hidden=48, gated_branches=3,
        eval_sgd_steps=300, sgd_lr=0.1,
    )
    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()

    ids, eps = E.sample_unseen(store, cfg, E.N_TASKS, seed=E.SEED)
    print(f"Unseen set-13 (seed {E.SEED}), Adam lr={ADAM_LR} at budgets {BUDGETS}")
    print(f"tasks: {ids}")

    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]
    with open("sweep_lr10_e12_pop16_s0.pkl", "rb") as f:
        best_g = pickle.load(f)

    t0 = time.time()
    results = {}
    for label, genome in [("best pop16-e12-s0", best_g), ("random", random_g)]:
        print(f"\n{label}:")
        results[label] = per_task_budgets(genome, cfg, eps, pc, neat_cfg, device)

    print("\n=== Adam mean accuracy by budget (gap = best - random) ===")
    print(f"{'budget':>7s}  {'best':>8s}  {'random':>8s}  {'gap':>7s}")
    for s in BUDGETS:
        b = np.mean(results["best pop16-e12-s0"][s])
        r = np.mean(results["random"][s])
        print(f"{s:>7d}  {b:8.4f}  {r:8.4f}  {b - r:+7.4f}")
    print("\ntotal wall: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
