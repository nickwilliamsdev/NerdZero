"""OFFICIAL big-grid budget probe: does the genome's value persist on the TRUE
test distribution at full (300-step) budget, or is it pure sample efficiency?

The train-pool budget probe (diag_adam_budget_grid30.py) found random-300 ->
0.999 on train tasks, i.e. the genome's value there is sample efficiency
(solving in ~100 steps where random needs ~300). But the all-120 harness smoke
showed random at only 0.45 @100 on OFFICIAL tasks (vs 0.86 on train) — official
tasks are harder for a random init. If random still can't reach ~1.0 @300 on
official tasks, the genome buys REAL accuracy on the true test distribution,
not just speed.

Three genomes:
  grid-15 adam-winner   (transferred init; the known bar, 0.98 @100 on train)
  grid-30 native gen-13 (the in-flight run's best-seen checkpoint; measures
                          whether grid-30-native selection already beats transfer)
  random                (the Adam-alone ceiling per budget)

12 OFFICIAL big-grid test tasks (16..30, the ones grid-15 couldn't even run),
single 300-step Adam pass per task, read at budgets 10/20/50/100/300.
"""
import pickle
import random

import numpy as np
import torch

import train_cppn_arc as T
from diag_adam_budget_grid30 import adapt_curve

GRID = 30
BUDGETS = [10, 20, 50, 100, 300]
N_TASKS = 12
SET_SEED = 7


def load_official_biggrid(n):
    """n official ARC test tasks where at least one grid is >15 (so grid-15
    couldn't run it) but all grids fit <=GRID. Deterministic by task order."""
    import arckit
    _, test_ds = arckit.load_data()
    big = []
    for t in test_ds.tasks:
        dims = []
        for grp in ("train", "test"):
            for x, y in getattr(t, grp, []):
                x, y = np.asarray(x), np.asarray(y)
                dims.extend([max(x.shape), max(y.shape)])
        if not dims:
            continue
        if max(dims) <= GRID and max(dims) > 15:   # fits 30, and is a big-grid task
            big.append(t)
    rng = random.Random(SET_SEED)
    return rng.sample(big, min(n, len(big)))


def main():
    store = T.ARCStore.load(GRID, seed=0)
    tasks = load_official_biggrid(N_TASKS)
    print(f"{len(tasks)} OFFICIAL big-grid tasks (grids >15; seed {SET_SEED})")
    print(f"task ids: {[str(getattr(t, 'id', ''))[:8] for t in tasks]}")
    print(f"(these are the tasks grid-15 could NOT run; all fit grid {GRID})")

    cfg = T.Config(phenotype='gated', channels=8, attn_dim=48, attn_heads=4,
                   attn_layers=2, grid_size=GRID, pop_size=16, n_eval=12,
                   seed=0, cppn_num_hidden=48, gated_branches=3,
                   eval_sgd_steps=300, sgd_lr=0.1,
                   in_context_optimizer='adam', adam_lr=0.01)
    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim, torch.device("cpu"),
                              schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()
    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]
    g15 = pickle.load(open("cppn_arc_adam100_pop16_e12_s0.pkl", "rb"))
    import glob
    g30_ckpt = sorted(glob.glob("ckpt_grid30_gen*.pkl"))[-1]
    g30 = pickle.load(open(g30_ckpt, "rb"))
    print(f"grid-30 native checkpoint: {g30_ckpt}")

    def score(genome):
        cppn = T.create_torch_cppn(genome=genome, config=neat_cfg,
                                   leaf_names=[f"c{i}" for i in range(pc.dim)],
                                   node_names=["w"])
        per = {b: [] for b in BUDGETS}
        rng = random.Random(0)
        for t in tasks:
            ep = store.make_episode(t, cfg.max_demos, rng)
            phen = T.build_phenotype(cfg).to(device)
            substrate = T._substrate_for(ep, pc, cfg)
            with torch.no_grad():
                W = T.encode(cppn, pc, substrate).to(device)
            T.load_param_vector(phen, pc, W)
            curve = adapt_curve(phen, ep, device, 300)
            for b in BUDGETS:
                per[b].append(curve[b])
        return {b: float(np.mean(v)) for b, v in per.items()}

    print("scoring grid-15 winner (transferred) on official big-grid tasks...")
    b15 = score(g15)
    print("scoring grid-30 native gen-13 checkpoint...")
    b30 = score(g30)
    print("scoring fresh random genome...")
    br = score(random_g)

    print()
    print(f"{'steps':>6} | {'g15-win':>8} | {'g30-gen13':>9} | {'random':>8} | {'g15-rand':>8} | {'g30-rand':>8}")
    print("-" * 62)
    for s in BUDGETS:
        print(f"{s:>6d} | {b15[s]:>8.4f} | {b30[s]:>9.4f} | {br[s]:>8.4f} | "
              f"{b15[s]-br[s]:>+8.4f} | {b30[s]-br[s]:>+8.4f}")
    print()
    print("Read: if random@300 stays well below 1.0 on official tasks, the")
    print("genome's value PERSISTS at full budget (real accuracy, not just")
    print("sample efficiency). Compare the +g30-rand column at 300 vs the")
    print("train-pool's collapsed +0.001.")


if __name__ == "__main__":
    main()
