"""ALL-120 official baseline at grid 30: full official test set (120 tasks),
the two genomes that matter right now — the grid-15 winner (the cheap option:
transfer the known-best genome to grid 30) and random (the Adam-alone ceiling)
— scored with a single 300-step Adam pass per task, read at 100 and 300 steps.

Why this is the decisive number: the official big-grid budget probe showed the
genome is LOAD-BEARING on the true test distribution (random@300 = 0.8607, not
~1.0 as on the train pool), so the genome's value on the hard unseen tasks is
real accuracy, not just speed. This run measures that on the FULL official
set and identifies the HARD tasks (where random@300 < 0.9) and how many the
genome solves — the failure tasks are what every future iteration targets.

Cost on the contended GB10: ~125 s per task per genome (measured off the
budget probe) x 120 tasks x 2 genomes x 300-step single-pass ~ 8 h. Runs in
parallel with the in-flight grid-30 meta run (shared GPU, both slower).
"""
import pickle
import random

import numpy as np
import torch

import train_cppn_arc as T
from diag_adam_budget_grid30 import adapt_curve

GRID = 30
READS = [100, 300]
HARD_THR = 0.90   # a task is "hard" if random@300 < this
SOLVE_THR = 0.99  # a task is "solved" if a genome@300 >= this


def load_all_official():
    import arckit
    _, test_ds = arckit.load_data()
    ok = []
    for t in test_ds.tasks:
        fit = True
        for grp in ("train", "test"):
            for x, y in getattr(t, grp, []):
                x, y = np.asarray(x), np.asarray(y)
                if max(x.shape) > GRID or max(y.shape) > GRID:
                    fit = False
                    break
            if not fit:
                break
        if fit:
            ok.append(t)
    return ok


def main():
    import json
    store = T.ARCStore.load(GRID, seed=0)
    tasks = load_all_official()
    print(f"All {len(tasks)} official test tasks fit grid {GRID}")
    print(f"task ids: {[str(getattr(t, 'id', ''))[:8] for t in tasks]}")

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

    genomes = [("grid15-winner", g15), ("random", random_g)]
    out_path = "eval_all120_grid30.jsonl"
    out = open(out_path, "w")

    def score(genome, label):
        cppn = T.create_torch_cppn(genome=genome, config=neat_cfg,
                                   leaf_names=[f"c{i}" for i in range(pc.dim)],
                                   node_names=["w"])
        per = {r: [] for r in READS}
        rng = random.Random(0)
        for ti, t in enumerate(tasks):
            ep = store.make_episode(t, cfg.max_demos, rng)
            phen = T.build_phenotype(cfg).to(device)
            substrate = T._substrate_for(ep, pc, cfg)
            with torch.no_grad():
                W = T.encode(cppn, pc, substrate).to(device)
            T.load_param_vector(phen, pc, W)
            curve = adapt_curve(phen, ep, device, 300, reads=READS)
            rec = {"genome": label, "task": str(getattr(t, "id", "")),
                   "acc": {r: curve[r] for r in READS}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            for r in READS:
                per[r].append(curve[r])
            if ti % 10 == 0:
                print(f"  {label} task {ti}/{len(tasks)} "
                      f"(mean@300 so far {np.mean(per[300]):.3f})", flush=True)
        return per

    results = {}
    for label, g in genomes:
        print(f"scoring {label} on all {len(tasks)} official tasks...", flush=True)
        results[label] = score(g, label)
    out.close()

    # ---- report ----
    a100 = {k: np.array(v[100]) for k, v in results.items()}
    a300 = {k: np.array(v[300]) for k, v in results.items()}
    print()
    print(f"{'genome':16s} | {'mean@100':>9} | {'mean@300':>9} | {'solved@300':>10}")
    print("-" * 50)
    for k in results:
        solved = int(np.sum(a300[k] >= SOLVE_THR))
        print(f"{k:16s} | {a100[k].mean():>9.4f} | {a300[k].mean():>9.4f} | "
              f"{solved:>4d}/{len(tasks)}")
    print()
    # hard tasks = where random@300 < HARD_THR
    hard = np.where(a300["random"] < HARD_THR)[0]
    print(f"HARD tasks (random@300 < {HARD_THR}): {len(hard)} of {len(tasks)}")
    solved_hard = int(np.sum(a300["grid15-winner"][hard] >= SOLVE_THR))
    print(f"  grid15-winner solves {solved_hard}/{len(hard)} of those hard tasks")
    print()
    print("gap (grid15-winner - random): @100 = "
          f"{a100['grid15-winner'].mean()-a100['random'].mean():+.4f}   "
          f"@300 = {a300['grid15-winner'].mean()-a300['random'].mean():+.4f}")
    print()
    print("Per-task (id, random@300, g15@300) sorted by random@300 ascending:")
    order = np.argsort(a300["random"])
    for i in order:
        rid = str(getattr(tasks[i], "id", ""))[:8]
        flag = "HARD" if a300["random"][i] < HARD_THR else "    "
        gs = "SOLVED" if a300["grid15-winner"][i] >= SOLVE_THR else "miss"
        print(f"  {rid}  rand300={a300['random'][i]:.3f} g15_300="
              f"{a300['grid15-winner'][i]:.3f}  {flag} {gs}")


if __name__ == "__main__":
    main()
