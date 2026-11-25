"""Cross-eval on the OFFICIAL ARC test set at a chosen grid size.

Two modes:
  grid=15  -> only the 9 of 120 tasks that fit 15x15 (the others are 16..30).
  grid=30  -> ALL 120 tasks fit (verified: 120/120 at grid 30).

Protocol per task: encode genome on the task's dynamic substrate -> N in-
context steps -> cell-wise accuracy on the unpadded target region. Both the
established SGD protocol (lr 0.1, clip 10) and the Adam protocol (lr 0.01,
unclipped) can be run per genome.

Cost reality (grid 30, contended GB10): ~7800 s per full generation of the
meta run. A full 120-task pass at 100 steps is the ranking budget; 300 steps
is the final-verdict budget for the top 2-3 genomes. Use --steps 100 for the
wide ranking, --steps 300 only for finalists. Use --limit to smoke-test a
small slice first.

Genomes (each: label, pkl-or-None, [optimizers to run]):
  grid-30 Adam winner   (added here when it lands)
  grid-15 Adam winner   cppn_arc_adam100_pop16_e12_s0.pkl
  old best (grid-15)    sweep_lr10_e12_pop16_s0.pkl
  random                fresh random genome
"""
import argparse
import pickle

import numpy as np
import torch

import train_cppn_arc as T


def load_official_test(grid):
    import arckit
    _, test_ds = arckit.load_data()
    ok = []
    for t in test_ds.tasks:
        fit = True
        for grp in ("train", "test"):
            for x, y in getattr(t, grp, []):
                x, y = np.asarray(x), np.asarray(y)
                if (x.shape[0] > grid or x.shape[1] > grid or
                        y.shape[0] > grid or y.shape[1] > grid):
                    fit = False
                    break
            if not fit:
                break
        if fit:
            ok.append(t)
    return ok


def make_cfg(grid, steps, sgd_lr):
    return T.Config(
        phenotype='gated', channels=8, attn_dim=48, attn_heads=4, attn_layers=2,
        grid_size=grid, pop_size=16, n_eval=12, seed=0,
        cppn_num_hidden=48, gated_branches=3,
        eval_sgd_steps=steps, sgd_lr=sgd_lr,
    )


def episodes(tasks, cfg, grid, seed=0):
    import random
    store = T.ARCStore(grid_size=grid, tasks=[])
    rng = random.Random(seed)
    return [store.make_episode(t, cfg.max_demos, rng) for t in tasks]


def pure_eval(genome, cfg, steps, eps, pc, neat_cfg, device, optimizer, lr):
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
        T.sgd_adapt(phen, ep, steps, lr, device,
                    optimizer=optimizer, adam_lr=lr if optimizer == "adam" else None)
        accs.append(T.episode_accuracy(phen, ep, device))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100,
                    help="in-context steps (100 = ranking, 300 = final verdict)")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N tasks (0 = all). Smoke-test tool.")
    ap.add_argument("--sgd-only", action="store_true",
                    help="run only the SGD arm (skip Adam)")
    ap.add_argument("--adam-only", action="store_true",
                    help="run only the Adam arm (skip SGD)")
    args = ap.parse_args()

    grid, steps = args.grid, args.steps
    store = T.ARCStore.load(grid, seed=0)
    tasks = load_official_test(grid)
    if args.limit:
        tasks = tasks[:args.limit]
    ids = [str(getattr(t, "id", ""))[:8] for t in tasks]
    arms = [("sgd", 0.1), ("adam", 1e-2)]
    if args.sgd_only:
        arms = [("sgd", 0.1)]
    if args.adam_only:
        arms = [("adam", 1e-2)]

    cfg = make_cfg(grid, steps, 0.1)
    eps = episodes(tasks, cfg, grid, seed=0)
    total_test = 120
    print(f"Official ARC test set: {len(tasks)} tasks scored "
          f"({total_test} total; fit at grid {grid})")
    print(f"steps={steps}  arms={arms}  limit={args.limit}")
    print(f"task ids: {ids}")
    print()

    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()
    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]
    genomes = [
        # (label, pkl, fallback, [arms])
        ("grid15 adam-winner", "cppn_arc_adam100_pop16_e12_s0.pkl",
         random_g, arms),
        ("old best grid15", "sweep_lr10_e12_pop16_s0.pkl", random_g, arms),
        ("random", None, random_g, arms),
    ]
    import time
    t0 = time.time()
    for label, pkl, fallback, g_arms in genomes:
        genome = (pickle.load(open(pkl, "rb")) if pkl else fallback)
        for opt, lr in g_arms:
            a = pure_eval(genome, cfg, steps, eps, pc, neat_cfg, device, opt, lr)
            print(f"{label:22s} {opt:4s} steps={steps:3d} mean={np.mean(a):.4f}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
    print()
    print("total wall: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
