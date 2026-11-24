"""Cross-eval on the OFFICIAL ARC test set (the 9 of 120 tasks that fit 15x15).

Everything so far has been judged on unseen subsets of the ARC *train* set
(set-13 etc.) — "held-out within train". This is the first yardstick on the
actual ARC test split. Facts (verified 2026-11-23): the official test set has
120 tasks, only 9 have all grids <=15x15 (111 are 16x16..30x30, unusable at the
current grid_size=15). Those 9 are the full official-test eval possible today;
raising grid_size to 30 would be the move to judge the other 111.

Protocol per task: encode genome on the task's dynamic substrate -> 300 in-
context steps -> cell-wise accuracy on the unpadded target region. Both the
established SGD protocol (lr 0.1, clip 10) and the new Adam protocol
(lr 0.01, unclipped) are run, so the official-test number for each genome is
directly comparable to the set-13 numbers.

Genomes:
  old best pop16 e12 s0   sweep_lr10_e12_pop16_s0.pkl
  gated96 pop4 e300       cppn_arc_gated96_pop4_e300.pkl
  random                  fresh random genome
(Add the Adam-run winner here once it exists.)
"""
import pickle

import numpy as np
import torch

import train_cppn_arc as T

GRID = 15
STEPS = 300


def load_official_test():
    import arckit
    _, test_ds = arckit.load_data()
    import numpy as np
    ok = []
    for t in test_ds.tasks:
        fit = True
        for grp in ("train", "test"):
            for x, y in getattr(t, grp, []):
                x, y = np.asarray(x), np.asarray(y)
                if (x.shape[0] > GRID or x.shape[1] > GRID or
                        y.shape[0] > GRID or y.shape[1] > GRID):
                    fit = False
                    break
            if not fit:
                break
        if fit:
            ok.append(t)
    return ok


def make_cfg(sgd_lr):
    return T.Config(
        phenotype='gated', channels=8, attn_dim=48, attn_heads=4, attn_layers=2,
        grid_size=GRID, pop_size=16, n_eval=12, seed=0,
        fixed_eval=True, cppn_num_hidden=48, gated_branches=3,
        eval_sgd_steps=STEPS, sgd_lr=sgd_lr,
    )


def episodes(tasks, cfg, seed=0):
    import random
    store = T.ARCStore(grid_size=GRID, tasks=[])
    rng = random.Random(seed)
    return [store.make_episode(t, cfg.max_demos, rng) for t in tasks]


def pure_eval(genome, cfg, eps, pc, neat_cfg, device, optimizer, lr):
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
        T.sgd_adapt(phen, ep, STEPS, lr, device,
                    optimizer=optimizer, adam_lr=lr if optimizer == "adam" else None)
        accs.append(T.episode_accuracy(phen, ep, device))
    return accs


def main():
    store = T.ARCStore.load(GRID, seed=0)
    tasks = load_official_test()
    ids = [str(getattr(t, "id", "")) for t in tasks]
    cfg = make_cfg(0.1)
    eps = episodes(tasks, cfg, seed=0)
    print(f"Official ARC test set: {len(tasks)} of 120 tasks fit {GRID}x{GRID}")
    print(f"task ids: {ids}")
    print(f"(the other {120 - len(tasks)} have grids 16..30 x 16..30; needs "
          f"grid_size=30 to judge)")
    print()

    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()
    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]
    genomes = [
        ("old best pop16 e12 s0", "sweep_lr10_e12_pop16_s0.pkl", random_g),
        ("gated96 pop4 e300", "cppn_arc_gated96_pop4_e300.pkl", random_g),
        ("random", None, random_g),
    ]
    import time
    t0 = time.time()
    for label, pkl, fallback in genomes:
        genome = (pickle.load(open(pkl, "rb")) if pkl else fallback)
        for opt, lr in [("sgd", 0.1), ("adam", 1e-2)]:
            a = pure_eval(genome, cfg, eps, pc, neat_cfg, device, opt, lr)
            print(f"{label:22s} {opt:4s} mean={np.mean(a):.4f}  "
                  f"per-task={['%.3f' % x for x in a]}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
    print()
    print("total wall: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
