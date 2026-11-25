"""Cost model: time one official ARC task, 300-step Adam adaptation, on CPU and
GPU, for the grid-15 winner. Report seconds per task so I can size the all-120
baseline and pick a device that doesn't starve the in-flight training run.
"""
import time
import pickle

import torch

import train_cppn_arc as T
from diag_adam_budget_grid30 import adapt_curve  # reuse the single-pass curve

GRID = 30


def main():
    store = T.ARCStore.load(GRID, seed=0)
    import arckit
    _, test_ds = arckit.load_data()
    task = test_ds.tasks[0]  # one official test task (fits grid 30)

    cfg = T.Config(phenotype='gated', channels=8, attn_dim=48, attn_heads=4,
                   attn_layers=2, grid_size=GRID, pop_size=16, n_eval=12,
                   seed=0, cppn_num_hidden=48, gated_branches=3,
                   eval_sgd_steps=300, sgd_lr=0.1,
                   in_context_optimizer='adam', adam_lr=0.01)
    import random
    rng = random.Random(0)
    ep = store.make_episode(task, cfg.max_demos, rng)

    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim, torch.device("cpu"),
                              schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    genome = pickle.load(open("cppn_arc_adam100_pop16_e12_s0.pkl", "rb"))
    cppn = T.create_torch_cppn(genome=genome, config=neat_cfg,
                               leaf_names=[f"c{i}" for i in range(pc.dim)],
                               node_names=["w"])

    for dev in [torch.device("cpu"), torch.device("cuda")]:
        phen = T.build_phenotype(cfg).to(dev)
        substrate = T._substrate_for(ep, pc, cfg)
        with torch.no_grad():
            W = T.encode(cppn, pc, substrate).to(dev)
        T.load_param_vector(phen, pc, W)
        t0 = time.time()
        curve = adapt_curve(phen, ep, dev, 300)
        dt = time.time() - t0
        print(f"{dev}: {dt:.1f}s per task (300-step Adam); "
              f"acc@100={curve.get(100, float('nan')):.3f} "
              f"acc@300={curve.get(300, float('nan')):.3f}")

    print()
    print("=> extrapolation: all-120 x 3 genomes x 300-step single-pass =")
    print("   (per-task above) x 120 x 3 seconds.")


if __name__ == "__main__":
    main()
