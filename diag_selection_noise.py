"""Selection-noise probe: how much of the per-gen fitness is episode noise?

The in-flight grid-30 run selects on a rotating 12-task set, one episode per
task (random demo split per generation). If the fitness std across different
episode draws is comparable to the between-genome fitness differences, NEAT
is ranking on noise.

Method: 12 grid-30 train tasks (seed 42, disjoint from the budget probe's
seed-13 set). Score the grid-15 Adam winner 4 times with 4 independent
episode seeds (each: 12 tasks x 300-step single-pass, read at 100 steps).
Report the mean+std of the 100-step score across draws. That std is the
noise floor for the selection signal.
"""
import pickle
import random

import numpy as np
import torch

import train_cppn_arc as T
from diag_adam_budget_grid30 import adapt_curve

GRID = 30
N_TASKS = 12
DRAWS = 4
READ = 100


def main():
    store = T.ARCStore.load(GRID, seed=0)
    base_rng = random.Random(42)
    idx = base_rng.sample(list(range(len(store.eligible))), N_TASKS)
    tasks = [store.eligible[i] for i in idx]
    print(f"12 grid-30 tasks (seed 42): {[str(getattr(t, 'id', ''))[:8] for t in tasks]}")

    cfg = T.Config(phenotype='gated', channels=8, attn_dim=48, attn_heads=4,
                   attn_layers=2, grid_size=GRID, pop_size=16, n_eval=12,
                   seed=0, cppn_num_hidden=48, gated_branches=3,
                   eval_sgd_steps=READ, sgd_lr=0.1,
                   in_context_optimizer='adam', adam_lr=0.01)
    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim, torch.device("cpu"),
                              schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()
    genome = pickle.load(open("cppn_arc_adam100_pop16_e12_s0.pkl", "rb"))
    cppn = T.create_torch_cppn(genome=genome, config=neat_cfg,
                               leaf_names=[f"c{i}" for i in range(pc.dim)],
                               node_names=["w"])

    scores = []
    for d in range(DRAWS):
        rng = random.Random(1000 + d)  # independent episode draws
        accs = []
        for t in tasks:
            ep = store.make_episode(t, cfg.max_demos, rng)
            phen = T.build_phenotype(cfg)
            substrate = T._substrate_for(ep, pc, cfg)
            with torch.no_grad():
                W = T.encode(cppn, pc, substrate).to(device)
            T.load_param_vector(phen, pc, W)
            curve = adapt_curve(phen, ep, device, 300)
            accs.append(curve[READ])
        m = float(np.mean(accs))
        scores.append(m)
        print(f"draw {d+1}/{DRAWS}: {READ}-step mean over 12 tasks = {m:.4f} "
              f"(per-task min {min(accs):.3f})", flush=True)
    arr = np.array(scores)
    print()
    print(f"selection noise floor (same genome, different episode draws):")
    print(f"  mean {arr.mean():.4f}  std {arr.std(ddof=1):.4f}  "
          f"range {arr.min():.4f}..{arr.max():.4f}")
    print("  compare against the between-genome fitness spread of the")
    print("  in-flight run (mean ~0.80-0.87, stdev ~0.05 per gen log).")


if __name__ == "__main__":
    main()
