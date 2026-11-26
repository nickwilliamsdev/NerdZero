"""Budget-vs-expressivity test for the stuck set-13 tasks.

Re-score the best genome (gated96) on ONLY the low-scoring tasks (all
shape transforms: crop-to-content, 2x/3x/4x upscaling) at increasing
in-context budgets. If the stuck tasks climb as steps grow, the ceiling is
budget/optimizer-limited and the move is more/better in-context training.
If they stay flat, the ceiling is expressivity-limited and the move is a
phenotype / inductive-bias change.

Kept compact (5 stuck tasks x 3 budgets) so it finishes on the loaded GPU.
"""
import pickle

import numpy as np

import train_cppn_arc as T
import eval_set13_300steps as E

# (task_id) for the stuck tasks — see diag_stuck_tasks.py
STUCK = ["c8cbb738", "27f8ce4f", "cf5fd0ad", "d13f3404", "9f27f097"]
STEPS = [300, 600, 1000]
LR = 0.05


def score_task(genome, cfg, ep, pc, neat_cfg, device, steps):
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(pc.dim)], node_names=["w"],
    )
    phen = T.build_phenotype(cfg)
    substrate = T._substrate_for(ep, pc, cfg)
    import torch
    with torch.no_grad():
        W = T.encode(cppn, pc, substrate).to(device)
    T.load_param_vector(phen, pc, W)
    T.sgd_adapt(phen, ep, steps, LR, device)
    return T.episode_accuracy(phen, ep, device)


def main():
    import torch
    store = T.ARCStore.load(E.GRID, seed=0)
    cfg = E.make_cfg(LR)
    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()

    # build the full set-13 unseen episode list, then keep the stuck ones
    ids, eps = E.sample_unseen(store, cfg, E.N_TASKS, seed=E.SEED)
    by_id = {i: e for i, e in zip(ids, eps)}
    stuck_eps = [by_id[t] for t in STUCK if t in by_id]
    print(f"Stuck tasks to probe: {[t for t in STUCK if t in by_id]}")
    print(f"(base 300-step scores from the earlier run: "
          f"c8cbb738=0.4444 27f8ce4f=0.4815 cf5fd0ad=0.5556 "
          f"d13f3404=0.6111 9f27f097=0.6597)")
    print()

    with open("cppn_arc_gated96_pop4_e300.pkl", "rb") as f:
        genome = pickle.load(f)

    print(f"{'task':>10s}  " + "  ".join(f"{s:>5d}st" for s in STEPS))
    grid = {}
    for t, ep in zip(STUCK, stuck_eps):
        row = []
        for s in STEPS:
            a = score_task(genome, cfg, ep, pc, neat_cfg, device, s)
            row.append(a)
        grid[t] = row
        print(f"{t:>10s}  " + "  ".join(f"{x:5.4f}" for x in row))
    print()
    means = [np.mean(grid[t]) for t in STUCK]
    print("mean across stuck tasks per budget: " +
          "  ".join(f"{m:.4f}" for m in means))
    if means[-1] > means[0] + 0.03:
        print("=> BUDGET-limited (climbs with more steps)")
    else:
        print("=> EXPRESSIVITY-limited (flat with more steps)")


if __name__ == "__main__":
    main()
