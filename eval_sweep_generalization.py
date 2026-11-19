"""Common-set generalization cross-eval for the same-env gated 30g sweep winners.

All four runs: gated phenotype (ch8, d48, S=15), cppn_hidden=48, pop 8,
30 gens, 6-task fixed eval.  Only sgd_lr (0.05 vs 0.1) and seed differ, and
each run was selected on its OWN fixed-eval set (seeds use different sets), so
in-sample gen-30 "best" is not comparable across runs.  Every genome here is
judged on ONE common unseen 8-task set (seed 7, excluding both eval sets) with
the identical protocol:
    encode genome -> in-context SGD (6 steps, the genome's native sgd_lr)
    -> cell accuracy, averaged over the 8 unseen tasks.
"""
import pickle
import random

import numpy as np

import train_cppn_arc as T

GRID = 15
SEED = 7
S0_EVAL = {'007bbfb7', 'f5c89df1', '1b8318e3', 'a6953f00', '3aa6fb7a', '00dbd492'}
S1_EVAL = {'8403a5d5', '94414823', 'bb43febb', 'f0f8a26d', '794b24be', '4c177718'}

RUNS = [
    # (label, pkl, native sgd_lr)
    ("ctrl     delta lr0.05 steps12 s0", "sweep_ctrl_delta48_e6.pkl", 0.05),
    ("lr10     delta lr0.10 steps12 s0", "sweep_lr10_gated48_e8.pkl", 0.10),
    ("steps24  delta lr0.10 steps24 s0", "sweep_steps24_lr10_e8.pkl", 0.10),
    ("ctrl     delta lr0.05 steps12 s1", "sweep_ctrl_delta48_s1.pkl", 0.05),
    ("lr10     delta lr0.10 steps12 s1", "sweep_lr10_gated48_s1.pkl", 0.10),
]


def make_cfg(sgd_lr):
    return T.Config(
        phenotype='gated', channels=8, attn_dim=48, attn_heads=4, attn_layers=2,
        grid_size=GRID, pop_size=8, n_eval=6, seed=SEED,
        fixed_eval=True, cppn_num_hidden=48,
        eval_sgd_steps=6, sgd_lr=sgd_lr,
    )


def sample_unseen(store, cfg, n=8, seed=7):
    rng = random.Random(seed + 424242)
    exclude = S0_EVAL | S1_EVAL
    idx = [i for i, t in enumerate(store.eligible) if str(getattr(t, "id", "")) not in exclude]
    rng.shuffle(idx)
    ids, eps = [], []
    for i in idx[:n]:
        ep = store.make_episode(store.eligible[i], cfg.max_demos, rng)
        ids.append(ep.task_id)
        eps.append(ep)
    return ids, eps


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
        T.sgd_adapt(phen, ep, cfg.eval_sgd_steps, cfg.sgd_lr, device)
        accs.append(T.episode_accuracy(phen, ep, device))
    return accs


def main():
    import sys, torch
    unseen_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    store = T.ARCStore.load(GRID, seed=0)  # seed-0 store, as in all training runs
    template_cfg = make_cfg(0.05)
    template = T.build_phenotype(template_cfg)
    pc = T.build_param_coords(template, template_cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(template_cfg)
    device = template_cfg.resolve_device()

    ids, eps = sample_unseen(store, template_cfg, n_tasks,
                             seed=unseen_seed)
    print(f"Common unseen {n_tasks}-task set (seed {unseen_seed}): {ids}")
    print()

    rows = []
    for label, pkl, lr in RUNS:
        with open(pkl, "rb") as f:
            genome = pickle.load(f)
        cfg = make_cfg(lr)
        a = pure_eval(genome, cfg, eps, pc, neat_cfg, device)
        per = [round(x, 4) for x in a]
        print(f"{label}: mean={np.mean(a):.4f}  per-task={per}")
        print(f"  genome complexity: {len(genome.nodes)} nodes / {len(genome.connections)} links")
        rows.append((label, float(np.mean(a))))
    print()
    rows.sort(key=lambda r: -r[1])
    print("Ranking on common unseen set:")
    for i, (l, m) in enumerate(rows, 1):
        print(f"  {i}. {l}  {m:.4f}")


if __name__ == "__main__":
    main()
