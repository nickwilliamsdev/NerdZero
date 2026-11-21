"""300-step in-context cross-eval on the common unseen 16-task set (seed 13).

Re-creates the yardstick from eval_sweep_gen_set13_300steps.txt (that script
was run ad-hoc and never committed): every genome is judged on the SAME
unseen 16-task set with 300 in-context SGD steps, i.e. the phenotype is
trained until it "overfits" on each task, so the measurement is a ceiling
test on the CPPN's initial weights, not on the eval budget.

Genomes:
  - ctrl e6 s0       sweep_ctrl_delta48_e6.pkl   (baseline, lr 0.05)
  - gated96 pop4 s0  cppn_arc_gated96_pop4_e300.pkl (new run, lr 0.05)

Protocol per task (identical for every genome):
  encode genome on the task's dynamic substrate -> 300 SGD steps (sgd_lr)
  -> cell-wise accuracy on the unpadded target region.
"""
import pickle
import random

import numpy as np

import train_cppn_arc as T

GRID = 15
SEED = 13          # unseen-set seed (must match eval_sweep_gen_set13_300steps.txt)
N_TASKS = 16
SGD_STEPS = 300

# Same exclusion union as eval_sweep_generalization.py so "unseen" means
# "never used by any training/eval set in this line".
S0_EVAL = {'007bbfb7', 'f5c89df1', '1b8318e3', 'a6953f00', '3aa6fb7a', '00dbd492'}
S1_EVAL = {'8403a5d5', '94414823', 'bb43febb', 'f0f8a26d', '794b24be', '4c177718'}
E12_EXTRA = {'575b1a71', '1b60fb0c', '494ef9d7', '941d9a10', '03560426', 'e9b4f6fc'}
E12_S1 = {'3618c87e', '642248e4', 'aaef0977', 'a5f85a15', 'b7256dcd', '1c02dbbe'}

RUNS = [
    # (label, pkl, native sgd_lr)
    ("ctrl e6 s0 (base)", "sweep_ctrl_delta48_e6.pkl", 0.05),
    ("gated96 pop4 e300 s0", "cppn_arc_gated96_pop4_e300.pkl", 0.05),
]


def make_cfg(sgd_lr):
    return T.Config(
        phenotype='gated', channels=8, attn_dim=48, attn_heads=4, attn_layers=2,
        grid_size=GRID, pop_size=8, n_eval=6, seed=SEED,
        fixed_eval=True, cppn_num_hidden=48, gated_branches=3,
        eval_sgd_steps=SGD_STEPS, sgd_lr=sgd_lr,
    )


def sample_unseen(store, cfg, n=N_TASKS, seed=SEED):
    rng = random.Random(seed + 424242)
    exclude = S0_EVAL | S1_EVAL | E12_EXTRA | E12_S1
    idx = [i for i, t in enumerate(store.eligible)
           if str(getattr(t, "id", "")) not in exclude]
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
    store = T.ARCStore.load(GRID, seed=0)  # seed-0 store, as in all training runs
    template_cfg = make_cfg(0.05)
    template = T.build_phenotype(template_cfg)
    pc = T.build_param_coords(template, template_cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(template_cfg)
    device = template_cfg.resolve_device()

    ids, eps = sample_unseen(store, template_cfg, N_TASKS, seed=SEED)
    print(f"Common unseen {N_TASKS}-task set (seed {SEED}, {SGD_STEPS}-step "
          f"in-context SGD): {ids}")
    print()

    rows = []
    for label, pkl, lr in RUNS:
        with open(pkl, "rb") as f:
            genome = pickle.load(f)
        cfg = make_cfg(lr)
        a = pure_eval(genome, cfg, eps, pc, neat_cfg, device)
        per = [round(x, 4) for x in a]
        print(f"{label}: mean={np.mean(a):.4f}  per-task={per}")
        print(f"  genome complexity: {len(genome.nodes)} nodes / "
              f"{len(genome.connections)} links")
        rows.append((label, float(np.mean(a))))
        print(f"  {label} done", flush=True)
    print()
    rows.sort(key=lambda r: -r[1])
    print("Ranking on common unseen set (300-step in-context SGD):")
    for i, (l, m) in enumerate(rows, 1):
        print(f"  {i}. {l}  {m:.4f}")


if __name__ == "__main__":
    main()
