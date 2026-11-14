"""Evaluate the damped gate-blend winner (gate_meta=task, scale=0.5) vs the
delta baseline vs the best deltanet on (a) the fixed 8/4-task eval protocol
and (b) 10 independent random 8-task splits (the trustworthy generalization
test).  Also prints the post-adaptation gate to check routing vs over-commit.
"""
import pickle
import random

import numpy as np
import torch

import train_cppn_arc as T

GRID = 10
SEED = 0
IDS4 = ['b6afb2da', '941d9a10', '10fcaaa3', 'd511f180']
IDS8 = IDS4 + ['bd4472b8', '6e82a1ae', '4cd1b7b2', '03560426']


def make_cfg(phen, dim, n_eval, seed=SEED):
    return T.Config(phenotype=phen, attn_dim=dim, attn_heads=2, attn_layers=1,
                    grid_size=GRID, pop_size=4, n_eval=n_eval, seed=seed,
                    fixed_eval=True, sgd_steps=24, eval_sgd_steps=6)


def build_ctx(cfg):
    template = T.build_phenotype(cfg)
    n_types = len({T._layer_type(n) for n, _ in template.named_parameters()})
    schema = "layer" if n_types >= 4 else "flat"
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema=schema)
    return pc, T.build_neat_config(cfg)


def fixed_eps(store, phen, dim, n):
    cfg = make_cfg(phen, dim, n)
    rng = random.Random(SEED + 99991)
    eps = T.sample_fixed_eval(store, cfg, rng)
    want = IDS4 if n == 4 else IDS8
    assert [e.task_id for e in eps] == want
    return eps


def pure_eval(genome, cfg, eps, pc, neat_cfg, device, show_gate=False):
    cppn = T.create_torch_cppn(genome=genome, config=neat_cfg,
                               leaf_names=[f"c{i}" for i in range(pc.dim)],
                               node_names=["w"])
    phen = T.build_phenotype(cfg)
    accs, gates = [], []
    for ep in eps:
        substrate = T._substrate_for(ep, pc, cfg)
        with torch.no_grad():
            W = T.encode(cppn, pc, substrate).to(device)
        T.load_param_vector(phen, pc, W)
        T.sgd_adapt(phen, ep, cfg.eval_sgd_steps, cfg.sgd_lr, device)
        accs.append(T.episode_accuracy(phen, ep, device))
        if show_gate:
            g = torch.softmax(phen.gate.detach(), dim=0)
            gates.append([round(float(v), 2) for v in g.tolist()])
    return accs, gates


def main():
    store = T.ARCStore.load(GRID, seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "gated_task05": ("cppn_arc_gated_task05_e8.pkl", make_cfg("gated", 24, 8)),
        "gated_delta":  ("cppn_arc_gated_sgd24_e8.pkl", make_cfg("gated", 24, 8)),
        "dn24_sgd24":   ("cppn_arc_dn24_sgd24_e8.pkl", make_cfg("deltanet", 24, 8)),
    }
    ready = {}
    for label, (pkl, cfg) in models.items():
        pc, neat_cfg = build_ctx(cfg)
        genome = pickle.load(open(pkl, "rb"))
        ready[label] = (genome, cfg, pc, neat_cfg)

    # ---- fixed-eval protocol (8 + 4 task sets) ----
    print("== fixed-eval (seed 0, eval_sgd=6) ==")
    for label, (genome, cfg, pc, neat_cfg) in ready.items():
        phen, dim = ("gated", 24) if label.startswith("gated") else ("deltanet", 24)
        eps8 = fixed_eps(store, phen, dim, 8)
        eps4 = fixed_eps(store, phen, dim, 4)
        a8, g8 = pure_eval(genome, cfg, eps8, pc, neat_cfg, device,
                           show_gate=(label != "dn24_sgd24"))
        a4, _ = pure_eval(genome, cfg, eps4, pc, neat_cfg, device)
        print(f"  {label:<14} 4-task={np.mean(a4):.4f}  8-task={np.mean(a8):.4f}")
        if label != "dn24_sgd24":
            for tid, g in zip((e.task_id for e in eps8), g8):
                print(f"      {tid} gate(dn,at,conv)={g}")

    # ---- 10-split generalization ----
    print("\n== 10-split generalization (random 8-task sets) ==")
    print(f"{'split':>6}", *[f"{l:>13}" for l in ready])
    results = {l: [] for l in ready}
    for sp in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]:
        row = []
        for label, (genome, cfg, pc, neat_cfg) in ready.items():
            rng = random.Random(sp)
            eps = T.sample_fixed_eval(store, cfg, rng)
            m = float(np.mean(pure_eval(genome, cfg, eps, pc, neat_cfg, device)[0]))
            results[label].append(m)
            row.append(m)
        print(f"{sp:>6}" + "".join(f"{m:>13.4f}" for m in row), flush=True)
    print("\nmean +- std over 10 splits:")
    for label, vals in results.items():
        print(f"  {label:<14} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")


if __name__ == "__main__":
    main()
