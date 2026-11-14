"""Generalization check: evaluate the pure_attn (kqv, 4-output CPPN) winner
against the gated and deltanet winners on 10 INDEPENDENT random 8-task
splits (same grid/protocol).  The fixed-eval numbers measure luck on one
held-out set; this measures how the genomes actually generalize across ARC
tasks.
"""
import pickle
import random

import numpy as np
import torch

import train_cppn_arc as T

GRID = 10
SEED = 0


def make_cfg(phen, dim, sgd_steps, seed):
    return T.Config(
        phenotype=phen, attn_dim=dim, attn_heads=2, attn_layers=1,
        grid_size=GRID, pop_size=4, n_eval=8, seed=seed,
        fixed_eval=True, sgd_steps=sgd_steps, eval_sgd_steps=6,
        cppn_num_outputs=4 if phen == "pure_attn" else 1,
    )


def sample_split(store, cfg, split_seed):
    rng = random.Random(split_seed)
    eps = T.sample_fixed_eval(store, cfg, rng)
    return eps


def build_ctx(cfg):
    template = T.build_phenotype(cfg)
    n_types = len({T._layer_type(n) for n, _ in template.named_parameters()})
    schema = "kqv" if cfg.phenotype == "pure_attn" else (
        "layer" if n_types >= 4 else "flat")
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema=schema)
    neat_cfg = T.build_neat_config(cfg)
    return pc, neat_cfg


def pure_eval(genome, cfg, eps, pc, neat_cfg, device):
    n_out = cfg.cppn_num_outputs
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(pc.dim)],
        node_names=["k", "q", "v", "rest"][:n_out],
        output_mode="tensor" if n_out > 1 else "auto",
    )
    phen = T.build_phenotype(cfg)
    accs = []
    for ep in eps:
        substrate = T._substrate_for(ep, pc, cfg)
        with torch.no_grad():
            W = T.encode(cppn, pc, substrate).to(device)
        T.load_param_vector(phen, pc, W)
        T.sgd_adapt(phen, ep, cfg.eval_sgd_steps, cfg.sgd_lr, device)
        accs.append(T.episode_accuracy(phen, ep, device))
    return np.mean(accs)


def main():
    store = T.ARCStore.load(GRID, seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genomes = {
        "pureattn24_e8": ("cppn_arc_pureattn24_e8.pkl",
                          make_cfg("pure_attn", 24, 24, SEED)),
        "gated_sgd24_e8": ("cppn_arc_gated_sgd24_e8.pkl",
                           make_cfg("gated", 24, 24, SEED)),
        "dn24_sgd24_e8": ("cppn_arc_dn24_sgd24_e8.pkl",
                          make_cfg("deltanet", 24, 24, SEED)),
    }
    loaded = {}
    for label, (pkl, cfg) in genomes.items():
        pc, neat_cfg = build_ctx(cfg)
        with open(pkl, "rb") as f:
            genome = pickle.load(f)
        loaded[label] = (genome, cfg, pc, neat_cfg)

    print(f"{'split_seed':>10}", *[f"{l:>16}" for l in loaded])
    results = {l: [] for l in loaded}
    for split_seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]:
        row = []
        for label, (genome, cfg, pc, neat_cfg) in loaded.items():
            eps = sample_split(store, cfg, split_seed)
            m = pure_eval(genome, cfg, eps, pc, neat_cfg, device)
            results[label].append(m)
            row.append(m)
        print(f"{split_seed:>10}" + "".join(f"{m:>16.4f}" for m in row),
              flush=True)

    print("\nmean +- std over 10 random splits:")
    for label, vals in results.items():
        print(f"  {label:<16} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")


if __name__ == "__main__":
    main()
