"""Cross-evaluate ALL recent e8 winners under the identical fixed-eval protocol
(8 tasks, seed 0, grid 10, pure genome eval: encode -> in-context SGD -> acc),
plus each winner's 4-task subset.  One table, no confounds: same tasks, same
SGD steps, same seeds -- only the trained genome (and its phenotype
architecture) differs.
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

# (label, pkl, phenotype, attn_dim, sgd_steps, eval_sgd_steps)
# NOTE: the "cppn48" in deltanet_cppn48_* is the CPPN hidden size, NOT the
# phenotype dim.  eval_winners.py evaluated those genomes at attn_dim=24 with
# eval_sgd_steps=6, so dim=24, eval_sgd=6 below.  The clean, config-matched
# comparison rows are dn24_sgd24, dn48_sgd24, gated (their dims come from the
# filename encoding).  eval_sgd_steps=6 everywhere to match the established
# pure-eval protocol.
WINNERS = [
    ("dn24_sgd24_e8", "cppn_arc_dn24_sgd24_e8.pkl", "deltanet", 24, 24, 6),
    ("dn48_sgd24_e8", "cppn_arc_dn48_sgd24_e8.pkl", "deltanet", 48, 24, 6),
    ("gated_sgd24_e8", "cppn_arc_gated_sgd24_e8.pkl", "gated", 24, 24, 6),
    # references from eval_winners.py (deltanet cppn48, dim 24, 12 sgd steps,
    # eval_sgd_steps=6):
    ("dn48_cppn_e4", "cppn_arc_deltanet_cppn48_e4.pkl", "deltanet", 24, 12, 6),
    ("dn48_cppn_e8", "cppn_arc_deltanet_cppn48_e8.pkl", "deltanet", 24, 12, 6),
]


def make_cfg(phen, dim, n_eval, sgd_steps, eval_sgd_steps):
    return T.Config(
        phenotype=phen, attn_dim=dim, attn_heads=2, attn_layers=1,
        grid_size=GRID, pop_size=4, n_eval=n_eval, seed=SEED,
        fixed_eval=True, sgd_steps=sgd_steps, eval_sgd_steps=eval_sgd_steps,
    )


def rebuild_eps(store, n_eval, cfg):
    rng = random.Random(SEED + 99991)
    eps = T.sample_fixed_eval(store, cfg, rng)
    got = [e.task_id for e in eps]
    want = IDS4 if n_eval == 4 else IDS8
    assert got == want, f"episode mismatch: {got} != {want}"
    return eps


def pure_eval(genome, cfg, eps, pc, neat_cfg, device):
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(pc.dim)], node_names=["w"],
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
    return accs


def main():
    store = T.ARCStore.load(GRID, seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'winner':<22} {'n_nodes':>7} {'4-task':>8} {'8-task':>8}")
    rows = []
    for label, pkl, phen, dim, sgd, esgd in WINNERS:
        cfg = make_cfg(phen, dim, 8, sgd, esgd)
        template = T.build_phenotype(cfg)
        n_types = len({T._layer_type(n) for n, _ in template.named_parameters()})
        schema = "layer" if n_types >= 4 else "flat"
        pc = T.build_param_coords(template, cfg.substrate_dim,
                                  torch.device("cpu"), schema=schema)
        neat_cfg = T.build_neat_config(cfg)
        eps8 = rebuild_eps(store, 8, cfg)
        eps4 = rebuild_eps(store, 4, make_cfg(phen, dim, 4, sgd, esgd))

        with open(pkl, "rb") as f:
            genome = pickle.load(f)
        a4 = pure_eval(genome, cfg, eps4, pc, neat_cfg, device)
        a8 = pure_eval(genome, cfg, eps8, pc, neat_cfg, device)
        m4, m8 = float(np.mean(a4)), float(np.mean(a8))
        rows.append((label, m4, m8))
        print(f"{label:<22} {len(genome.nodes):>7} {m4:>8.4f} {m8:>8.4f}"
              f"   per-task8={[round(a, 3) for a in a8]}")

    print("\nsummary (higher better):")
    for label, m4, m8 in sorted(rows, key=lambda r: -r[2]):
        print(f"  {label:<22} 4-task={m4:.4f}  8-task={m8:.4f}")


if __name__ == "__main__":
    main()
