"""One-shot cross-eval of a single genome under the established protocol
(fixed 8-task set, seed 0, grid 10, eval_sgd_steps=6) so it can be dropped
into the same table as eval_all_winners.py output."""
import sys

import numpy as np
import torch

import eval_all_winners as E

PKL = sys.argv[1] if len(sys.argv) > 1 else "cppn_arc_gated_perparam_e8.pkl"
PHEN = sys.argv[2] if len(sys.argv) > 2 else "gated"
DIM = int(sys.argv[3]) if len(sys.argv) > 3 else 24
SGD = int(sys.argv[4]) if len(sys.argv) > 4 else 24


def main():
    store = E.T.ARCStore.load(E.GRID, seed=E.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = E.make_cfg(PHEN, DIM, 8, SGD, 6)
    template = E.T.build_phenotype(cfg)
    n_types = len({E.T._layer_type(n) for n, _ in template.named_parameters()})
    schema = "layer" if n_types >= 4 else "flat"
    pc = E.T.build_param_coords(template, cfg.substrate_dim,
                                torch.device("cpu"), schema=schema)
    neat_cfg = E.T.build_neat_config(cfg)
    eps8 = E.rebuild_eps(store, 8, cfg)
    eps4 = E.rebuild_eps(store, 4, E.make_cfg(PHEN, DIM, 4, SGD, 6))

    import pickle
    genome = pickle.load(open(PKL, "rb"))
    a4 = E.pure_eval(genome, cfg, eps4, pc, neat_cfg, device)
    a8 = E.pure_eval(genome, cfg, eps8, pc, neat_cfg, device)
    print(f"{PKL}: nodes={len(genome.nodes)} links={len(genome.connections)} "
          f"schema={schema}")
    print(f"  4-task: {np.mean(a4):.4f}  per={[round(x, 3) for x in a4]}")
    print(f"  8-task: {np.mean(a8):.4f}  per={[round(x, 3) for x in a8]}")


if __name__ == "__main__":
    main()
