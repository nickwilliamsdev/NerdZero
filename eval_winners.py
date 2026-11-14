"""Controlled cross-evaluation of the two fixed-eval deltanet winners.

Both winners are judged on BOTH task sets (4 and 8 tasks) with the same
protocol: pure genome evaluation (no outer meta-training phase):
    for each eval task: encode genome -> in-context SGD -> cell accuracy.
This isolates 'how good is this genome as a task solver' so the
4-task-trained vs 8-task-trained comparison is not confounded by which
tasks each was selected on.
"""
import random
import sys
import torch
import numpy as np

import train_cppn_arc as T

GRID = 10
SEED = 0
IDS4 = ['b6afb2da', '941d9a10', '10fcaaa3', 'd511f180']
IDS8 = IDS4 + ['bd4472b8', '6e82a1ae', '4cd1b7b2', '03560426']


def make_cfg(n_eval):
    return T.Config(
        phenotype='deltanet', attn_dim=24, attn_heads=2, attn_layers=1,
        grid_size=GRID, pop_size=4, n_eval=n_eval, seed=SEED,
        fixed_eval=True, cppn_num_hidden=48,
    )


def rebuild_eps(store, n_eval):
    """Reproduce the exact fixed-eval episodes (same seed + call order as the run)."""
    cfg = make_cfg(n_eval)
    rng = random.Random(SEED + 99991)
    eps = T.sample_fixed_eval(store, cfg, rng)
    got = [e.task_id for e in eps]
    want = IDS4 if n_eval == 4 else IDS8
    assert got == want, f"episode mismatch: {got} != {want}"
    return eps


def pure_eval(genome, cfg, eps, pc, neat_cfg, device):
    """Encode genome -> in-context SGD on each task -> per-task accuracy."""
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
    cfg4 = make_cfg(4)
    cfg8 = make_cfg(8)
    device = cfg4.resolve_device()

    # pc / neat_cfg: identical for both (same phenotype, dim, pop, hidden).
    template = T.build_phenotype(cfg4)
    n_types = len({T._layer_type(n) for n, _ in template.named_parameters()})
    schema = "layer" if n_types >= 4 else "flat"  # mirrors make_evaluator's auto
    assert schema == "flat", f"expected flat schema, got {schema}"
    pc = T.build_param_coords(template, cfg4.substrate_dim, torch.device("cpu"),
                              schema=schema)
    neat_cfg = T.build_neat_config(cfg4)

    eps4 = rebuild_eps(store, 4)
    eps8 = rebuild_eps(store, 8)

    import pickle
    for name, pkl in [
        ("trained on 4 tasks", "cppn_arc_deltanet_cppn48_e4.pkl"),
        ("trained on 8 tasks", "cppn_arc_deltanet_cppn48_e8.pkl"),
    ]:
        with open(pkl, "rb") as f:
            genome = pickle.load(f)
        a4 = pure_eval(genome, cfg4, eps4, pc, neat_cfg, device)
        a8 = pure_eval(genome, cfg8, eps8, pc, neat_cfg, device)
        print(f"\n== winner {name} (genome complexity "
              f"{len(genome.nodes)} nodes / {len(genome.connections)} links) ==")
        print(f"  4-task set: per-task {[round(a,4) for a in a4]}  mean={np.mean(a4):.4f}")
        print(f"  8-task set: per-task {[round(a,4) for a in a8]}  mean={np.mean(a8):.4f}")


if __name__ == "__main__":
    main()
