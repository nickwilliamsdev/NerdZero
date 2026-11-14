"""Evaluate the gated-phenotype winner (cppn_arc_gated_sgd24_e8.pkl) on the
SAME fixed-eval protocol as eval_winners.py (8 tasks, pure genome eval:
encode -> in-context SGD -> accuracy), plus its 4-task subset, so it is
directly comparable to the two deltanet cppn48 winners.
"""
import pickle
import random
import torch

import train_cppn_arc as T

GRID = 10
SEED = 0
IDS4 = ['b6afb2da', '941d9a10', '10fcaaa3', 'd511f180']
IDS8 = IDS4 + ['bd4472b8', '6e82a1ae', '4cd1b7b2', '03560426']


def make_cfg(n_eval, sgd_steps=24):
    return T.Config(
        phenotype='gated', attn_dim=24, attn_heads=2, attn_layers=1,
        grid_size=GRID, pop_size=4, n_eval=n_eval, seed=SEED,
        fixed_eval=True,
        sgd_steps=sgd_steps,          # phenotype SGD steps (run used 24)
        eval_sgd_steps=sgd_steps,
    )


def rebuild_eps(store, n_eval):
    cfg = make_cfg(n_eval)
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
    accs, gates = [], []
    for ep in eps:
        substrate = T._substrate_for(ep, pc, cfg)
        with torch.no_grad():
            W = T.encode(cppn, pc, substrate).to(device)
        T.load_param_vector(phen, pc, W)
        T.sgd_adapt(phen, ep, cfg.eval_sgd_steps, cfg.sgd_lr, device)
        accs.append(T.episode_accuracy(phen, ep, device))
        # gate vector is the 'gate' parameter of the gated phenotype
        g = torch.softmax(phen.gate.detach(), dim=0)
        gates.append([round(float(v), 3) for v in g.tolist()])
    return accs, gates


def main():
    store = T.ARCStore.load(GRID, seed=SEED)
    cfg = make_cfg(8)
    device = cfg.resolve_device()

    template = T.build_phenotype(cfg)
    n_types = len({T._layer_type(n) for n, _ in template.named_parameters()})
    schema = "layer" if n_types >= 4 else "flat"
    pc = T.build_param_coords(template, cfg.substrate_dim, torch.device("cpu"),
                              schema=schema)
    neat_cfg = T.build_neat_config(cfg)
    print(f"gated phenotype: params={pc.flat_size} layer_types={n_types} "
          f"schema={schema}")

    eps4 = rebuild_eps(store, 4)
    eps8 = rebuild_eps(store, 8)

    with open("cppn_arc_gated_sgd24_e8.pkl", "rb") as f:
        genome = pickle.load(f)

    import numpy as np
    for label, eps in [("4-task set", eps4), ("8-task set", eps8)]:
        accs, gates = pure_eval(genome, cfg, eps, pc, neat_cfg, device)
        print(f"\n== gated winner (50? nodes={len(genome.nodes)} / "
              f"links={len(genome.connections)}) {label} ==")
        print(f"  per-task {[round(a, 4) for a in accs]}  mean={np.mean(accs):.4f}")
        print(f"  gates(dn,attn,conv) after adapt: {gates}")


if __name__ == "__main__":
    main()
