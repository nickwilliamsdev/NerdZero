"""Is the 300-step ceiling limited by the in-context OPTIMIZER (SGD) or by the
phenotype's capability?

The per-task 300-step scores for the best genome (gated96) show ~5 tasks stuck
at 0.42-0.62 while others hit 1.0. If plain SGD (lr 0.05, grad-clip 1.0) is
underfitting / unstable on the hard tasks, a better in-context optimizer
(Adam) should push those stuck tasks higher and raise the mean.

For each genome, at 300 in-context steps, compare per task:
  SGD  = current clipped SGD (sgd_adapt)
  Adam = Adam, same lr scale, no grad-clip
on the unseen 16-task set (seed 13).

If Adam clearly beats SGD on the stuck tasks, the ceiling is SGD-limited and
the move is to meta-select with Adam in-context adaptation.
"""
import pickle

import numpy as np
import torch
import torch.nn.functional as F

import train_cppn_arc as T
import eval_set13_300steps as E

STEPS = 300


def adapt(phen, ep, steps, lr, device, optimizer):
    phen.train()
    if optimizer == "sgd":
        opt = torch.optim.SGD(phen.parameters(), lr=lr)
        clip = True
    elif optimizer == "adam":
        opt = torch.optim.Adam(phen.parameters(), lr=lr)
        clip = False
    elif optimizer == "adam1e3":
        opt = torch.optim.Adam(phen.parameters(), lr=lr * 0.02)  # 0.001
        clip = False
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    epd = T._ep_on_device(ep, device)
    target = (epd.target.squeeze(0) * 9.0).round().long().unsqueeze(0)
    for _ in range(steps):
        opt.zero_grad()
        pred = phen(epd)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(phen.parameters(), 1.0)
        opt.step()
    phen.eval()


def per_task(genome, cfg, eps, pc, neat_cfg, device, optimizer):
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
        adapt(phen, ep, STEPS, cfg.sgd_lr, device, optimizer)
        accs.append(T.episode_accuracy(phen, ep, device))
    return accs


def main():
    store = T.ARCStore.load(E.GRID, seed=0)
    cfg = E.make_cfg(0.05)
    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()

    ids, eps = E.sample_unseen(store, cfg, E.N_TASKS, seed=E.SEED)
    print(f"Unseen {E.N_TASKS}-task set (seed {E.SEED}), {STEPS}-step in-context "
          f"SGD vs Adam:")
    print(f"tasks: {ids}")
    print()

    with open("cppn_arc_gated96_pop4_e300.pkl", "rb") as f:
        g96 = pickle.load(f)
    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]

    grids = {}
    for label, genome in [("gated96", g96), ("random", random_g)]:
        row = {}
        for opt in ["sgd", "adam", "adam1e3"]:
            a = per_task(genome, cfg, eps, pc, neat_cfg, device, opt)
            row[opt] = a
            print(f"{label:8s} {opt:8s} mean={np.mean(a):.4f}  "
                  f"per-task={['%.3f' % x for x in a]}")
        print()
        grids[label] = row

    print("=== per-task, gated96: SGD vs Adam vs Adam1e-3 ===")
    print(f"{'task':>10s} " + "  ".join(f"{o:>8s}" for o in ["sgd", "adam",
                                                           "adam1e3"]))
    for i, t in enumerate(ids):
        print(f"{t:>10s}  " + "  ".join(
            f"{grids['gated96'][o][i]:8.3f}" for o in ["sgd", "adam", "adam1e3"]))
    print()
    print("means:   " + "  ".join(
        f"{np.mean(grids['gated96'][o]):8.4f}" for o in ["sgd", "adam", "adam1e3"]))
    print("n stuck (<0.65) gated96 sgd:",
          sum(1 for x in grids["gated96"]["sgd"] if x < 0.65))
    print("n stuck (<0.65) gated96 adam:",
          sum(1 for x in grids["gated96"]["adam"] if x < 0.65))


if __name__ == "__main__":
    main()
