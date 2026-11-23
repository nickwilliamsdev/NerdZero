"""Is the ~0.85 300-step ceiling limited by the in-context OPTIMIZER (SGD) or by
the phenotype's expressivity?

Background (see CPPN_TRAINING.md "Synthesis"): the current best genome
(pop16-e12-s0) scores 0.8547 on the unseen 16-task set-13 under the established
300-step in-context protocol (clipped SGD, clip 10, lr 0.1). ~5 tasks are stuck
at 0.42-0.62 while the rest hit ~1.0. diag_budget_sweep.py already showed that
MORE SGD steps (300->1000) do NOT lift the stuck tasks (flat). The remaining
cheap decisive test is a different in-context OPTIMIZER: Adam (adaptive per-param
lr) may escape the stuck tasks where plain SGD stalls.

Protocol: for each genome, at 300 in-context steps, compare per task on the
unseen 16-task set (seed 13):
  sgd     = current clipped SGD  (clip 10, lr 0.1)  -- the established protocol
  adam_01 = Adam  lr 0.01, no clip
  adam_001= Adam  lr 0.001, no clip
so the SGD baseline reproduces the 0.8547 number and the Adam arms are the
sensible adaptive-optimizer sweep.

If Adam clearly beats SGD on the stuck tasks and raises the mean, the ceiling is
optimizer-limited -> the move is to meta-select with Adam in-context adaptation
(the genomes keep their init quality). If Adam is flat/worse, the ceiling is
phenotype-expressivity-limited -> consistent with round-2 deep3/lookup losing;
the move is a different architecture direction, judged at this budget.
"""
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

import train_cppn_arc as T
import eval_set13_300steps as E

STEPS = 300
SGD_LR = 0.1          # matches the current best's established 300-step protocol
CLIP = 10.0          # matches train_cppn_arc.sgd_adapt (current code)


def adapt(phen, ep, steps, lr, device, clip, optimizer):
    phen.train()
    if optimizer == "sgd":
        opt = torch.optim.SGD(phen.parameters(), lr=lr)
    elif optimizer in ("adam_01", "adam_001"):
        a_lr = 0.01 if optimizer == "adam_01" else 0.001
        opt = torch.optim.Adam(phen.parameters(), lr=a_lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    epd = T._ep_on_device(ep, device)
    target = (epd.target.squeeze(0) * 9.0).round().long().unsqueeze(0)
    for _ in range(steps):
        opt.zero_grad()
        pred = phen(epd)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(phen.parameters(), clip)
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
        clip = CLIP if optimizer == "sgd" else None
        adapt(phen, ep, STEPS, SGD_LR, device, clip, optimizer)
        accs.append(T.episode_accuracy(phen, ep, device))
    return accs


def make_best_cfg(sgd_lr):
    return T.Config(
        phenotype='gated', channels=8, attn_dim=48, attn_heads=4, attn_layers=2,
        grid_size=E.GRID, pop_size=16, n_eval=12, seed=E.SEED,
        fixed_eval=True, cppn_num_hidden=48, gated_branches=3,
        eval_sgd_steps=STEPS, sgd_lr=sgd_lr,
    )


def main():
    only = []  # optional: limit to a subset of task ids (passed via argv)
    if len(__import__("sys").argv) > 1:
        only = __import__("sys").argv[1:]

    store = T.ARCStore.load(E.GRID, seed=0)
    cfg = make_best_cfg(SGD_LR)
    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()

    ids, eps = E.sample_unseen(store, cfg, E.N_TASKS, seed=E.SEED)
    if only:
        keep = [i for i, t in enumerate(ids) if t in only]
        ids = [ids[i] for i in keep]
        eps = [eps[i] for i in keep]
    print(f"Unseen set-13 (seed {E.SEED}), {STEPS}-step in-context optimizer A/B "
          f"({len(ids)} tasks): {ids}")
    print(f"arms: sgd(lr{SGD_LR},clip{CLIP})  adam_01(lr0.01)  adam_001(lr0.001)")
    print()

    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]
    genomes = [
        ("pop16-e12-s0 (best)", "sweep_lr10_e12_pop16_s0.pkl"),
        ("gated96 pop4 e300", "cppn_arc_gated96_pop4_e300.pkl"),
        ("random", None),
    ]
    loaded = {}
    for label, pkl in genomes:
        if pkl:
            with open(pkl, "rb") as f:
                loaded[label] = pickle.load(f)
        else:
            loaded[label] = random_g

    arms = ["sgd", "adam_01", "adam_001"]
    grid = {}
    t0 = time.time()
    for label, _pkl in genomes:
        genome = loaded[label]
        grid[label] = {}
        for opt in arms:
            a = per_task(genome, cfg, eps, pc, neat_cfg, device, opt)
            grid[label][opt] = a
            print(f"{label:22s} {opt:8s} mean={np.mean(a):.4f}  "
                  f"per-task={['%.3f' % x for x in a]}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
        print(flush=True)

    print("=== per-task, pop16-e12-s0 (best): SGD vs Adam ===")
    print(f"{'task':>10s}  " + "  ".join(f"{o:>8s}" for o in arms))
    for i, t in enumerate(ids):
        print(f"{t:>10s}  " + "  ".join(
            f"{grid['pop16-e12-s0 (best)'][o][i]:8.3f}" for o in arms))
    print()
    for label, _p in genomes:
        means = [np.mean(grid[label][o]) for o in arms]
        n_stuck = [sum(1 for x in grid[label][o] if x < 0.65) for o in arms]
        print(f"{label:22s} means " + "  ".join(f"{m:.4f}" for m in means)
              + "   | n_stuck(<0.65) " + "  ".join(f"{n}" for n in n_stuck))
    print()
    print("total wall: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
