"""Grid-30 Adam eval-budget probe: does the in-flight run's 100-step selection
budget still look right at grid_size=30 (where the token length is 4x the
grid-15 value the 100-step number was chosen on)?

Method: for each genome, run ONE 300-step Adam adaptation per task and read
cell-accuracy at budgets {10,20,50,100,300}. Prints best-vs-random gap per
budget; the budget where the gap peaks / saturates is the selection budget.

Best genome = grid-15 Adam winner (cppn_arc_adam100_pop16_e12_s0.pkl) — its
init transfers to grid 30 (same gated d48 phenotype, S=30). Control = fresh
random genome. 16 unseen tasks (seed 13) from the grid-30 eligible pool.
"""
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F

import train_cppn_arc as T

GRID = 30
BUDGETS = [10, 20, 50, 100, 300]
N_TASKS = 16
SET_SEED = 13


def adapt_curve(phen, ep, device, max_steps, adam_lr=0.01):
    """Adapt `phen` on one episode up to max_steps with Adam; return a dict
    {budget: accuracy} read at each budget in BUDGETS <= max_steps."""
    phen.train()
    opt = torch.optim.Adam(phen.parameters(), lr=adam_lr)
    epd = T._ep_on_device(ep, device)
    target = (epd.target.squeeze(0) * 9.0).round().long().unsqueeze(0)
    got = {}
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        pred = phen(epd)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        opt.step()
        if step in BUDGETS:
            phen.eval()
            got[step] = T.episode_accuracy(phen, epd, device)
            phen.train()
    return got


def score(genome, eps, pc, neat_cfg, cfg, device, max_steps):
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(pc.dim)], node_names=["w"])
    per_budget = {b: [] for b in BUDGETS}
    for ep in eps:
        phen = T.build_phenotype(cfg)
        substrate = T._substrate_for(ep, pc, cfg)
        with torch.no_grad():
            W = T.encode(cppn, pc, substrate).to(device)
        T.load_param_vector(phen, pc, W)
        curve = adapt_curve(phen, ep, device, max_steps)
        for b in BUDGETS:
            if b in curve:
                per_budget[b].append(curve[b])
    return {b: (float(np.mean(v)) if v else float("nan")) for b, v in per_budget.items()}


def main():
    store = T.ARCStore.load(GRID, seed=0)
    rng = random.Random(SET_SEED)
    eligible = list(range(len(store.eligible)))
    idx = rng.sample(eligible, N_TASKS)
    tasks = [store.eligible[i] for i in idx]
    print(f"grid-30 eligible pool: {len(eligible)} tasks; using {N_TASKS} (seed {SET_SEED})")
    print(f"task ids: {[str(getattr(t, 'id', ''))[:8] for t in tasks]}")

    cfg = T.Config(phenotype='gated', channels=8, attn_dim=48, attn_heads=4,
                   attn_layers=2, grid_size=GRID, pop_size=16, n_eval=12,
                   seed=0, cppn_num_hidden=48, gated_branches=3,
                   eval_sgd_steps=300, sgd_lr=0.1,
                   in_context_optimizer='adam', adam_lr=0.01)
    eps = [store.make_episode(t, cfg.max_demos, rng) for t in tasks]

    template = T.build_phenotype(cfg)
    pc = T.build_param_coords(template, cfg.substrate_dim, torch.device("cpu"),
                              schema="layer")
    neat_cfg = T.build_neat_config(cfg)
    device = cfg.resolve_device()
    pop = T.neat.Population(neat_cfg)
    random_g = list(pop.population.values())[0]
    best_g = pickle.load(open("cppn_arc_adam100_pop16_e12_s0.pkl", "rb"))

    print("scoring grid-15 adam-winner on grid-30 tasks (10/20/50/100/300-step Adam)...")
    b = score(best_g, eps, pc, neat_cfg, cfg, device, 300)
    print("scoring fresh random genome...")
    r = score(random_g, eps, pc, neat_cfg, cfg, device, 300)

    print()
    print(f"{'steps':>6} | {'best':>7} | {'random':>7} | {'gap':>7}")
    print("-" * 34)
    for s in BUDGETS:
        print(f"{s:>6d} | {b[s]:>7.4f} | {r[s]:>7.4f} | {b[s]-r[s]:>+7.4f}")
    print()
    print("=> pick the budget where the best-vs-random gap saturates for the")
    print("   grid-30 meta-selection run's eval_sgd_steps.")


if __name__ == "__main__":
    main()
