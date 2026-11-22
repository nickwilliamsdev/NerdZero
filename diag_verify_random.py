"""Verify the speed-vs-steps diagnostic's random-genome result.

Two checks:
  1. The all-background baseline score on the 16 set-13 tasks (the score you
     get from predicting colour-0 everywhere). If the random genome's 0-step
     score (~0.387) ~= this, the random init is a degenerate all-background
     prediction, not a genuine non-degenerate start.
  2. The actual node/connection count of the seeded random genome, so the
     "randomNN" label is accurate (the config's cppn_num_hidden, not a guess).
"""
import numpy as np

import train_cppn_arc as T
import eval_set13_300steps as E


def background_score(eps, cfg):
    """Fraction of target cells that are background (colour 0), per task and
    averaged — the score of an all-background prediction."""
    fracs = []
    for ep in eps:
        # ep.target is normalised /9 in [0,1]; background = value ~0.
        target = (ep.target.squeeze(0) * 9.0).round().long()
        fracs.append(float((target == 0).float().mean()))
    return float(np.mean(fracs))


def main():
    store = T.ARCStore.load(E.GRID, seed=0)
    template_cfg = E.make_cfg(0.05)
    pc = T.build_param_coords(T.build_phenotype(template_cfg),
                              template_cfg.substrate_dim,
                              __import__("torch").device("cpu"), schema="layer")
    neat_cfg = T.build_neat_config(template_cfg)
    ids, eps = E.sample_unseen(store, template_cfg, E.N_TASKS, seed=E.SEED)
    print("16 tasks:", ids)
    print()
    print("all-background baseline (mean over 16 tasks): %.4f"
          % background_score(eps, template_cfg))
    print()
    # random genome of the SAME hidden size the eval config uses
    pop = T.neat.Population(neat_cfg)
    g = list(pop.population.values())[0]
    print("random genome: %d nodes / %d links "
          "(cppn_num_hidden=%d)"
          % (len(g.nodes), len(g.connections), template_cfg.cppn_num_hidden))


if __name__ == "__main__":
    main()
