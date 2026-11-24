"""Grid-30 cost probe: 2 generations at grid_size=30 (pop 4, n_eval 4, Adam),
reporting per-generation wall time, so we can extrapolate the full-run cost
before committing to a long grid-30 meta-selection run.

Extrapolation model: cost/generation scales with
  - n_eval (linear, eval is the dominant cost)
  - population size (linear)
  - in-context steps per step (linear)
  - token length: attention is ~quadratic, conv/linear ~linear
At grid15 (225 tokens) the reference run was ~3200 s/gen at pop16/n_eval12.
"""
import dataclasses
import time

import train_cppn_arc as T


def main():
    cfg = dataclasses.replace(
        T.Config(), grid_size=30, pop_size=4, n_eval=4, generations=2,
        in_context_optimizer="adam", adam_lr=0.01,
        cppn_num_hidden=48, eval_sgd_steps=100,
    )
    print(f"cost probe config: {cfg}")
    t0 = time.time()
    T.run(cfg)
    wall = time.time() - t0
    print(f"\nTWO-GEN WALL: {wall:.0f}s  (~{wall/2:.0f}s/gen at pop4 n_eval4)")
    # extrapolate to full recipe pop16 n_eval12
    full = (wall / 2) * (16 / 4) * (12 / 4)
    print(f"EXTRAPOLATED full recipe (pop16 n_eval12): ~{full:.0f}s/gen "
          f"= {full/3600:.1f} h/gen; 30 gens ~{30*full/3600:.0f} h")


if __name__ == "__main__":
    main()
