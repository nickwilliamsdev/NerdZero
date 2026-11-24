"""Grid-30 smoke: verify the pipeline works at grid_size=30 (all 1000 train
tasks + all 120 official-test tasks become usable) and measure the cost.

Runs T.smoke with a tiny population (pop 4, n_eval 3) under the Adam
protocol. Also prints how many tasks fit at 30 vs 15.
"""
import dataclasses
import time

import arckit
import numpy as np

import train_cppn_arc as T


def fits(t, S):
    for grp in ("train", "test"):
        for x, y in getattr(t, grp, []):
            x, y = np.asarray(x), np.asarray(y)
            if max(x.shape) > S or max(y.shape) > S:
                return False
    return True


def main():
    tr, te = arckit.load_data()
    print(f"grid15: official-test fit {sum(fits(t, 15) for t in te.tasks)}/120, "
          f"train fit {sum(fits(t, 15) for t in tr.tasks)}/1000")
    print(f"grid30: official-test fit {sum(fits(t, 30) for t in te.tasks)}/120, "
          f"train fit {sum(fits(t, 30) for t in tr.tasks)}/1000")
    cfg = dataclasses.replace(
        T.Config(), grid_size=30, pop_size=4, n_eval=3,
        in_context_optimizer="adam", adam_lr=0.01,
    )
    print("smoke grid30 (pop4, n_eval=3, adam)...")
    t0 = time.time()
    w = T.smoke(cfg)
    print(f"SMOKE OK in {time.time()-t0:.0f}s, winner saved: {w}")


if __name__ == "__main__":
    main()
