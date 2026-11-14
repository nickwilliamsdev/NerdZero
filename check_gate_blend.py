"""Numerical check that gate_task_scale blends correctly:
  scale=0 -> loss == pure mean delta loss (old behaviour)
  scale=1 -> loss == pure task (gate-excluded delta + dot) as in the first
             gate_meta=task run
Run evaluate_genome with each scale and compare the inner losses via a
small monkeypatch that records the scalar loss value per iters.
"""
import random

import torch

import train_cppn_arc as T


def run_with_scale(scale, gate_meta="task"):
    cfg = T.Config(phenotype='gated', attn_dim=24, attn_heads=2, attn_layers=1,
                   grid_size=10, pop_size=2, n_eval=2, seed=0, fixed_eval=True,
                   cppn_iters=3, sgd_steps=6, eval_sgd_steps=4,
                   cppn_num_hidden=6, gate_meta=gate_meta,
                   gate_task_scale=scale)
    neat_cfg = T.build_neat_config(cfg)
    store = T.ARCStore.load(10, seed=0)
    eps = T.sample_fixed_eval(store, cfg, random.Random(99991))
    sgd_ep = T.sample_sgd_task(store, cfg, random.Random(0),
                               {e.task_id for e in eps})
    template = T.build_phenotype(cfg)
    n_types = len({T._layer_type(n) for n, _ in template.named_parameters()})
    schema = "layer" if n_types >= 4 else "flat"
    pc = T.build_param_coords(template, cfg.substrate_dim,
                              torch.device("cpu"), schema=schema)
    dev = cfg.resolve_device()

    losses = []
    real_backward = torch.Tensor.backward

    def spy_backward(self, *a, **kw):
        losses.append(float(self.detach()))
        real_backward(self, *a, **kw)

    torch.Tensor.backward = spy_backward
    try:
        pop = __import__("neat").Population(neat_cfg)
        genome = next(iter(pop.population.values()))
        fit = T.evaluate_genome(genome, neat_cfg, pc, sgd_ep, eps, cfg, dev)
    finally:
        torch.Tensor.backward = real_backward
    return fit, losses


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # scale 0 (task mode) should match pure delta (delta mode): same genome,
    # same losses -> same trajectory -> same fitness.
    fit_d0, l_d0 = run_with_scale(0.0)
    fit_delta, l_delta = run_with_scale(0.0, gate_meta="delta")
    print(f"scale=0.0 (task):  fitness={fit_d0:.6f} losses={[round(x,6) for x in l_d0]}")
    print(f"meta=delta:        fitness={fit_delta:.6f} losses={[round(x,6) for x in l_delta]}")
    same = all(abs(a - b) < 1e-5 for a, b in zip(l_d0, l_delta)) and abs(fit_d0 - fit_delta) < 1e-5
    print(f"  -> identical trajectory: {same}")

    # scale 1 should differ from both (pure task).
    fit_t1, l_t1 = run_with_scale(1.0)
    print(f"scale=1.0 (task):  fitness={fit_t1:.6f} losses={[round(x,6) for x in l_t1]}")
    print(f"  -> differs from pure delta: {abs(fit_t1 - fit_delta) > 1e-4}")

    # scale 0.5 in between: just check it runs and produces a distinct value.
    fit_s5, l_s5 = run_with_scale(0.5)
    print(f"scale=0.5 (task):  fitness={fit_s5:.6f} losses={[round(x,6) for x in l_s5]}")


if __name__ == "__main__":
    main()
