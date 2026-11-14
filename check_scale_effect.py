"""Controlled test: does gate_task_scale actually change the CPPN weights?

Take ONE fixed random genome, run the *same* inner meta-training loop that
evaluate_genome uses, at scale 0.0 / 0.5 / 1.0, and compare the resulting CPPN
parameter vector.  If the weights diverge across scales, the blend is active and
an identical 10-gen winner would be a genuine selection artifact.  If they do
NOT diverge, the scale is not taking effect (bug).
"""
import random

import numpy as np
import torch

import train_cppn_arc as T


def inner_train(genome, cfg, neat_cfg, pc, sgd_ep, device):
    """Replicate evaluate_genome's inner delta loop, return final cppn params."""
    dim = pc.dim
    cppn = T.create_torch_cppn(genome=genome, config=neat_cfg,
                               leaf_names=[f"c{i}" for i in range(dim)],
                               node_names=["w"])
    opt = torch.optim.SGD(cppn.parameters(), lr=cfg.cppn_lr)
    phen = T.build_phenotype(cfg)
    gate_idx = next((i for i, n in enumerate(pc.names) if n == "gate"), None)
    gate_slice = (pc.offsets[gate_idx], int(np.prod(pc.shapes[gate_idx]))) \
        if gate_idx is not None else None
    scale = float(cfg.gate_task_scale) if (cfg.gate_meta == "task" and gate_slice) else 0.0
    for _ in range(cfg.cppn_iters):
        substrate = T._substrate_for(sgd_ep, pc, cfg)
        W_init = T.encode(cppn, pc, substrate).to(device)
        T.load_param_vector(phen, pc, W_init)
        T.sgd_adapt(phen, sgd_ep, cfg.sgd_steps, cfg.sgd_lr, device)
        W_after = T.param_vector(phen, pc)
        delta = W_after - W_init
        if scale > 0.0:
            off, size = gate_slice
            epd = T._ep_on_device(sgd_ep, device)
            g_gate = torch.autograd.grad(
                T.task_ce_loss(phen, epd), phen.gate)[0].detach()
            n_all = delta.numel()
            n_ng = n_all - size
            part_all = (delta * delta).sum() / n_all
            ng = torch.cat([delta[:off], delta[off + size:]])
            part_ng = (ng * ng).sum() / n_ng
            dot = (g_gate.reshape(-1) * W_init[off:off + size]).sum()
            loss = ((1.0 - scale) * part_all + scale * (part_ng + dot))
        else:
            loss = (delta * delta).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        vec = torch.cat([p.reshape(-1) for p in cppn.parameters()])
    return vec.clone()


def main():
    cfg = T.Config(phenotype='gated', attn_dim=24, attn_heads=2, attn_layers=1,
                   grid_size=10, pop_size=2, n_eval=2, seed=0, fixed_eval=True,
                   cppn_iters=6, sgd_steps=12, eval_sgd_steps=4,
                   cppn_num_hidden=14, gate_meta='task')
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
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    pop = __import__("neat").Population(neat_cfg)
    genome = next(iter(pop.population.values()))

    vecs = {}
    for s in [0.0, 0.5, 1.0]:
        c = T.Config(**{**cfg.__dict__, "gate_task_scale": s,
                        "gate_meta": "task" if s > 0 else "delta"})
        vecs[s] = inner_train(genome, c, neat_cfg, pc, sgd_ep, dev)

    base = vecs[0.0]
    print(f"n cppn params: {base.numel()}")
    for s in [0.0, 0.5, 1.0]:
        d = (vecs[s] - base).abs()
        print(f"  scale={s:.1f}: max|dW|={d.max():.6f}  mean|dW|={d.mean():.8f}")
    d_05_10 = (vecs[0.5] - vecs[1.0]).abs()
    print(f"  scale 0.5 vs 1.0: max|dW|={d_05_10.max():.6f} "
          f"mean|dW|={d_05_10.mean():.8f}")
    same = d_05_10.max() == 0.0
    print(f"  -> scale 0.5 identical to scale 1.0: {same}")
    print("  VERDICT:", "BLEND ACTIVE (weights move with scale)"
          if not same else "BLEND DEAD (scale has no effect — bug)")


if __name__ == "__main__":
    main()
