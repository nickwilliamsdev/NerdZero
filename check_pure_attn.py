"""Unit check for the pure_attn / 4-output CPPN / kqv-substrate plumbing.

Verifies, before trusting any benchmark:
  1. ArcPureAttnPhenotype forward is finite on a random episode.
  2. build_param_coords(schema="kqv") gives k/q/v projections distinct,
     well-separated axis-0 regions and keeps params in [0,1]^5.
  3. encode() with a 4-output TorchCPPN returns exactly pc.flat_size values,
     and the k/q/v chunks really come from CPPN outputs 0/1/2 while the rest
     comes from output 3 (swap test: perturbing channel c changes only the
     params owned by channel c).
  4. Gradients flow to every output channel after a meta-loss step.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import neat
import train_cppn_arc as T


def make_episode(grid_size=10, d=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    def rand(h, w):
        return torch.randint(0, 10, (h, w), generator=g).float() / 9.0
    S = grid_size
    def pad(x):
        t = torch.zeros(S, S)
        t[:x.shape[0], :x.shape[1]] = x
        return t.unsqueeze(0)
    demo_in = torch.stack([pad(rand(5, 5)) for _ in range(d)])
    demo_out = torch.stack([pad(rand(5, 5)) for _ in range(d)])
    q = pad(rand(6, 6))
    tgt = pad(rand(6, 6))
    return T.Episode(
        task_id="unit", demo_in=demo_in, demo_out=demo_out,
        query_in=q, target=tgt, shape=(6, 6), n_demos=d,
        demo_in_shape=[(5, 5)] * d, demo_out_shape=[(5, 5)] * d,
        query_in_shape=(6, 6),
    )


def main():
    cfg = T.Config(phenotype="pure_attn", attn_dim=24, attn_heads=2,
                   attn_layers=1, grid_size=10, cppn_num_outputs=4,
                   cppn_num_hidden=8, seed=0)
    ep = make_episode()

    # 1. forward finiteness
    phen = T.build_phenotype(cfg)
    out = phen(ep)
    assert out.shape == (1, 10, 10, 10), out.shape
    assert torch.isfinite(out).all(), "non-finite logits"
    print(f"1. forward OK  shape={tuple(out.shape)}  "
          f"range=[{out.min():.3f},{out.max():.3f}]  finite=True")

    # 2. kqv coords
    pc = T.build_param_coords(phen, 5, torch.device("cpu"), schema="kqv")
    assert pc.coords.min() >= 0.0 and pc.coords.max() <= 1.0
    for name, p, off in zip(pc.names, [None]*len(pc.names), pc.offsets):
        pass
    groups = {"k_proj": [], "q_proj": [], "v_proj": [], "other": []}
    for name, shape, off in zip(pc.names, pc.shapes, pc.offsets):
        numel = int(torch.prod(torch.tensor(shape)))
        g = "other"
        for k in ("k_proj", "q_proj", "v_proj"):
            if k in name:
                g = k
        groups[g].append((off, numel, pc.coords[off][0].item()))
    for g, rows in groups.items():
        if rows:
            vals = {round(a[2], 3) for a in rows}
            assert len(vals) == 1, (g, vals)
            print(f"2. {g:8s}: {len(rows):2d} params, axis0={rows[0][2]:.3f} "
                  f"(uniform, n={sum(n for _, n, _ in rows)} elems)")
    k0 = groups["k_proj"][0][2]; q0 = groups["q_proj"][0][2]
    v0 = groups["v_proj"][0][2]; o0 = groups["other"][0][2]
    assert abs(k0 - 0.0) < 1e-6 and abs(q0 - 1/3) < 1e-3
    assert abs(v0 - 2/3) < 1e-3 and abs(o0 - 1.0) < 1e-6
    print("   kqv axis-0 regions distinct: k=0.0  q=0.333  v=0.667  rest=1.0")

    # 3. encode dispatch via a real multi-output genome
    neat_cfg = T.build_neat_config(cfg)
    pop = neat.Population(neat_cfg)
    genome = next(iter(pop.population.values()))
    cppn = T.create_torch_cppn(
        genome=genome, config=neat_cfg,
        leaf_names=[f"c{i}" for i in range(5)],
        node_names=["k", "q", "v", "rest"], output_mode="tensor")
    substrate = pc.coords
    vec = T.encode(cppn, pc, substrate)
    assert vec.shape[0] == pc.flat_size, vec.shape
    assert torch.isfinite(vec).all()
    print(f"3. encode OK: flat_size={pc.flat_size}, finite=True, "
          f"range=[{vec.min():.3f},{vec.max():.3f}]")

    # swap test: zero channel c in output nodes -> only channel-c params change
    out_nodes = cppn.output_nodes
    for c in range(4):
        saved = out_nodes[c].weights.detach().clone()
        out_nodes[c].weights.data.zero_()
        with torch.no_grad():
            vec_c = T.encode(cppn, pc, substrate)
        out_nodes[c].weights.data.copy_(saved)
        changed = (vec_c != vec)
        assert changed.any(), f"channel {c} has no effect"
        # changed entries must lie exactly in the params owned by channel c
        own = []
        for name, shape, off in zip(pc.names, pc.shapes, pc.offsets):
            numel = int(torch.prod(torch.tensor(shape)))
            low = name.lower()
            ch = 0 if "k_proj" in low else 1 if "q_proj" in low else \
                 2 if "v_proj" in low else 3
            if ch == c:
                own.append((off, off + numel))
        ok = True
        for s, e in own:
            if not changed[s:e].all():
                ok = False
        others = torch.ones(vec.shape[0], dtype=torch.bool)
        for s, e in own:
            others[s:e] = False
        if changed[others].any():
            ok = False
        assert ok, f"channel {c} leaked into other params"
        print(f"   channel {c} affects exactly its own params "
              f"({int(changed.sum())} values)")

    # 4. gradient flow to every output channel
    phen_m = T.build_phenotype(cfg)
    dev = phen_m.head.weight.device
    w = T.encode(cppn, pc, substrate).to(dev)
    T.load_param_vector(phen_m, pc, w)
    T.sgd_adapt(phen_m, ep, 2, 5e-2, dev)
    delta = T.param_vector(phen_m, pc) - w
    loss = (delta * delta).mean()
    loss.backward()
    grads = {c: out_nodes[c].weights.grad.abs().sum().item()
             for c in range(4)}
    for c, gv in grads.items():
        assert gv > 0, f"output channel {c} got zero gradient"
    print(f"4. meta-loss backward OK; per-channel |grad| sums: "
          + "  ".join(f"ch{c}={v:.4f}" for c, v in grads.items()))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
