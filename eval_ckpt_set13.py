"""Early read: score the in-flight Adam run's checkpoint (best-seen genome) on
set-13 Adam-300, alongside the old best, in the same process (fair CUDA
conditions). The final winner may differ; this is the gen-26 signal."""
import pickle, numpy as np, torch
import train_cppn_arc as T
import eval_set13_adam as A

store = T.ARCStore.load(A.E.GRID, seed=0)
cfg = A.make_cfg()
template = T.build_phenotype(cfg)
pc = T.build_param_coords(template, cfg.substrate_dim, torch.device("cpu"), schema="layer")
neat_cfg = T.build_neat_config(cfg)
device = cfg.resolve_device()
ids, eps = A.E.sample_unseen(store, cfg, A.E.N_TASKS, seed=A.E.SEED)
print(f"set-13 Adam-{A.SGD_STEPS}: {ids}", flush=True)

rows = []
for label, pkl in [("ckpt gen26 (adam run)", "/tmp/ckpt_eval.pkl"),
                   ("old best pop16 e12 s0", "sweep_lr10_e12_pop16_s0.pkl")]:
    genome = pickle.load(open(pkl, "rb"))
    a = A.pure_eval(genome, cfg, eps, pc, neat_cfg, device)
    print(f"{label}: mean={np.mean(a):.4f} per-task={['%.3f' % x for x in a]}", flush=True)
    rows.append((label, float(np.mean(a))))
print("\nearly gap ckpt - old best:", f"{rows[0][1]-rows[1][1]:+.4f}")
