# Yetirah v30 — Modular Refactor

This directory is a pure structural refactor of `yetirah_v0_neat_cppn_v30_recovery_deltanet.py`. The validated monolithic source is preserved in `legacy/yetirah_v30_monolithic.py`.

## Run

```bash
python smoke_test.py
python run_v30.py
```

Run from the `yetirah_v30_modular` directory (or add it to `PYTHONPATH`). The PyTorch-NEAT integration searches ancestor/sibling locations for a `PyTorch-NEAT/` checkout.

## Layout

- `sefer/geometry/` — substrate geometry
- `sefer/representation/` — fixed scalar codec and DeltaNet state encoder
- `sefer/algebra/` — CPPN, operator hypernetwork, substrate core, rollouts
- `sefer/controllers/` — `TinyReasoner` and policy/value heads
- `sefer/planning/` — exact Bellman utilities and PUCT
- `sefer/evolution/` — NEAT runtime/config and memetic transport evolution
- `sefer/tasks/` — synthetic algebra benchmark
- `sefer/training/` — losses, trainability phases, training loop
- `sefer/evaluation/` — primitive, program, and planning evaluation
- `sefer/experiments/` — experiment entrypoints
- `legacy/` — untouched v30 monolith

## Refactor rule

No intended algorithmic changes were made. Keep `legacy/yetirah_v30_monolithic.py` as the regression source while beginning ARC work in new modules.


## DGX Spark speed path

The optimized package preserves the validated v30 objective while adding faster execution paths.

- `python run_v30.py` keeps the validated batch size (32), but reuses `yetirah_v30_neat_winner.pkl` when present.
- `python run_v30_fast.py` uses `V30DGXConfig` (batch 256, larger eval batches).
- `python run_v30_program_only.py` loads `yetirah_v30_neat_winner.pkl` plus `yetirah_v30_algebra.pt` and skips both evolution and algebra training.
- `python speed_smoke_test.py` validates the cached operator bank, vectorized action application, and batched Bellman targets.

At the algebra/program boundary the full `[22,32,32]` transport bank plus affine laws are materialized once. Exact Bellman targets are `no_grad` and breadth-vectorized through depth 4.

For a fresh run, first let the code produce both checkpoints. Subsequent controller/search experiments can use the program-only runner.
