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
