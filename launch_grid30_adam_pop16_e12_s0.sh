#!/usr/bin/env bash
# Grid-30 Adam meta-selection run (2026-11-24).
#
# Same recipe as the Adam winner (gated d48/2L, pop16/e12, lr0.1 inner SGD
# lr for the delta step, Adam lr0.01 in-context, 100-step eval, seed 0),
# but:
#   1. --grid-size 30: unlocks the FULL train set (1000 tasks) and the FULL
#      official test set (120/120 tasks) as selection/eval pools
#      (verified: smoke_grid30.py).
#   2. NO --fixed-eval: the eval sample rotates every generation (12 tasks
#      drawn from the 1000 eligible each gen), fixing the in-sample
#      saturation the 12-task fixed eval hit at grid 15 (best hit 1.0000 at
#      gen 26 and fitness stopped discriminating). Selection now sees the
#      full train pool, per the "select on the full train set, judge on the
#      full eval set" protocol.
#   3. Eval budget stays at 100 Adam steps (grid-15 probe showed the
#      selection signal saturates by 100 and doesn't grow at 300; re-probe
#      at grid-30 only if the selection signal looks weak).
#
# Cost: ~600 s/gen extrapolated from diag_grid30_cost.py (quiet-GPU probe)
# -> ~5 h for 30 gens; up to ~26 h under heavy vLLM contention.
# Per-gen checkpoint -> cppn_arc_grid30_adam100_pop16_e12_s0.pkl.ckpt
#
# Final verdict when it lands: score the winner on ALL 120 official test
# tasks at grid-30 (eval_official_test.py generalized to grid 30).
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python train_cppn_arc.py \
  --phenotype gated --attn-dim 48 --attn-heads 4 --attn-layers 2 \
  --grid-size 30 --channels 8 \
  --pop-size 16 --n-eval 12 --generations 30 \
  --sgd-steps 12 --eval-sgd-steps 100 --sgd-lr 0.10 \
  --in-context-optimizer adam --adam-lr 0.01 \
  --cppn-hidden 48 \
  --seed 0 \
  --save-name cppn_arc_grid30_adam100_pop16_e12_s0.pkl
