#!/usr/bin/env bash
# Adam meta-selection run (2026-11-23): the same recipe as the current best
# (pop16 / e12 / lr10 / 30 gens / fixed-eval, gated d48, layer substrate),
# but in-context adaptation (inner SGD task + eval fitness) uses Adam lr 0.01
# instead of clipped SGD. diag_adam_ceiling.py showed the FIXED pop16-e12-s0
# genome scores 0.9837 on the unseen set-13 under Adam-300 vs 0.8544 under
# SGD-300, so selection under the Adam protocol should be able to find a
# genome whose encoded init is even better for the Adam-adapted phenotype.
#
# Eval budget = 100 steps (NOT 300): diag_adam_budget.py measured the
# best-vs-random gap per budget on set-13: -0.016 @10, -0.009 @20,
# +0.086 @50, +0.087 @100, +0.067 @300. The full selection signal appears by
# 100 steps and does not grow at 300, so 100 keeps the signal at 1/3 the
# eval cost. (10-20-step budgets are actively uninformative under Adam, the
# same short-budget-harmful finding as under SGD.)
#
# Final verdict for the winner: eval_set13_adam.py (Adam 300-step on set-13,
# vs the fixed old best and a random genome).
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python train_cppn_arc.py \
  --phenotype gated --attn-dim 48 --attn-heads 4 --attn-layers 2 \
  --grid-size 15 --channels 8 \
  --pop-size 16 --n-eval 12 --generations 30 \
  --sgd-steps 12 --eval-sgd-steps 100 --sgd-lr 0.10 \
  --in-context-optimizer adam --adam-lr 0.01 \
  --cppn-hidden 48 --fixed-eval \
  --seed 0 \
  --save-name cppn_arc_adam100_pop16_e12_s0.pkl
