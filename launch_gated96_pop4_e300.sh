#!/usr/bin/env bash
# Pick up the nerd-muzero CPPN handoff: take the most promising combo
# (gated phenotype + 5D layer/dynamic substrate) and apply the requested levers:
#   - reduce population size        8  -> 4
#   - increase genotype hidden nodes 48 -> 96  (cppn_num_hidden)
#   - increase phenotype grad steps  12 -> 24  (inner sgd_steps)
#   - train until overfit on task    6 -> 300  (eval_sgd_steps)
# Base combo: gated 3-branch, sgd_lr 0.05, fixed-eval, delta meta-loss, seed 0.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python train_cppn_arc.py \
  --phenotype gated \
  --pop-size 4 \
  --cppn-hidden 96 \
  --sgd-steps 24 \
  --eval-sgd-steps 300 \
  --n-eval 6 \
  --cppn-iters 6 \
  --sgd-lr 0.05 \
  --attn-dim 48 --attn-heads 4 --attn-layers 2 \
  --grid-size 15 \
  --fixed-eval \
  --meta-loss delta \
  --seed 0 \
  --generations 30 \
  --device auto \
  --save-name cppn_arc_gated96_pop4_e300.pkl
