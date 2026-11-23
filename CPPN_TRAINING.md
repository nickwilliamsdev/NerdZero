# nerd-muzero — CPPN meta-training: how it works (2026-11-13)

The live research line in this repo is **not** the original MuZero/NEAT ARC
agent (`scripts/train_arc.py`, `nerd_muzero/`); it is a separate, self-contained
experiment: **a neuroevolved CPPN that generates a small few-shot ARC solver,
meta-trained so the initial weights need little in-context adaptation.**

Everything below is driven by `train_cppn_arc.py` (single file, ~1500 lines).
The eval harnesses `eval_*.py` only *read* the saved winner genomes.

---

## The idea in one paragraph

A NEAT genome is a **CPPN** (a tiny feed-forward network). We query that CPPN
at 2-D "coordinate" points; at each point it emits a scalar, and the whole set
of points is the **flat parameter vector of a phenotype** — a small neural
network that solves one ARC task. So the genome *is* a weight generator: it
decides the starting weights of the solver. We then do a little **in-context
SGD** on a single ARC task to move those weights toward the task, and we use
*how far the weights had to move* (the weight delta) as a training signal back
into the CPPN. A good genome is one whose initial weights are already close to
a task solution, so they need little adaptation. NEAT evolves the CPPN genomes
by that signal; the fitness is the mean cell-accuracy on a held-out eval set of
tasks.

Two nested loops, both on the GPU/CPU:

```
OUTER  NEAT evolution of CPPN genomes          (structural: nodes/connections)
  for generation in range(N):
    for genome in population:
      INNER  per-genome meta-training          (the CPPN weight step)
        for it in range(cppn_iters):
          1. W_init  = cppn(substrate)          # encode weights (grad -> cppn)
          2. adapt   = sgd_adapt(phen, task)    # in-context SGD on 1 task
          3. loss    = f(W_after, W_init)       # delta or MAML meta-loss
          4. cppn_opt.step()                    # update CPPN weights
      FITNESS = mean accuracy on n_eval tasks   # re-encode + a little SGD + score
```

The inner loop only updates the CPPN's *weight/bias/response* scalars
(neat-python nodes), **not** its topology — topology changes come from NEAT's
crossover/mutation between generations.

---

## Key objects

| Name | Role |
|---|---|
| `Config` | every hyperparameter (grid size, phenotype arch, dims, lr's, loop counts, seed, device). |
| `Episode` | one ARC task: demo (in,out) grids, a query input grid, and the target output grid, all padded/normalised to `S x S`, colours in [0,1]. |
| `ARCStore.load` | pulls the ARC train set via `arckit`, keeps tasks whose grids fit `grid_size` and have ≥2 demos, shuffles deterministically by seed. |
| `ParamCoords` (`pc`) | the *stable layout* of the phenotype's flat params: names, shapes, offsets, total size `flat_size`, substrate dimensionality `dim`, the [N, dim] coordinate points, and a per-param init `scale`. This is the contract between "flat vector" and "named tensors." |
| `TorchCPPN` / `create_torch_cppn` | the differentiable CPPN (from the `PyTorch-NEAT` submodule). nn.Module so `loss.backward()` reaches the node weight/bias/response Parameters. |
| phenotype `nn.Module` | the actual ARC solver the CPPN encodes. `conv` / `attn` / `deltanet` / `mixed` / `gated`. |

### Substrate: fixed vs dynamic
The CPPN is queried at `pc.coords` (a D-D point per parameter). By default the
substrate is **dynamic**: those points are shifted by a closed-form vector
derived from the *query grid's own statistics* (centre of mass, colour balance,
density, spatial extent — `task_shift`). So the **same genome emits different
weights for different tasks**: the CPPN sees the shifted coordinates and the
task's fingerprint is baked into where on its response surface it's sampled.
`substrate_radius` controls shift magnitude; `dynamic_substrate=False` reverts
to a fixed abstract hypercube.

### Coordinate schemas (`build_param_coords`)
* `flat` — each param is a mixed-radix point in a D-D grid over the flat index.
  A conv kernel and a DeltaNet projection are indistinguishable abstract points.
* `layer` — 5-D, axis 0 = layer-type code (bias/linear/embedding/conv/norm),
  axis 1 = layer index, axes 2-4 = position within the tensor. Lets the CPPN
  specialise per layer type. Used when the phenotype has ≥4 layer types
  (`coord_schema="auto"`); the mixed/gated phenotypes use it, deltanet (3 types)
  uses flat — that heuristic is calibrated on a single-seed A/B and is the
  main suspect to revisit if you see layer-aware underperform.

---

## The phenotypes (the network the CPPN encodes)

All take an `Episode` and return `[1, out_channels, S, S]` logits, so the
encode → SGD → accuracy plumbing is identical across them.

* **conv** — 3 Conv2d layers; encodes demo-in, demo-out, query-in, concatenates,
  1×1 head to colours.
* **attn** — flatten grid into a cell-token sequence (colour + row + col + role
  + within-grid-order embeddings), run a small `TransformerEncoder`, read out the
  query cells, linear head to colours.
* **deltanet** — same token sequence, but a single **DeltaNet** block (fast-weight
  linear-attention with an error-correcting `S_{t+1} = S_t P_t + A_t` recurrence,
  computed by a divide-and-conquer prefix scan `_deltanet_prefix_scan`). Each cell
  reads the memory and writes an update; query cells read out after the scan.
* **mixed** — DeltaNet block **then** a TransformerEncoder (memory → global
  attention), one shared head.
* **gated** — three branches in parallel (DeltaNet + head, Transformer + head,
  conv stem + head), combined by a **softmax gate** over 3 scalars. The gate is
  itself a phenotype parameter the CPPN encodes, so in-context SGD can tune the
  branch mix per task; the design intent is "the CPPN picks which layer type
  dominates per task family."
* **pure_attn** — attention only: a stack of plain multi-head attention blocks
  (`PureAttnBlock`) over the cell-token sequence + colour head. Each block uses
  **separate** k/q/v projections (own `nn.Linear`, no bias) instead of one
  fused in-projection, so the three attention projections are individually
  addressable parameters. This phenotype pairs with a **4-output CPPN**
  (`cppn_num_outputs=4`, auto for `--phenotype pure_attn`): CPPN output
  channel 0 encodes the k-projection weights, 1 the q-projection, 2 the
  v-projection, 3 everything else (embeddings, FFN, norms, head, biases).
  `encode()` dispatches each parameter's substrate chunk to the channel that
  owns it (by name prefix), so the generator gets a dedicated response surface
  per projection. Substrate is the **kqv schema** (auto for pure_attn):
  axis 0 = projection code (k/q/v/other), axis 1 = layer index, axes 2–4 =
  position within the tensor — the k/q/v weight matrices occupy well-separated
  regions of the substrate, matching the 4-channel generator.

Token sequence layout (`episode_to_sequence`): `[demo0_in][demo0_out]…[query_in]`,
each demo grid truncated to its true size; the query-in grid stays full `S x S`
so the output is a uniform `S x S` map.

---

## The meta-loss (the training signal into the CPPN)

`cfg.meta_loss` selects the inner objective:

* **`delta` (1st order, default)** — `L = ||W_after − W_init||²` with `W_after`
  *detached*. The CPPN only feels *where the weights landed*, not the task loss.
  Cheap, stable. Two weightings via `meta_loss_weight`:
    * `mean` — mean over all coordinates (a 3-value gate gets ~N/3× less gradient
      than a big projection, so it barely learns).
    * `per_param` — mean of per-tensor squared deltas, so every tensor counts
      equally and tiny tensors (gates) learn on the same timescale.
* **`maml` (2nd order, FOMAML)** — minimise the *post-adaptation task loss*,
  backpropping through the inner SGD steps. `maml_meta_loss` runs the inner SGD
  with **detached per-step gradients** (so `W` stays a differentiable function of
  `W_init` but no per-step activation graph is retained — cheap on memory), then
  forms `dot(g_final, W_init)` whose gradient w.r.t. `W_init` is exactly the
  FOMAML meta-gradient `∇_W L_task(W_after)`. The drop of the second-order
  Jacobian is what makes it FOMAML.

### Gate meta-signal (`cfg.gate_meta`, gated phenotype only)
The architecture gate is tiny, so its delta signal is weak and (when switched to
the full task-loss gradient) it **over-commits** to one branch per task. The
control is `gate_meta ∈ {delta, task}` + `gate_task_scale ∈ [0,1]`:

* `gate_meta=delta` (default) — gate stays in the weight-delta loss.
* `gate_meta=task` — the gate's contribution is replaced by the post-adaptation
  task-loss gradient `g_gate = ∇_gate L_task`, injected via the same FOMAML
  dot trick (`+ dot(g_gate, W_init[gate])`).
* `gate_task_scale=s` damps it to a **blend** between the two endpoints:
  `s=0` → pure delta; `s=1` → pure task (gate-excluded delta + task dot);
  in between the gate feels `(1-s)·delta + s·task` *— see `evaluate_genome` in
  `train_cppn_arc.py` for the exact per-weighting forms.* Empirically the
  full-task (`s=1`) run made the gate route (DeltaNet
  dropped from ~0.25 to ~0.04–0.11) but *generalised worse* than the delta
  baseline on the 10-split test, i.e. it overfit the branch choice. The damped
  blend is the current experiment.

---

## Fitness + the NEAT loop

`make_evaluator(cfg)` builds `pc`, the `ARCStore`, and the
`eval_genomes(genomes, config)` closure passed to `neat.Population.run`.

* **Shared, fair signal:** the SGD task and the eval task sample are drawn once
  per generation (same for every member). With `fixed_eval=True` the eval set is
  drawn *once* and reused every generation (stable selection pressure); the SGD
  task still varies per generation but is never drawn from the eval ids.
* **Per-genome fitness** = mean of `episode_accuracy` over `n_eval` tasks:
  re-encode the (now trained) CPPN on each eval task's substrate, do
  `eval_sgd_steps` of in-context SGD, then count the cell-wise match rate on the
  unpadded target region.
* **Robustness:** any genome that raises is assigned fitness 0.0 (pop stays
  alive); best genome across all generations is pickled to `cfg.save_name`.

The NEAT config (`neat_config_text`) is a small default genome (default:
`cppn_num_hidden=14`, 5 inputs / 1 output) with node-add / conn-add / conn-delete
rates, tanh/sigmoid activations, sum aggregation, and standard
speciation/stagnation/reproduction. Pop size is kept small (12 default) because
the fitness fn is expensive.

---

## CLI

```
# tiny end-to-end smoke test (1 gen, few genomes)
python train_cppn_arc.py --smoke

# a real (small) run, gated phenotype, damped gate blend, 10 gens
python train_cppn_arc.py \
  --phenotype gated --attn-dim 24 --attn-heads 2 --attn-layers 1 \
  --grid-size 10 --cppn-hidden 48 --pop-size 4 --n-eval 8 \
  --sgd-steps 24 --eval-sgd-steps 6 --generations 10 --fixed-eval \
  --seed 0 --gate-meta task --gate-task-scale 0.5 \
  --save-name cppn_arc_gated_task05_e8.pkl
```

Relevant flags: `--phenotype {conv,attn,deltanet,mixed,gated,pure_attn}`, `--attn-dim/heads/layers`,
`--grid-size`, `--cppn-hidden`, `--cppn-outputs` (0 = auto: 4 for pure_attn, 1 otherwise),
`--pop-size`, `--n-eval`, `--sgd-steps`, `--eval-sgd-steps`,
`--meta-loss {delta,maml}`, `--meta-loss-weight {mean,per_param}`,
`--gate-meta {delta,task}`, `--gate-task-scale`, `--substrate-dim`, `--substrate-radius`,
`--coord-schema {auto,layer,flat,kqv}`, `--dynamic-substrate/--no-dynamic-substrate`,
`--fixed-eval`, `--seed`, `--save-name`, `--device`.
`check_pure_attn.py` unit-checks the pure_attn/kqv/4-output plumbing (forward
finiteness, kqv regions, per-channel encode dispatch, per-channel meta-grad).

---

## What's known to work / be a dead end (2026-11-13)

* **Best so far:** `gated` phenotype (delta gate meta), `attn_dim=24`, 10 gens,
  seed 0 → 10-split generalisation mean ≈ 0.579. Beats plain deltanet
  (`dn24_sgd24`) ≈ 0.552. All are *cell-accuracy* numbers on ARC, not a
  buy-and-hold-style metric.
* **Dead ends (removed):**
  * full-task gate signal (`gate_task_scale=1.0`) — gate routes (DeltaNet
    ~0.25 → ~0.04–0.11) but generalises worse (0.4958 on the 10-split test):
    it overfits the branch choice.
  * damped gate blend (`gate_task_scale=0.5`, `check_scale_effect.py` proves the
    blend is *active* — CPPN weights move monotonically 0→0.5→1) — but at
    10 gens / pop-4 the gate's effect on fitness is too weak to change *which*
    genome NEAT selects, so the winner is ≈ identical to the full-task run and
    still below the delta baseline (0.4958 vs 0.5789). Not a bug; the signal is
    just too small to matter at this budget.
  * `per_param` meta-loss weighting alone — no clear gain.
  * maml on the smaller dims — no clear gain over delta.
* **Open questions:** (1) is the layer-vs-flat substrate heuristic right for
  deltanet? (2) the gate only becomes worth tuning if its fitness signal is
  amplified (longer run, bigger pop, or a per-cell / soft gate) — at the current
  budget the gate is effectively a noise channel. (3) multi-seed stability of
  the gated winner (only seed 0 has been run for it).
* **pure_attn / kqv / 4-output CPPN (2026-11-14):** the plumbing works and is
  unit-verified (`check_pure_attn.py`: finite forward, distinct k/q/v substrate
  regions, exact per-channel encode dispatch, meta-grad reaches all 4 CPPN
  output channels). But a **single attention layer at d=24** is a weaker
  phenotype than the gated mix: 10-split generalisation mean 0.4426 (10 gens,
  pop-8) / 0.4337 (30 gens, pop-8) vs gated 0.5789 and dn24 0.5515 on the
  identical splits. The 30-gen run did NOT improve on 10 — the winner
  plateaued in-sample at ~0.42 after gen ~19 — so the gap is a *phenotype
  expressiveness ceiling* (attention alone, 1 layer), not a training-budget
  problem. To beat the gated baseline the pure-attention branch likely needs
  depth (≥2–3 layers) or a larger d, at the cost of more encoded params.
  The kqv 4-output-CPPN mechanism itself is sound and reusable for any
  phenotype where specific sub-matrices deserve dedicated generator channels.

---

## In-flight run (2026-11-21): gated96 / pop4 / e300

Pickup after the 300-step yardstick settled the architecture (gated 3-branch +
5D layer/dynamic substrate wins; see `eval_sweep_gen_set13_300steps.txt`).
Taking the most promising combo and applying the requested levers, in one run
(`launch_gated96_pop4_e300.sh`, log `run_gated96_pop4_e300_s0.log`,
winner → `cppn_arc_gated96_pop4_e300.pkl`, 30 gens, seed 0):

* population size 8 → **4**
* genotype initial hidden nodes `cppn_num_hidden` 48 → **96**
* phenotype inner grad steps `sgd_steps` 12 → **24**
* eval budget `eval_sgd_steps` 6 → **300** (train the phenotype until it
  "overfits" on each task; the established overfit yardstick)

Everything else held at the baseline combo: gated, d48, ch8, S15, lr 0.05,
delta meta-loss, fixed-eval, dynamic 5D layer substrate, radius 0.25.

Cost: ~1150–1300 s/gen (300-step eval dominates) → 30 gens ≈ 10 h on the GB10.
Early signal (gen 1/2): in-sample best 0.8968, mean 0.8140 / 0.8035, stdev
0.07–0.11 — healthy selection spread, NOT frozen (contrast the old
plain-attention NaN-freeze at the 0.578 background fraction).

When done: score it with `eval_set13_300steps.py` (re-created, committed) —
same unseen 16-task set seed 13, 300 in-context SGD steps per task, so it is
directly comparable to the baseline `ctrl e6 s0` = **0.7956**.

### Interim result (same session)

* 30 gens finished in ~9.3 h (avg ~1100 s/gen). In-sample winner fitness
  **0.9255** (peak 0.9364 gen 8); winner genome (97 nodes / 582 links) — the
  bigger genotype is genuinely in use.
* **Protocol drift discovered:** the recorded 0.7956 baseline was scored
  2026-11-20 11:11, BEFORE commit 8012613 (11:37) added the grad-norm clip to
  `sgd_adapt`. The only code diff between that scoring point and HEAD is clip
  + NaN guard. So:
  * clipped protocol (current code): baseline **0.6015** vs new genome
    **0.7994** (+0.198, same 16 tasks) — new genome wins decisively.
  * unclipped protocol (baseline's original): baseline re-scores **0.7492**
    vs recorded 0.7956 (−0.046; 6/16 per-task values exact, rest slightly
    lower). Two independent unclipped baseline rescoring runs agree to the
    4th decimal → **CUDA run-to-run is NOT the cause**; the original ad-hoc
    protocol differed from my re-creation in some other way. **Found it**
    (`diag_protocol_scan.py`): the ad-hoc run used **sgd_lr 0.10, unclipped**
    — 15/16 per-task values match the recorded vector exactly; the one
    differing task (recorded 0.8056 vs reproduced 0.5278) accounts for the
    whole 0.7956 vs 0.7782 gap (candidate non-determinism at the higher lr,
    being re-confirmed).
  * So the true head-to-heads (same 16 tasks, same protocol for both genomes;
    values re-read from the eval output files, not from memory):
    * **original ad-hoc protocol (unclipped, lr 0.1)**: baseline
      **0.7782** (recorded 0.7956, one task differs) vs new genome
      **0.7774** — **a wash, −0.0008**.
    * unclipped, lr 0.05: baseline **0.7492** vs new genome **0.8280**
      (+0.079) — new genome wins decisively.
    * clipped (current code), lr 0.05: baseline **0.6015** vs new genome
      **0.7994** (+0.198) — new genome wins decisively.
  * **Verdict:** the levers (bigger genotype + smaller pop + longer overfit
    training) did NOT beat the baseline under the protocol the baseline was
    originally judged on — that number is a wash. What they DID buy is
    robustness: the baseline's score swings −0.177 across protocol variants
    (0.7782 / 0.7492 / 0.6015 for the three lr×clip combinations), while the
    new genome swings only +0.051 and never drops below its reference score
    (0.7774 / 0.8280 / 0.7994) — it is stable or better under every variant. The 300-step overfit training also moved
    in-sample selection to the 0.92–0.94 band (vs ~0.75 peaks before), so
    selection is now running on an overfit signal, which was the goal.
  * Side finding: the clip's effect depends on lr — at lr 0.1 it slightly
    HELPS the old baseline (0.7966 vs 0.7782) while at lr 0.05 it badly
    HURTS it (0.6015 vs 0.7492). The clip is not uniformly good or bad.

---

## Synthesis (2026-11-21, after the gated96 run + diagnostics)

### What has been tried (whole line, condensed)

| Axis | Tried | Outcome |
|---|---|---|
| Phenotype | conv, attn, deltanet, mixed, gated(3), gated(4), d64, pure_attn | gated(3) best; d64 wash; 4th branch worse; pure_attn weakest (even at 300 steps) |
| Substrate layout | flat / layer / kqv, dynamic vs fixed | layer + dynamic (radius 0.25) best for gated |
| Meta-loss | delta vs MAML (FOMAML) | delta wins at every budget tested so far |
| Delta weighting | mean vs per_param | no clear gain |
| Gate signal | delta vs task (blend 0.25/0.5) | negative or null; keep delta |
| In-context knobs | lr (0.05/0.10), sgd steps (12/24), attn dim, pop (8/16), cppn hidden (14/48/96) | small or null moves |
| Eval budget | 6 vs 300 steps | huge: raised scores ~+0.2 and "overturned the ceiling" |
| **This session** | gated96 / pop4 / inner24 / eval300, 30 gens | in-sample 0.9255; unseen a wash at 300 steps vs baseline, but much more protocol-robust |

### What the speed-vs-steps diagnostic reveals (the new finding)

On the unseen 16-task set (seed 13), accuracy vs in-context steps
(clipped SGD, lr 0.05; all-background baseline on these tasks = 0.7514):

| provider | 0 steps | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| baseline49 (selected at 6 steps) | 0.0443 | 0.0405 | 0.0569 | 0.2002 | 0.3180 |
| gated96 (selected at 300 steps) | 0.3590 | 0.3608 | 0.3692 | 0.3961 | 0.5129 |
| random genome (49 nodes) | 0.3873 | 0.3873 | 0.3873 | 0.3887 | 0.3467 |

Three things fall out (each verified against its source):

1. **The random genome is a genuine, non-degenerate start** — 0.3873 at 0
   steps is *below* the 0.7514 all-background baseline, and its flat
   0/5/10-then-drop curve is random-noise behaviour, not a background
   degenerate. So it is a valid control.
2. **Short-budget selection is actively harmful.** The baseline49 genome —
   selected on a 6-step fitness — generalizes *worse than random at every
   step count* (0.044@0, 0.318@50 vs random 0.387@0, 0.347@50). Selecting
   "weights barely move in 6 steps" selected a genome whose encoded init is
   near-useless, and the phenotype has to claw its way out from below
   background. This is the null result of the whole first half of the line
   stated precisely: at a 6-step yardstick, selection has no real signal.
3. **Long-budget selection is what works.** gated96 — selected at 300 steps
   — beats random at *every* step count (0.359@0 vs 0.387 is within noise,
   but +0.007@20, +0.166@50, and it was +0.198 at 300 vs baseline49 in the
   clipped protocol). The CPPN encodes real "start close to a solution"
   quality, and that quality transfers to shorter budgets. The earlier
   "wash at 300 steps" was not evidence the genome does nothing; it is
   evidence the yardstick was saturated while the genome's advantage is a
   faster/better start.

**Corrected decision (replacing the short-budget idea I first wrote):**
Do NOT move the yardstick to a short budget — the diagnostic just showed
short-budget selection is the harmful one. The real limit is the ~0.80
ceiling at 300 steps, where ~5 tasks per genome are stuck at 0.42–0.62
while the rest hit ~1.0. Two candidate causes for that ceiling:
(a) the in-context **optimizer** (plain clipped SGD, lr 0.05) underfits the
hard tasks; (b) the **phenotype** cannot represent them at d48.
Cheap decisive test for (a): `diag_adam_ceiling.py` re-scores the same
genomes at 300 steps with Adam vs SGD per task. If Adam lifts the stuck
tasks, the move is to meta-select with Adam in-context adaptation (and the
genomes keep their init quality). If not, the phenotype is the bottleneck
and the move is back to architecture/capacity — but now judged at the
budget that actually discriminates.

---

### 2026-11-23: Adam-vs-SGD ceiling test — DECISIVE (optimizer was the bottleneck)

`diag_adam_ceiling.py` (rewritten to score the CURRENT best under the CURRENT
protocol: clip 10, lr 0.1, 300 steps; Adam arms unclipped at lr 0.01 and
lr 0.001) on the unseen 16-task set-13:

| genome | SGD (lr .1, clip 10) | Adam lr 0.01 | Adam lr 0.001 |
|---|---|---|---|
| pop16-e12-s0 (current best) | **0.8544** | **0.9837** | 0.7604 |
| gated96 pop4 e300 | 0.8766 | 0.9338 | 0.8352 |
| random genome | (in diag_adam_ceiling.txt) | | |

- The SGD arm reproduces the recorded 0.8547 for pop16-e12-s0 (0.8544 this
  run) — pipeline verified faithful.
- **Adam lr 0.01 lifts the fixed best genome from 0.8544 to 0.9837 with NO
  retraining.** The two tasks stuck under SGD (cf5fd0ad 0.556→0.972,
  27f8ce4f 0.617→1.000) are solved. So the ~0.85 ceiling was
  **optimizer-limited, NOT phenotype-expressivity-limited** — the opposite of
  what round 2's deep3/lookup losses suggested, and the exact branch
  `diag_adam_ceiling.py` was set up to test.
- Adam lr 0.001 is *worse than SGD* (0.7604) — it's the specific adaptive
  step-size, not "Adam generally".
- Note: under clip 10 + lr 0.1 the gated96 genome actually re-scores 0.8766
  (higher than pop16's 0.8544) — the old "round 2 verdict" numbers were
  scored before the clip change and do not transfer; the Adam comparison is
  what matters now.

**Decision:** move the in-context optimizer to Adam (lr 0.01) and re-run
meta-selection so NEAT selects on the Adam-adapted fitness. Implemented:
`--in-context-optimizer adam --adam-lr 0.01` (threaded through Config,
`sgd_adapt`, inner meta-training + eval; default stays SGD for backward
compat). Smoke-verified under Adam (fitness spread stdev 0.277, no freeze).

### 2026-11-23 (cont): Adam eval-budget probe + meta-selection run launched

`diag_adam_budget.py` (Adam lr 0.01, best genome vs random, full set-13):

| budget | best | random | gap |
|---|---|---|---|
| 10 | 0.3588 | 0.3744 | **−0.016** |
| 20 | 0.5411 | 0.5497 | −0.009 |
| 50 | 0.7267 | 0.6405 | +0.086 |
| 100 | 0.8795 | 0.7930 | **+0.087** |
| 300 | 0.9718 | 0.9050 | +0.067 |

- Under Adam, short budgets (≤20) are *uninformative* (gap negative — random
  is even slightly ahead), mirroring the earlier SGD finding that short-budget
  selection is harmful. The full selection signal appears by **100 steps** and
  does NOT grow at 300, so the Adam meta-selection run uses **eval 100 steps**
  (1/3 the cost of 300 for the same signal).
- Random Adam-300 = 0.9050, i.e. the ceiling for a random genome under Adam
  is ~0.90; the evolved genome's 0.9837 is the init-quality contribution.
  Meta-selection has +0.087 of signal to work with at 100 steps — real, but
  note the fixed old best already gets 0.9837 under Adam, so the bar for the
  new run to "win" on set-13 is high; the win case is a genome that clears
  0.9837 AND/beat random by more.

**In flight:** `launch_adam300_pop16_e12_s0.sh` (same recipe as the current
best: gated d48/2L, pop16, e12, lr0.1, 30 gens, fixed-eval, seed 0 — but
`--in-context-optimizer adam --adam-lr 0.01 --eval-sgd-steps 100`),
log `run_adam100_pop16_e12_s0.log`, winner ->
`cppn_arc_adam100_pop16_e12_s0.pkl`. Verdict when done:
`eval_set13_adam.py` (Adam-300 on set-13 vs the old best and a random genome).

## File map

| File | What |
|---|---|
| `train_cppn_arc.py` | the whole pipeline (config, phenotypes, substrate, encode/SGD/accuracy, meta-loss, NEAT loop, CLI). |
| `eval_winners.py` | cross-eval 4-task vs 8-task for the two deltanet cppn48 winners (pure genome eval). |
| `eval_gated_winner.py` | eval the gated winner on the same fixed-eval protocol + print post-adapt gate. |
| `eval_all_winners.py` | one protocol-consistent table across all recent winners (dims/sgd steps matched per genome). |
| `eval_one_winner.py` | drop-in: eval a single genome under the established protocol. |
| `eval_generalization.py` | 10 independent random 8-task splits — the trustworthy generalisation test. |
| `eval_set13_300steps.py` | 300-step in-context cross-eval on the unseen 16-task set (seed 13) — the overfit yardstick; re-creates the baseline 0.7956 ranking. |
| `diag_speed_vs_steps.py` | accuracy-vs-in-context-steps curve for several genomes (evolved vs random) on set 13 — the diagnostic that localises the CPPN's value to the low-step regime. |
| `diag_adam_ceiling.py` / `.txt` | in-context OPTIMIZER A/B (SGD clip10 lr0.1 vs Adam lr 0.01/0.001, 300 steps) on set-13 for current best + gated96 + random — the decisive test that the ~0.85 ceiling was optimizer-limited (2026-11-23). |
| `diag_adam_budget.py` | Adam budget-vs-expressivity probe (10/20/50/100/300 steps, best vs random genome) to pick the eval budget for Adam meta-selection (2026-11-23; result: 100). |
| `eval_set13_adam.py` | Adam-300-step cross-eval on the unseen set-13 (successor yardstick to eval_set13_300steps.py); scores the Adam-run winner vs old best vs random (2026-11-23). |
| `launch_adam300_pop16_e12_s0.sh` | launch for the Adam meta-selection run (pop16/e12/lr0.1/30 gens, `--in-context-optimizer adam --adam-lr 0.01 --eval-sgd-steps 100`). |
| `eval_gateblend.py` | fixed-eval + 10-split for the damped gate-blend winner vs delta baseline vs dn24, with gate values. |
| `check_gate_blend.py` | numerical unit check that `gate_task_scale` blends to the two endpoints. |
| `cppn_arc_*.pkl` | saved winner genomes (one per run). |
| `scripts/train_arc.py` | the *legacy* MuZero/NEAT ARC agent (separate, older line). |
| `nerd_muzero/` | the original MuZero model + env package (not used by the CPPN line). |
