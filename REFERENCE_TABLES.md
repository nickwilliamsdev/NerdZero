# nerd-muzero CPPN line — reference tables

Plain-language reference for the pieces you can vary when running
`train_cppn_arc.py`. Everything below is read from the current code
(`train_cppn_arc.py`) and the committed run logs, not from memory.

## What the two layers mean (one-line plain definitions)

- **Genotype** = the evolvable "recipe" (a NEAT genome). It is a small
  neural net called a **CPPN** (a "central pattern producing network" — a
  tiny net that, when you feed it a position, spits out a number).
- **Phenotype** = the actual ARC-solving network that the genotype's CPPN
  *generates the starting weights for*. You pick a phenotype architecture;
  the CPPN fills in its weights.
- **Substrate** = the set of "position" points you feed the CPPN at, one per
  weight of the phenotype. How those points are laid out is the "substrate
  layout". The CPPN reads its response surface at those points to produce
  every weight.
- **In-context SGD** = after the CPPN produces a starting weight set, we run
  a short ordinary training loop (SGD) of the phenotype on one ARC task.
  "More steps" = we let it train longer before we score it.

---

## Table 1 — Phenotypes (the network the genotype generates)

| Name (`--phenotype`) | Class in code | What it is, in plain terms | Where it shines / notes |
|---|---|---|---|
| `conv` | `ArcPhenotype` | 3 convolution layers over the grid images. Looks at local neighbourhoods of cells. | Spatial / local operations. Baseline-style. |
| `attn` | `ArcAttnPhenotype` | Flattens cells into a list of "tokens", runs a small Transformer (self-attention) over them, then predicts colours. | Global reasoning across the whole grid. |
| `deltanet` | `ArcDeltaNetPhenotype` | A single **DeltaNet** block: a fast-weight memory (each cell reads a running memory and writes an error-corrected update), then a head. | Contextual / sequential memory. |
| `mixed` | `ArcMixedPhenotype` | DeltaNet block first, then a Transformer, then a head. Memory feeds into global attention. | Combines memory + global reasoning in one fixed chain. |
| `gated` | `ArcGatedPhenotype` | **Three branches in parallel** — (1) DeltaNet memory, (2) self-attention, (3) conv — each makes a full answer, and a small learned "gate" (3 numbers) blends them. Optionally a 4th "global statistics" branch. | The gate lets the genome lean on whichever branch fits the task. **Best performer so far.** |
| `pure_attn` | `ArcPureAttnPhenotype` | A stack of plain multi-head attention layers only, where each layer keeps its key/query/value projections separate. Pairs with a 4-output CPPN (one output channel per key/query/value + "the rest"). | Attention-only ablation. Weaker than `gated` at the same size. |

Notes:
- All of them read the same input (the demo in/out grids + the query grid) and
  output a 10-colour prediction for the query. So the same scoring code works
  for all of them.
- `gated` with `--gated-branches 4` adds a 4th branch that pools the whole
  grids into one "global statistics" number, aimed at counting / fill-the-rest
  tasks.

---

## Table 2 — CPPN (genotype) configurations

These are the knobs on the *generator* network itself, separate from the
phenotype.

| Knob (config field) | CLI flag | Default | What it controls |
|---|---|---|---|
| `cppn_num_hidden` | `--cppn-hidden` | 14 | **Initial number of hidden nodes in the CPPN** (the genotype's brain size at the start of evolution). NEAT can grow/shrink it later. Runs have used 48 and 96. |
| `cppn_num_outputs` | `--cppn-outputs` | 0 (auto) | How many output channels the CPPN emits per substrate point. Auto = 4 for `pure_attn` (key/query/value/rest), 1 otherwise. |
| `substrate_dim` | `--substrate-dim` | 5 | Dimensionality of the substrate points (how many "address" numbers each point has). 5D for the layout-aware schemes. |
| `dynamic_substrate` | `--dynamic-substrate` / `--no-dynamic-substrate` | on | **On:** the substrate points are shifted per task by a fingerprint of the query grid (so the same genome gives different weights for different tasks). **Off:** a fixed abstract substrate (same weights regardless of task). |
| `substrate_radius` | `--substrate-radius` | 0.25 | How far the per-task shift can move the substrate points (half-size of the shift window). |
| `coord_schema` | `--coord-schema` | auto | The substrate **layout** — see Table 3. |

### Table 3 — Substrate layouts (`--coord-schema`)

| Layout | Plain meaning | Address axes (what each "coordinate" tells the CPPN) |
|---|---|---|
| `flat` | Simplest. Each weight is one point in a regular grid; a conv filter and an attention projection are just two unrelated points. | Mixed-radix digits over the flat weight index. |
| `layer` | Layout-aware. Tells the CPPN *which kind of layer* a weight belongs to, so it can specialise per layer type. (Used when the phenotype has ≥4 layer types, e.g. `gated`.) | Axis 0 = layer type (bias / linear / embedding / conv / norm / other); Axis 1 = which layer in the network; Axes 2–4 = position inside that weight matrix. |
| `kqv` | Like `layer` but for `pure_attn`: tells the CPPN whether a weight belongs to the key, query, or value projection. | Axis 0 = projection (key / query / value / other); Axis 1 = which layer; Axes 2–4 = position inside the matrix. |

- `auto` picks: `kqv` for `pure_attn`; `layer` if the phenotype has ≥4 distinct
  layer types; `flat` otherwise.
- **Current best combo uses `layer` + dynamic substrate, radius 0.25** (auto for `gated`).

---

## Table 4 — The training loop (the nested structure and its knobs)

There are two nested loops plus a scoring step:

```
OUTER loop  — NEAT evolves the genotype (CPPN) topology, generation by generation.
  For each genome in the population:
    INNER loop  — "meta-train" the CPPN on one task, for cppn_iters rounds:
        1. generate starting weights  (CPPN -> phenotype weights)
        2. run in-context SGD on the task (sgd_steps)
        3. compute a meta-loss from how far the weights moved
        4. take one CPPN update (cppn_lr)
    SCORE (fitness)  — on n_eval held-out tasks:
        generate weights -> run eval_sgd_steps of in-context SGD -> count
        cell-accuracy; fitness = the average accuracy.
```

| Knob (config field) | CLI flag | Default | Which loop | What it controls |
|---|---|---|---|---|
| `generations` | `--generations` | 20 | outer | How many NEAT generations to run. |
| `pop_size` | `--pop-size` | 12 | outer | How many genomes (candidates) are in the population. Bigger = more diverse selection; each is expensive. |
| `cppn_iters` | `--cppn-iters` | 6 | inner | How many meta-training rounds per genome (how many times the CPPN gets an update on the task). |
| `sgd_steps` | `--sgd-steps` | 12 | inner | How many in-context SGD steps the phenotype takes per meta-training round ("how hard we train the phenotype inside the loop"). |
| `sgd_lr` | `--sgd-lr` | 0.05 | inner + scoring | Learning rate for the phenotype's in-context SGD. |
| `cppn_lr` | `--cppn-lr` | 0.005 | inner | Learning rate for the CPPN's own (meta) update. |
| `n_eval` | `--n-eval` | 6 | scoring | How many held-out tasks to average over when scoring a genome. |
| `eval_sgd_steps` | `--eval-sgd-steps` | 6 | scoring | How many in-context SGD steps we allow *before* scoring a task. **This is the "train until it overfits the task" knob** — 300 = let it train long. |
| `fixed_eval` | `--fixed-eval` | off | scoring | **On:** the scoring tasks are drawn once and reused every generation (stable selection pressure). **Off:** a fresh random sample of scoring tasks each generation. |

---

## Table 5 — Fitness / meta-loss functions (the signal that trains the CPPN)

Two independent choices: **what loss** drives the inner CPPN update, and
**how the final fitness score is computed**.

### 5a — The meta-loss (`--meta-loss`): what the inner loop optimises

| Option | Plain meaning | Cost / behaviour |
|---|---|---|
| `delta` (default) | "Make the starting weights as close as possible to the weights the phenotype lands on after a bit of training." It measures how far the weights *moved* (squared distance) and pushes the CPPN to start them closer. The moved-to weights are treated as a fixed target (detached). | Cheap and stable. The CPPN only feels *where the weights ended up*, not the task error itself. |
| `maml` | "Make the starting weights that lead to the lowest task error after a bit of training." Backpropagates the actual task loss through the inner SGD steps. (Uses a first-order shortcut — "FOMAML".) | More directly optimises accuracy, but heavier and less stable; in this repo it has not beaten `delta`. |

### 5b — Delta-loss weighting (`--meta-loss-weight`)

| Option | Plain meaning |
|---|---|
| `mean` (default) | Average the squared movement over *every weight*. Big matrices dominate; tiny tensors (like the 3-number gate) get almost no signal. |
| `per_param` | Average the movement *per weight-matrix*, so every matrix (big or tiny) contributes equally. Lets small tensors like the gate learn on the same timescale. |

### 5c — Gate signal (`--gate-meta`, `--gate-task-scale`) — only for `gated`

The `gated` phenotype's 3-number selector (the gate) is tiny, so it barely
learns under the normal `delta` signal.

| Option | Plain meaning |
|---|---|
| `gate-meta=delta` (default) | The gate is trained the same way as everything else (by weight movement). |
| `gate-meta=task` | Replace the gate's signal with the *task-error* gradient, so it's pushed directly to pick the branch that improves accuracy. |
| `gate-task-scale` (0–1) | Blend between the two: 0 = pure movement signal, 1 = pure task-error signal. In-between = a mix. Full task-error (1.0) makes the gate over-commit to one branch and generalised worse; a damped value (0.25–0.5) is the safer middle. |

### 5d — The final fitness (always the same)

Whichever meta-loss is used, a genome's **fitness** is always:
*average cell-by-cell accuracy across `n_eval` held-out tasks, after
`eval_sgd_steps` of in-context SGD.* Cell accuracy = the fraction of grid
cells predicted in the correct colour.

---

## Table 6 — What has actually been measured (from committed run logs)

Scores are cell-accuracy averages. "In-sample" = on the tasks used during
training; "unseen set" = tasks never used in training (seed-13 set, 16 tasks).

| Run (phenotype / key knobs) | In-sample best (fitness) | Unseen set 13 |
|---|---|---|
| `gated`, hidden 48, pop 8, seed 0 (baseline) | 0.7407 (in-sample best, 6 eval steps) | **0.7956** recorded (300 steps, unclipped, lr 0.1) |
| `gated`, d64 (bigger attention branch) | — | 0.7454 (300 steps) — worse than baseline |
| `gated`, 4th branch (global readout) | — | 0.7237 (300 steps) — worse than baseline |
| `pure_attn` (attention-only) | peak ~0.587 | 0.6055 (300 steps) — clearly weaker |
| `gated`, **hidden 96, pop 4, inner 24 steps, eval 300 steps, seed 0** (this session's run) | **0.9255** (peak 0.9364) | **0.7774** (300 steps, unclipped, lr 0.1) |

Reading of the last row (full detail in `CPPN_TRAINING.md`):
- Under the *original* scoring protocol the baseline was judged with, the new
  genome is a **wash** (0.7774 vs 0.7782 baseline).
- Under the other two protocol variants it clearly **wins** (0.8280 and 0.7994
  vs 0.7492 and 0.6015). So the bigger-genotype/smaller-pop/longer-overfit
  recipe made the genome much more *robust* to how you score it, even though
  it didn't raise the ceiling on the original yardstick.
- In-sample selection moved to the 0.92–0.94 band (from ~0.75), i.e. the
  overfit training signal is now doing what it was asked to do.
