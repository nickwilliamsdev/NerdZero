# Yetirah ARC-v1

ARC-v1 is the first real ARC branch built on top of the optimized modular v30 code. The synthetic v30 benchmark remains intact and can still be run with `run_v30.py`, `run_v30_fast.py`, or `run_v30_program_only.py`.

## What changed

ARC-v1 does **not** train through the 9,000-action pixel-edit Gym space. That environment is kept in `sefer/tasks/arc_env.py` for later RL work, but the main learner treats each ARC task as a meta-learning episode:

1. Remaining train pairs are demonstrations.
2. One train pair is held out as query/target during training.
3. Demonstrations are aggregated into a rule embedding.
4. The query grid is encoded into the existing 32-node substrate.
5. A short controller chooses among 22 operators + STOP.
6. During training only, the real output is encoded as a latent target and supplies reconstruction/closure supervision.
7. At evaluation, only demonstrations + query input are available.

## ARC-v1 architecture

```text
ARC demos ──> shared grid encoder ──> demo-pair encoder ──> rule
                                                        │
query grid ──> grid encoder ──> 32x32 substrate H0      │
                         │                              │
                         └────> ARC program controller <┘
                                      │
                               operator / STOP
                                      │
                    adaptive residual around frozen v30
                         22-operator NEAT algebra
                                      │
                                      v
                                  H_final
                                      │
                                 grid decoder
                                      │
                           colors + height + width
```

The adaptive operator module starts from the validated v30 `[22,32,32]` bank and learns a small low-rank residual in log-transport space. The original synthetic algebra remains frozen and unchanged.

## Training curriculum

ARC-v1 deliberately stabilizes the representation before asking operators to compose:

- **Codec phase** (`codec_steps=500`): ARC grid <-> 32-slot substrate autoencoding.
- **Direct meta phase** (`direct_steps=1000`): demonstrations -> rule embedding, query -> output baseline.
- **Program phase** (`program_steps=4000`): freeze grid codec; learn short operator programs and low-rank operator adaptations using latent-target + grid reconstruction losses.

## Run

From this directory, with `arckit` installed:

```bash
python arc_smoke_test.py
python run_arc_v1.py
```

DGX Spark starting point:

```bash
python run_arc_v1.py --batch-size 64 --program-steps 4000
```

For a quick subset experiment:

```bash
python run_arc_v1.py --limit-tasks 100 --codec-steps 250 --direct-steps 500 --program-steps 1500
```

ARC-v1 automatically uses these v30 files when they are present in the working directory:

```text
yetirah_v30_neat_winner.pkl
yetirah_v30_algebra.pt
```

You can override them:

```bash
python run_arc_v1.py \
  --neat-winner /path/to/yetirah_v30_neat_winner.pkl \
  --algebra-checkpoint /path/to/yetirah_v30_algebra.pt
```

Or start ARC operators without the v30 prior:

```bash
python run_arc_v1.py --no-v30-init
```

## Evaluate

```bash
python eval_arc_v1.py --checkpoint yetirah_arc_v1.pt --limit 100
```

Metrics include exact solve rate, pixel accuracy, shape accuracy, and operator usage.

## Important limitation of v1

The test-time controller is demonstration-conditioned rather than goal-conditioned, because ARC does not reveal the query output. This is the correct direction for ARC, but it means the exact Bellman teacher from the synthetic benchmark is no longer available. ARC-v1 is therefore testing whether reusable latent operators can emerge from task demonstrations + output reconstruction alone.

The next likely steps after collecting ARC-v1 curves are object-centric tokens, persistent DeltaNet task memory, demonstration-consistency search, and PUCT over programs scored jointly across every training pair in the task.
