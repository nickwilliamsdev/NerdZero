# v30 Module Map

## Core dataflow

`tasks -> representation -> controllers -> algebra -> planning -> evaluation`

Evolution targets the algebra; training coordinates representation/controllers/algebra; experiments orchestrate the complete run.

## Files

| Module | Responsibility |
|---|---|
| `sefer/geometry/hypercube.py` | 5D hypercube substrate coordinates |
| `sefer/representation/scalar_codec.py` | Exact scalar lift/readout |
| `sefer/representation/delta_state.py` | DeltaNet-style fast-weight program-state encoder |
| `sefer/algebra/cppn.py` | Legacy differentiable CPPN |
| `sefer/algebra/operator_hypernet.py` | Operator-conditioned transport + feature affine generation |
| `sefer/algebra/core.py` | `YetirahCore` substrate/operator interface |
| `sefer/algebra/rollout.py` | Differentiable/deterministic primitive and program execution |
| `sefer/controllers/reasoner.py` | `TinyReasoner`, primitive policy, program policy/value, DeltaNet integration |
| `sefer/planning/bellman.py` | Goal scores and exact finite-horizon Bellman targets |
| `sefer/planning/puct.py` | PUCT tree search and transposition/state-signature logic |
| `sefer/evolution/neat_runtime.py` | PyTorch-NEAT integration and differentiable genome CPPN |
| `sefer/evolution/neat_config.py` | NEAT configuration text |
| `sefer/evolution/evolve_transport.py` | Memetic/Lamarckian CPPN evolution |
| `sefer/tasks/synthetic_algebra.py` | Synthetic primitive/composition benchmark |
| `sefer/training/losses.py` | Transport supervision loss |
| `sefer/training/phases.py` | Train/freeze phase helpers |
| `sefer/training/trainer.py` | Original v30 three-phase training loop |
| `sefer/evaluation/primitives.py` | Representation/atomic/composition diagnostics |
| `sefer/evaluation/programs.py` | Greedy and exact program evaluations |
| `sefer/evaluation/planning.py` | MuZero/PUCT and root-planning evaluation |
| `sefer/config.py` | Convenience dataclass mirroring v30 defaults |
| `sefer/experiments/synthetic_v30.py` | Modular v30 experiment entrypoint |
| `legacy/yetirah_v30_monolithic.py` | Untouched validated v30 source |

## Recommended reading order

1. `sefer/experiments/synthetic_v30.py`
2. `sefer/training/trainer.py`
3. `sefer/controllers/reasoner.py`
4. `sefer/algebra/core.py`
5. `sefer/algebra/operator_hypernet.py`
6. `sefer/planning/puct.py`
7. `sefer/planning/bellman.py`
8. `sefer/evolution/evolve_transport.py`
