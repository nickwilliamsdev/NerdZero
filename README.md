# Nerd MuZero

A hybrid Reinforcement Learning framework built to solve the **ARC (Abstraction and Reasoning Corpus)**. This project fuses the structural weight-generation capabilities of **NeuroEvolution (HyperNEAT CPPNs)** with the rigorous, scalable planning of **MuZero**, all powered by modern, linear-attention **DeltaNet** architectures under the hood.

## 🚀 Key Features

* **ARC Environment Wrapper (`nerd_muzero/envs`)**: A `Gymnasium`-compatible environment mapping ARC tasks to interactive spatial grids via the `arckit` library.
* **HyperNEAT Integration (`nerd_muzero/tensorneat_pt`)**: Seamless integration with a locally cloned PyTorch-NEAT adapter. Multi-output CPPNs evolve structurally and map their weights natively onto the geometry of PyTorch tensors.
* **Linear-Attention Encoders (`nerd_muzero/models/delta_net.py`)**: Replaces traditional ResNets/MLPs with a fast-weight memory sequence encoder (Delta Rule) utilizing `einops` for spatial representations.
* **Causal World Models (`nerd_muzero/models/recursive.py`)**: The Dynamics network operates as a causal DeltaNet, capable of auto-regressive unrolling during MCTS inference and parallel associative scans during training.
* **Monte-Carlo Tree Search (`nerd_muzero/mcts/search.py`)**: PyTorch-native MCTS implementing PUCT strategies and MinMax bounds tracking for continuous exploration.

## 🛠️ Project Structure

```text
nerd_muzero/
├── envs/
│   └── arc_wrapper.py         # Gymnasium interface for ARC grids
├── mcts/
│   └── search.py              # Monte-Carlo Tree Search loop
├── models/
│   ├── delta_net.py           # Linear-attention fast-weight memory representation
│   ├── prediction.py          # Policy and Value head predictions
│   └── recursive.py           # Causal DeltaNet dynamics (world model)
├── tensorneat_pt/
│   ├── cppn.py                # Wrapper mapping neat-python to PyTorch CPPNs
│   ├── evolution.py           # Evaluation hooks for the genome population
│   └── substrate.py           # Logic for querying PyTorch weights
├── training/
│   ├── inner_loop.py          # MuZero agent interactions and backprop (BPTT)
│   └── outer_loop_neat.py     # End-to-end framework joining NEAT + MuZero
scripts/
├── train_arc.py               # E2E Training Entrypoint (Runs evolution & saves winner)
└── evaluate_agent.py          # Deterministic evaluation of saved genomes
tests/                         # Pytest suite for network bounds and MCTS rollout integrity
```

## ⚙️ Installation & Setup

Ensure you have Python 3.10+ installed.

1. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: The `requirements.txt` correctly targets `arckit` from GitHub and PyTorch 2.0+).*

## 🧠 Running the System

### 1. Training (NeuroEvolution + MCTS)
To spin up the `neat-python` population loop, evaluate the CPPNs, query the substrates for DeltaNet weights, and test fitness on the ARC wrapper:
```bash
source .venv/bin/activate
python scripts/train_arc.py
```
*This will run generations based on `neat_config.txt` and automatically output `best_genome.pkl`.*

### 2. Evaluation
To load `best_genome.pkl`, extract the Multi-Output CPPN geometry, overwrite the PyTorch DeltaNet weights, and deterministically roll out an exploitation trajectory:
```bash
source .venv/bin/activate
python scripts/evaluate_agent.py
```

### 3. Running the Test Suite
Validate tensor routing, gradients, and logical tree expansions using `pytest`:
```bash
source .venv/bin/activate
PYTHONPATH=. pytest tests/
```

## 📜 Future Roadmap
- [ ] Connect full BPTT (Backpropagation Through Time) into `inner_loop.py` to allow gradient-based finetuning of NEAT-generated weights.
- [ ] Upgrade `PredictionNetwork` to evaluate purely spatial 2D feature maps instead of flattening the latent state.
- [ ] Add `wandb` telemetry for tracking structural complexity vs generation fitness limits.

---
*Built for geometric reasoning and the search for generic abstraction logic.*
