from dataclasses import dataclass
import torch

@dataclass
class V30Config:
    seed: int = 0
    steps: int = 4200
    batch_size: int = 32
    inner_rollout_steps: int = 1
    warmup_steps: int = 250
    algebra_steps: int = 1200
    es_every: int = 0
    diagnostic_every: int = 25
    mcts_train_every: int = 5
    mcts_train_samples: int = 4
    mcts_simulations: int = 96
    mcts_eval_simulations: int = 384
    mcts_eval_batch: int = 16
    neat_generations: int = 100
    neat_population: int = 256
    neat_workers: int = 12
    neat_inner_steps: int = 32
    neat_inner_lr: float = 1e-2
    neat_seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
