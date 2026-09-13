from dataclasses import dataclass
import os
import torch

@dataclass
class V30Config:
    # Validated v30 defaults. Keep these for apples-to-apples regression runs.
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

    # Speed / resume controls. These do not alter the v30 objective.
    reuse_neat_winner: bool = True
    neat_winner_path: str = "yetirah_v30_neat_winner.pkl"
    resume_from_algebra: bool = False
    algebra_checkpoint_path: str = "yetirah_v30_algebra.pt"
    posttrain_checkpoint_path: str = "yetirah_v30_posttrain.pt"
    materialize_frozen_operators: bool = True
    matmul_precision: str = "high"

@dataclass
class V30DGXConfig(V30Config):
    """DGX-Spark-oriented defaults.

    The algorithm/losses are unchanged. The larger batch changes the SGD noise
    profile, so use V30Config for exact regression and this config for throughput.
    """
    batch_size: int = 256
    mcts_eval_batch: int = 64
    neat_workers: int = min(16, os.cpu_count() or 12)
    reuse_neat_winner: bool = True
    materialize_frozen_operators: bool = True
