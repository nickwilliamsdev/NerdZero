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

@dataclass
class ARCConfig:
    """ARC-v1 meta-learning defaults.

    ARC-v1 keeps the 32-node Yetirah substrate, replaces the scalar codec with a
    grid<->slot codec, and learns task-conditioned operator programs from ARC
    demonstrations.  The synthetic v30 checkpoint is used only as an
    initialization/prior; its original files remain untouched.
    """
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_grid_size: int = 30
    color_count: int = 10
    n_slots: int = 32
    node_dim: int = 32
    coord_dim: int = 5
    operator_count: int = 22
    operator_code_dim: int = 8
    rule_dim: int = 128
    max_demos: int = 4
    max_program_steps: int = 4

    # Dataset/meta-learning.
    split: str = "train"
    validation_fraction: float = 0.10
    limit_tasks: int | None = None
    batch_size: int = 64
    eval_tasks: int = 64
    num_workers: int = 0  # sampling is in-memory; GPU work dominates

    # Three-stage curriculum.
    codec_steps: int = 500
    direct_steps: int = 1000
    program_steps: int = 4000
    diagnostic_every: int = 50
    eval_every: int = 500

    # Optimizer.
    codec_lr: float = 3e-4
    direct_lr: float = 3e-4
    program_lr: float = 3e-4
    operator_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    matmul_precision: str = "high"

    # Program-learning losses.
    latent_weight: float = 1.0
    grid_weight: float = 1.0
    shape_weight: float = 0.25
    direct_aux_weight: float = 0.15
    length_weight: float = 0.01
    operator_reg_weight: float = 0.01
    usage_balance_weight: float = 0.01
    value_weight: float = 0.10
    gumbel_temp_start: float = 1.5
    gumbel_temp_end: float = 0.35

    # Grid codec.
    cell_dim: int = 64
    attention_heads: int = 10
    codec_dropout: float = 0.0

    # Adaptive operator residual around the frozen v30 algebra.
    operator_residual_rank: int = 4
    operator_residual_scale: float = 0.35
    feature_residual_scale: float = 0.10

    # Initialize from validated synthetic geometry when files are available.
    initialize_from_v30: bool = True
    neat_winner_path: str = "yetirah_v30_neat_winner.pkl"
    algebra_checkpoint_path: str = "yetirah_v30_algebra.pt"

    # Checkpoints.
    arc_checkpoint_path: str = "yetirah_arc_v1.pt"
    arc_best_checkpoint_path: str = "yetirah_arc_v1_best.pt"
