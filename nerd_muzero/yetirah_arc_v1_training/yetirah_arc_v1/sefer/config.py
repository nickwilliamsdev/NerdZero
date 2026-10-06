from __future__ import annotations

ARC_PATCH_ID = "arc-scratch-v3-complete"
ARC_RECURSIVE_TRAINING_REV = "soft-state-ranker-v7"

from dataclasses import dataclass
import torch


@dataclass
class ARCConfig:
    """Yetirah ARC recursive-v1 defaults.

    ARC tasks are trained directly.  The active reasoning path is a
    task-conditioned recursive DeltaNet with learned fast state.
    """

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    matmul_precision: str = "high"

    # ARC representation.
    max_grid_size: int = 30
    color_count: int = 10
    max_demos: int = 4
    n_slots: int = 32
    node_dim: int = 32
    rule_dim: int = 128
    cell_dim: int = 96
    attention_heads: int = 4
    codec_dropout: float = 0.0
    codec_latent_layers: int = 2

    # Dataset.
    split: str = "train"
    validation_fraction: float = 0.10
    limit_tasks: int | None = None
    batch_size: int = 64
    eval_tasks: int = 64
    num_workers: int = 0

    # Training curriculum.  program_steps is retained as the CLI-compatible
    # name for the recursive-reasoning phase.
    codec_steps: int = 1500
    direct_steps: int = 1500
    operator_discovery_steps: int = 0
    program_steps: int = 1500
    diagnostic_every: int = 50
    eval_every: int = 200

    # Optimizer.
    codec_lr: float = 3e-4
    direct_lr: float = 3e-4
    recursive_lr: float = 2e-4
    recursive_unfreeze_rule_encoder: bool = False
    recursive_rule_lr_scale: float = 0.10
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Grid losses.
    foreground_boost: float = 1.25
    color_balance_mix: float = 0.35
    shape_weight: float = 0.25

    # Direct/meta-learning.
    direct_latent_weight: float = 0.20
    loo_demo_weight: float = 0.35
    recursive_loo_weight: float = 1.25
    recursive_step_improvement_weight: float = 0.50
    recursive_step_improvement_margin: float = 0.01

    # Recursive DeltaNet.
    recursive_steps: int = 6
    recursive_heads: int = 4
    fast_state_dim: int = 32
    fast_update_scale: float = 0.35
    fast_decay_min: float = 0.90
    fast_decay_max: float = 0.999
    recursive_residual_scale: float = 0.50
    recursive_grid_weight: float = 1.0
    recursive_latent_weight: float = 0.35
    recursive_intermediate_weight: float = 0.15
    recursive_consistency_weight: float = 0.10
    recursive_fast_reg_weight: float = 0.002
    halt_weight: float = 0.02
    halt_threshold: float = 0.90
    adaptive_halt_weight: float = 0.20
    state_rank_temperature: float = 0.35
    state_rank_listwise_weight: float = 1.00
    state_rank_pairwise_weight: float = 0.50
    state_rank_regret_weight: float = 0.25
    state_rank_pairwise_margin: float = 0.00
    recursive_per_step_grid_weight: float = 0.15
    adaptive_halt_use_argmax: bool = True
    adaptive_halt_min_step: int = 1
    recursive_gain_score_weight: float = 0.50
    recursive_early_stop_patience: int = 4
    recursive_early_stop_min_delta: float = 0.001
    recursive_min_steps_before_stop: int = 400

    # Checkpoints.
    arc_checkpoint_path: str = "yetirah_arc_recursive_v1.pt"
    arc_best_checkpoint_path: str = "yetirah_arc_recursive_v1_best.pt"
    arc_pre_neat_checkpoint_path: str = "yetirah_arc_recursive_v1.pt"
    restore_best_at_end: bool = True

    # Launcher compatibility only. NEAT is intentionally inactive in recursive-v1.
    arc_neat_enabled: bool = False
    arc_neat_generations: int = 0
    arc_neat_population: int = 0
    arc_neat_eval_tasks: int = 0
    arc_neat_config_path: str | None = None
