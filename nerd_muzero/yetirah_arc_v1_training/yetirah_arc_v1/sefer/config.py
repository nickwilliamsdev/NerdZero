from dataclasses import dataclass
import os
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

    reuse_neat_winner: bool = True
    neat_winner_path: str = "yetirah_v30_neat_winner.pkl"
    resume_from_algebra: bool = False
    algebra_checkpoint_path: str = "yetirah_v30_algebra.pt"
    posttrain_checkpoint_path: str = "yetirah_v30_posttrain.pt"
    materialize_frozen_operators: bool = True
    matmul_precision: str = "high"


@dataclass
class V30DGXConfig(V30Config):
    batch_size: int = 256
    mcts_eval_batch: int = 64
    neat_workers: int = min(16, os.cpu_count() or 12)
    reuse_neat_winner: bool = True
    materialize_frozen_operators: bool = True


@dataclass
class ARCConfig:
    """ARC-v1.4 meta-learning defaults.

    v1.3 keeps the v30 prior and adds:
      * blended raw/balanced grid reconstruction
      * best-direct checkpoint restore before operator learning
      * a dedicated ARC operator-discovery warmup
      * frozen task inference during program composition
      * automatic restore of the best validation checkpoint at the end
      * task-conditioned CPPN-initialized fast operator networks
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
    num_workers: int = 0

    # Three-stage curriculum.
    codec_steps: int = 1500
    direct_steps: int = 1500
    operator_discovery_steps: int = 1500
    program_steps: int = 4000
    diagnostic_every: int = 50
    eval_every: int = 500

    # Optimizer.
    codec_lr: float = 3e-4
    direct_lr: float = 3e-4
    program_lr: float = 3e-4
    operator_discovery_lr: float = 2e-4
    operator_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    matmul_precision: str = "high"

    # Program-learning losses.
    latent_weight: float = 1.0
    grid_weight: float = 1.0
    shape_weight: float = 0.25
    direct_aux_weight: float = 0.25
    goal_consistency_weight: float = 0.50
    length_weight: float = 0.005
    operator_reg_weight: float = 0.01
    usage_balance_weight: float = 0.001
    value_weight: float = 0.10
    oracle_policy_weight: float = 0.35
    oracle_improvement_margin: float = 0.001

    # Operator-discovery warmup.  All 22 residual operators compete to explain
    # real ARC input->output latent transitions before the controller learns
    # compositions.  Softmin gives every useful candidate gradient while the
    # load-balancing term discourages collapse to one operator.
    operator_discovery_temp_start: float = 0.20
    operator_discovery_temp_end: float = 0.04
    operator_discovery_grid_weight: float = 0.35
    operator_discovery_usage_weight: float = 0.02
    operator_discovery_reg_weight: float = 0.01
    operator_discovery_trust_weight: float = 0.03
    operator_discovery_val_batches: int = 4
    gumbel_temp_start: float = 1.50
    gumbel_temp_end: float = 0.75

    # Progressive action/program curriculum. Fractions are cumulative progress
    # boundaries through program training. Each stage uses only the first K ARC
    # operators and at most D actions before STOP.
    program_stage_fractions: tuple[float, float, float] = (0.20, 0.55, 0.80)
    program_stage_active_ops: tuple[int, int, int, int] = (4, 8, 12, 22)
    program_stage_depths: tuple[int, int, int, int] = (1, 2, 3, 4)

    # Grid codec.
    cell_dim: int = 96
    attention_heads: int = 4
    codec_dropout: float = 0.0
    codec_latent_layers: int = 2
    foreground_boost: float = 1.25
    color_balance_mix: float = 0.35

    # Task-conditioned fast operator network around the frozen v30/CPPN algebra.
    # The CPPN-generated transports become W0; a rule+operator hypernetwork emits
    # low-rank task-local fast weights. A smaller static residual remains as a
    # trainable ARC-wide correction.
    fast_operator_rank: int = 8
    fast_operator_static_rank: int = 4
    fast_operator_code_dim: int = 16
    fast_operator_hidden_dim: int = 192
    fast_transport_delta_scale: float = 0.35
    fast_static_delta_scale: float = 0.15
    fast_feature_delta_scale: float = 0.08
    fast_gate_init: float = -1.0
    fast_gate_max: float = 0.50
    fast_delta_rms_cap: float = 1.0
    fast_transport_kl_weight: float = 0.02
    fast_gate_penalty_weight: float = 0.005

    # Safe program residual: the direct predicted goal remains an inference-time
    # fallback. The program learns only a gated residual around that prediction.
    program_blend_init: float = -1.5

    # Legacy aliases retained so older launcher/config code does not break.
    operator_residual_rank: int = 8
    operator_residual_scale: float = 0.50
    feature_residual_scale: float = 0.10

    # Initialize from validated synthetic geometry when files are available.
    initialize_from_v30: bool = True
    neat_winner_path: str = "yetirah_v30_neat_winner.pkl"
    algebra_checkpoint_path: str = "yetirah_v30_algebra.pt"
    checkpoint_search_depth: int = 3
    require_v30_init: bool = True

    # Checkpoints.
    arc_checkpoint_path: str = "yetirah_arc_v1.pt"
    arc_best_checkpoint_path: str = "yetirah_arc_v1_best.pt"
    restore_best_at_end: bool = True
