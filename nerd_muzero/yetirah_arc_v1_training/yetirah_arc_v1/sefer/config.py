
    
  
ARC_PATCH_ID = "arc-scratch-v1"


from dataclasses import dataclass
import torch


@dataclass
class ARCConfig:
    """ARC-only scratch-training configuration; no imported model weights."""

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
    # One-step oracle remains a local stabilizer, while the beam teacher below
    # supplies composition-aware first-action targets.
    oracle_policy_weight: float = 0.15
    oracle_improvement_margin: float = 0.001
    beam_teacher_weight: float = 0.35
    beam_teacher_width: int = 4
    beam_teacher_lookahead: int = 2

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

