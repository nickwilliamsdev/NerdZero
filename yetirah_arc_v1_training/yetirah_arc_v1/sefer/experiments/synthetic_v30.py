from yetirah_arc_v1_training.yetirah_arc_v1.sefer.config import V30Config
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.training.trainer import train_smoke_test

def run(cfg: V30Config | None = None):
    cfg = cfg or V30Config()
    return train_smoke_test(
        steps=cfg.steps,
        batch_size=cfg.batch_size,
        inner_rollout_steps=cfg.inner_rollout_steps,
        warmup_steps=cfg.warmup_steps,
        algebra_steps=cfg.algebra_steps,
        es_every=cfg.es_every,
        diagnostic_every=cfg.diagnostic_every,
        mcts_train_every=cfg.mcts_train_every,
        mcts_train_samples=cfg.mcts_train_samples,
        mcts_simulations=cfg.mcts_simulations,
        mcts_eval_simulations=cfg.mcts_eval_simulations,
        mcts_eval_batch=cfg.mcts_eval_batch,
        neat_generations=cfg.neat_generations,
        neat_population=cfg.neat_population,
        neat_workers=cfg.neat_workers,
        neat_inner_steps=cfg.neat_inner_steps,
        neat_inner_lr=cfg.neat_inner_lr,
        neat_seed=cfg.neat_seed,
        device=cfg.device,
        reuse_neat_winner=cfg.reuse_neat_winner,
        neat_winner_path=cfg.neat_winner_path,
        resume_from_algebra=cfg.resume_from_algebra,
        algebra_checkpoint_path=cfg.algebra_checkpoint_path,
        posttrain_checkpoint_path=cfg.posttrain_checkpoint_path,
        materialize_frozen_operators=cfg.materialize_frozen_operators,
        matmul_precision=cfg.matmul_precision,
    )

if __name__ == "__main__":
    run()
