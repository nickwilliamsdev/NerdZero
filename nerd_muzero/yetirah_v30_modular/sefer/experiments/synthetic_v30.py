from sefer.config import V30Config
from sefer.training.trainer import train_smoke_test

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
    )

if __name__ == "__main__":
    run()
