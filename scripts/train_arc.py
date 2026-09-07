import os
import sys

# Ensure nerd_muzero modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import neat
from nerd_muzero.envs.gym_env import ARCEnv
from nerd_muzero.training.outer_loop_neat import evaluation_hook
from nerd_muzero.tensorneat_pt.evolution import run_neat_evolution

def run_experiment():
    print("Starting NEAT-driven MuZero training loop...")
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(local_dir, '../neat_config.txt'))
    
    # ARC data path (dummy placeholder for now, the wrapper handles missing data gracefully)
    data_path = os.path.abspath(os.path.join(local_dir, '../data/arc_tasks'))
    
    # Load configuration
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    
    # Initialize the ARC environment
    env = ARCEnv(max_steps=20)
    
    # Initialize Population
    p = neat.Population(config)

    checkpointer = neat.Checkpointer(generation_interval=5, 
                                    time_interval_seconds=None, 
                                    filename_prefix='./checkpoints/neat-checkpoint-')

    # 3. Add the checkpointer to the population reporters
    p.add_reporter(checkpointer)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Create the closure for the evaluation hook mapping
    def eval_genomes(genomes, config):
        evaluation_hook(genomes, config, env)
        
    print(f"Beginning evolution across {config.pop_size} genomes per generation.")
    # Run evolution
    winner = p.run(eval_genomes, n=100)  # Limited to 2 generations for testing
    
    print("\nBest genome found:")
    print(winner)
    
    # Save the winner safely to be loaded by evaluate_agent.py
    import pickle
    winner_path = os.path.abspath(os.path.join(local_dir, "../best_genome.pkl"))
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"Saved best genome to {winner_path}")

if __name__ == '__main__':
    run_experiment()
