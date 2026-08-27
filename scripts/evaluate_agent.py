import os
import sys
import pickle
import torch
import neat

# Ensure nerd_muzero modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nerd_muzero.envs.gym_env import ARCEnv
from nerd_muzero.training.outer_loop_neat import build_models, get_substrate_coords
from nerd_muzero.tensorneat_pt.cppn import build_cppn
from nerd_muzero.training.inner_loop import MuZeroAgent

def evaluate_best_genome(genome_path, config_path, data_path):
    print(f"Loading evaluated genome from: {genome_path}")
    
    if not os.path.exists(genome_path):
        print(f"Error: Genome file not found at {genome_path}.")
        print("Please train an agent and save the best genome using pickle first.")
        return

    # Load configuration
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    
    # Load the winner genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
        
    # Initialize the ARC environment with rendering meant for human evaluation
    env = ARCEnv(max_steps=20)
    
    h, w = env.max_grid_size
    input_dim = h * w
    latent_dim = 128
    
    # 1. HyperNEAT: Construct Multi-output CPPN
    leaf_names = ['x1', 'y1', 'x2', 'y2']
    node_names = ['weight_q', 'weight_k', 'weight_v', 'weight_beta']
    cppn_net = build_cppn(genome, config, leaf_names, node_names)
    
    # 2. Build local agent architecture
    encoder, dynamics, prediction = build_models(env)
    x1, y1, x2, y2 = get_substrate_coords(input_dim, latent_dim)
    
    # 3. Query the Substrate and overwrite PyTorch model weights
    with torch.no_grad():
        kwargs = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        wq = cppn_net[0](**kwargs)
        wk = cppn_net[1](**kwargs)
        wv = cppn_net[2](**kwargs)
        wbeta = cppn_net[3](**kwargs)
        
        encoder.q_proj.weight.data = wq.view(input_dim, latent_dim).t()
        encoder.k_proj.weight.data = wk.view(input_dim, latent_dim).t()
        encoder.v_proj.weight.data = wv.view(input_dim, latent_dim).t()
        encoder.beta_proj.weight.data = wbeta[:input_dim].view(input_dim, 1).t()
        
    # 4. Instantiate the inner agent 
    # Use 0 temperature for deterministic evaluation
    agent_config = {
        "lr": 0.0, 
        "num_simulations": 50, # High simulations for evaluation
        "max_episode_steps": 20
    }
    agent = MuZeroAgent(encoder, dynamics, prediction, env, agent_config)
    
    print("Playing validation episode...")
    history = agent.play_episode(temperature=0.0)
    
    total_reward = sum(history["rewards"])
    print(f"\nEvaluation Complete! Total Reward: {total_reward}")
    print(f"Total Steps Taken: {len(history['obs'])}")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    genome_file = os.path.abspath(os.path.join(local_dir, '../best_genome.pkl'))
    config_file = os.path.abspath(os.path.join(local_dir, '../neat_config.txt'))
    data_folder = os.path.abspath(os.path.join(local_dir, '../data/arc_tasks'))
    
    evaluate_best_genome(genome_file, config_file, data_folder)
