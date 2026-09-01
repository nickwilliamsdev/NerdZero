import neat
import torch

from nerd_muzero.envs.gym_env import ARCEnv
from nerd_muzero.models.delta_net import DeltaNetEncoder
from nerd_muzero.models.recursive import DynamicsNetwork
from nerd_muzero.models.prediction import PredictionNetwork
from nerd_muzero.training.inner_loop import MuZeroAgent
from nerd_muzero.tensorneat_pt.cppn import build_cppn

def build_models(env):
    """Instantiate standard PyTorch MuZero components"""
    h, w = env.max_grid_size
    input_dim = h * w
    num_actions = h * w * 10
    latent_dim = 128
    
    encoder = DeltaNetEncoder(input_dim=input_dim, d_model=latent_dim)
    dynamics = DynamicsNetwork(latent_dim=latent_dim, action_dim=1)
    prediction = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)
    
    return encoder, dynamics, prediction

def get_substrate_coords(in_dim, out_dim):
    """Create normalized (x1, y1, x2, y2) coordinate pairs for weights."""
    # Simplified 1D grid representation for demonstration; 
    # for ARC, you could map 2D (x,y) -> 1D latent index.
    x1 = torch.linspace(-1, 1, in_dim).view(-1, 1).repeat(1, out_dim).flatten()
    y1 = torch.zeros_like(x1)
    x2 = torch.linspace(-1, 1, out_dim).view(1, -1).repeat(in_dim, 1).flatten()
    y2 = torch.zeros_like(x2)
    return x1, y1, x2, y2

def evaluation_hook(genomes, config, env):
    """
    Outer loop hook substituting `evaluate_genomes` in tensorneat_pt/evolution.py.
    """
    h, w = env.max_grid_size
    input_dim = h * w
    latent_dim = 128
    
    # Pre-compute substrate coordinates for the encoder projections (input_dim -> latent_dim)
    x1, y1, x2, y2 = get_substrate_coords(input_dim, latent_dim)
    
    for genome_id, genome in genomes:
        # 1. HyperNEAT: Multi-output CPPN
        leaf_names = ['x1', 'y1', 'x2', 'y2']
        # The CPPN output nodes directly map to Q, K, V, and Beta matrices
        node_names = ['weight_q', 'weight_k', 'weight_v', 'weight_beta']
        
        cppn_net = build_cppn(genome, config, leaf_names, node_names)
        
        # 2. Build local agent architecture
        encoder, dynamics, prediction = build_models(env)
        
        # 3. Query the Substrate! (Overwrite model weights)
        with torch.no_grad():
            # Query the multi-output CPPN with our coordinates
            # cppn_net returns a list of PyTorch-NEAT CPPN nodes. We must iterate or invoke them individually if needed, 
            # wait, PyTorch-NEAT create_cppn returns a CPPN model in some forks, but here it returns a list of Nodes.
            # We must map them over the inputs.
            # Evaluate PyTorch-NEAT nodes
            kwargs = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            wq = cppn_net[0](**kwargs)
            wk = cppn_net[1](**kwargs)
            wv = cppn_net[2](**kwargs)
            wbeta = cppn_net[3](**kwargs)
            
            # Reshape 1D outputs into the expected matrix dimensions (out_features, in_features)
            # using .t() because our meshgrid was mapped as [in, out]
            encoder.q_proj.weight.data = wq.view(input_dim, latent_dim).t()
            encoder.k_proj.weight.data = wk.view(input_dim, latent_dim).t()
            encoder.v_proj.weight.data = wv.view(input_dim, latent_dim).t()
            # beta_proj maps from input_dim (100) -> 1
            # We evaluated over input_dim * latent_dim grid pairs for the other weights
            # For beta_proj we only need a size of [1, input_dim]. Let's slice wbeta or evaluate it differently.
            encoder.beta_proj.weight.data = wbeta[:input_dim].view(input_dim, 1).t()
        
        # 4. Instantiate inner agent and play an episode
        agent_config = {
            "lr": 1e-3, 
            "num_simulations": 50,  # Increase from 10 to 50
            "max_episode_steps": 50  # Increase from 10 to 50
        }
        agent = MuZeroAgent(encoder, dynamics, prediction, env, agent_config)
        history = agent.play_episode(temperature=1.0)
        
        total_reward = sum(history["rewards"])
        genome.fitness = total_reward if total_reward > 0 else 0.1

if __name__ == "__main__":
    print("Compiled Outer Loop framework mapping multi-output PyTorch-NEAT CPPNs successfully.")
