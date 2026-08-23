import torch
import pytest
from nerd_muzero.mcts.search import MCTS, MinMaxStats
from nerd_muzero.models.recursive import DynamicsNetwork
from nerd_muzero.models.prediction import PredictionNetwork

@pytest.mark.unit
def test_mcts_run():
    latent_dim = 16
    action_dim = 1
    num_actions = 5
    
    # Initialize dummy networks
    dynamics = DynamicsNetwork(latent_dim=latent_dim, action_dim=action_dim)
    prediction = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)
    
    # Initialize MCTS
    mcts = MCTS(dynamics, prediction, num_simulations=10)
    min_max_stats = MinMaxStats()
    
    # Create an initial hidden state (batch_size=1)
    initial_hidden_state = torch.randn(1, latent_dim)
    
    # Run MCTS
    root_node = mcts.run(initial_hidden_state, min_max_stats)
    
    # Verify the root node expanded and simulated
    assert root_node.visit_count == 10, f"Expected 10 visits, got {root_node.visit_count}"
    assert len(root_node.children) == num_actions, f"Expected {num_actions} children, got {len(root_node.children)}"
    
    # Verify UCB stats bounds
    assert min_max_stats.maximum != -float('inf'), "MinMax stats maximum should be updated"

