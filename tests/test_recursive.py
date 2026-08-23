import torch
import pytest
from nerd_muzero.models.recursive import DynamicsNetwork
from nerd_muzero.models.prediction import PredictionNetwork

@pytest.mark.unit
def test_dynamics_network():
    batch_size = 4
    latent_dim = 128
    action_dim = 1
    
    dynamics = DynamicsNetwork(latent_dim=latent_dim, action_dim=action_dim)
    
    dummy_state = torch.randn(batch_size, latent_dim)
    dummy_action = torch.randn(batch_size, action_dim)
    
    next_state, reward = dynamics(dummy_state, dummy_action)
    
    assert next_state.shape == (batch_size, latent_dim), \
        f"Expected next_state shape {(batch_size, latent_dim)}, got {next_state.shape}"
    assert reward.shape == (batch_size, 1), \
        f"Expected reward shape {(batch_size, 1)}, got {reward.shape}"
    
    assert next_state.requires_grad, "Dynamics output must require gradients."

@pytest.mark.unit
def test_prediction_network():
    batch_size = 4
    latent_dim = 128
    num_actions = 100
    
    prediction = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)
    
    dummy_state = torch.randn(batch_size, latent_dim)
    
    policy_logits, value = prediction(dummy_state)
    
    assert policy_logits.shape == (batch_size, num_actions), \
        f"Expected policy_logits shape {(batch_size, num_actions)}, got {policy_logits.shape}"
    assert value.shape == (batch_size, 1), \
        f"Expected value shape {(batch_size, 1)}, got {value.shape}"
        
    assert policy_logits.requires_grad, "Prediction output must require gradients."
