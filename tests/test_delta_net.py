import torch
import pytest
from nerd_muzero.models.delta_net import DeltaNetEncoder

@pytest.mark.unit
def test_delta_net_memory_update():
    # Setup parameters
    batch_size = 4
    seq_len = 10
    input_dim = 64
    d_model = 128
    
    # Initialize PyTorch module
    encoder = DeltaNetEncoder(input_dim=input_dim, d_model=d_model)
    
    # Create dummy observation sequence (Batch, Seq, Features)
    dummy_obs = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    latent_state = encoder(dummy_obs)
    
    # Assertions to ensure tensor routing and dimensions are correct
    assert latent_state.shape == (batch_size, d_model), \
        f"Expected shape {(batch_size, d_model)}, got {latent_state.shape}"
        
    assert latent_state.requires_grad, "DeltaNet output must remain attached to the compute graph for inner-loop loss."