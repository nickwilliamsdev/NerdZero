import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsNetwork(nn.Module):
    """
    The Recursive Dynamics core of MuZero.
    Maps (latent_state, action) -> (next_latent_state, reward).
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Merge latent state and action
        self.fc_layer_1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        
        # Predict next latent state
        self.fc_state = nn.Linear(hidden_dim, latent_dim)
        
        # Predict support/scalar reward
        # E.g. raw immediate reward prediction
        self.fc_reward = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Tensor of shape (batch, latent_dim)
            action: Tensor of shape (batch, action_dim) - assuming one-hot or flat embedded representation
            
        Returns:
            next_state: Tensor of shape (batch, latent_dim)
            reward: Tensor of shape (batch, 1)
        """
        # Concatenate on feature dimension
        x = torch.cat([state, action], dim=-1)
        
        x = F.relu(self.fc_layer_1(x))
        
        # The next state should have the same bound constraints if any (often normalized in MuZero)
        next_state = self.fc_state(x)
        
        # Normalize the next latent state (common practice in MuZero to bound activations)
        # Using min-max normalization via minmax scaling or simply layernorm
        # Here we'll do a simple standardization (or user might want custom scaling)
        # normalize across feature dim to avoid exploding states in recursion.
        next_state_norm = torch.norm(next_state, p=2, dim=-1, keepdim=True) + 1e-5
        
        # Scale to max norm of 1
        max_norm = torch.max(next_state_norm, torch.ones_like(next_state_norm))
        next_state = next_state / max_norm 
        
        reward = self.fc_reward(x)
        
        return next_state, reward
