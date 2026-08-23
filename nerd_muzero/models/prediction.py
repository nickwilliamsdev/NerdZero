import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionNetwork(nn.Module):
    """
    The Prediction core of MuZero.
    Maps latent_state -> (policy_logits, value).
    """
    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        # Value head
        self.val_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.val_fc2 = nn.Linear(hidden_dim, 1)
        
        # Policy head
        self.pol_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.pol_fc2 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Tensor of shape (batch, latent_dim)
            
        Returns:
            policy_logits: Tensor of shape (batch, num_actions)
            value: Tensor of shape (batch, 1)
        """
        # Value forward
        v = F.relu(self.val_fc1(state))
        value = self.val_fc2(v)
        
        # Policy forward
        p = F.relu(self.pol_fc1(state))
        policy_logits = self.pol_fc2(p)
        
        return policy_logits, value
