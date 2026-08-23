import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class DeltaNetEncoder(nn.Module):
    """
    A DeltaNet-based sequence encoder.
    Maps a sequence of observations to a single MuZero latent representation.
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Projections to generate Query, Key, Value, and beta (update rate)
        self.q_proj = nn.Linear(input_dim, d_model)
        self.k_proj = nn.Linear(input_dim, d_model)
        self.v_proj = nn.Linear(input_dim, d_model)
        self.beta_proj = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass for DeltaNet.
        
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
            return_sequence: If True, returns states for all timesteps. 
                             Otherwise, returns only the final hidden state.
                             
        Returns:
            Tensor of shape (batch, d_model) if not return_sequence
            Tensor of shape (batch, seq_len, d_model) if return_sequence
        """
        b, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        # Apply non-linearity to keys for numeric stability (as commonly done in linear attention)
        k = F.elu(self.k_proj(x)) + 1.0 
        v = self.v_proj(x)
        # Beta is gated between 0 and 1
        beta = torch.sigmoid(self.beta_proj(x))

        # Initial memory state (Fast Weight matrix) (batch, d_model, d_model)
        S = torch.zeros(b, self.d_model, self.d_model, device=x.device)
        
        states = []
        
        # Recurrent application of the explicit Delta rule
        for t in range(seq_len):
            qt = q[:, t, :]    # (b, d_model)
            kt = k[:, t, :]    # (b, d_model)
            vt = v[:, t, :]    # (b, d_model)
            betat = beta[:, t, :] # (b, 1)

            # Read from memory with Query: h_t = S_{t-1} @ q_t
            # (b, d_model, d_model) x (b, d_model) -> (b, d_model)
            ht = einsum(S, qt, 'b d_k d_v, b d_k -> b d_v')
            states.append(ht)

            # Update memory using the Delta rule: S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) outer k_t
            # First compute expected reading for k_t
            read_kt = einsum(S, kt, 'b d_k d_v, b d_k -> b d_v')
            
            # Delta to apply: beta * (target_v - read_v)
            delta_v = betat * (vt - read_kt)
            
            # Outer product update
            update = einsum(delta_v, kt, 'b d_v, b d_k -> b d_k d_v')
            S = S + update
            
        outputs = torch.stack(states, dim=1) # (b, seq_len, d_model)
        
        if return_sequence:
            return outputs
            
        # Standard Representation network returns the final state
        return outputs[:, -1, :]
