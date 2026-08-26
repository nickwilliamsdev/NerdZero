import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class DynamicsNetwork(nn.Module):
    """
    A DeltaNet-based Causal Dynamics (World) Model for MuZero.
    Allows for both sequential inference (during MCTS) and parallel sequence 
    unrolling (during inner-loop training).
    
    Maps: (memory_state, action_seq) -> (next_memory_state, latent_seq, reward_seq)
    """
    def __init__(self, latent_dim: int, action_dim: int, d_model: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        # We embed the action combined with the current latent state features 
        self.input_proj = nn.Linear(latent_dim + action_dim, d_model)
        
        # DeltaNet causal parameters
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, 1)
        
        # Decoding projections
        self.out_state_proj = nn.Linear(d_model, latent_dim)
        self.reward_proj = nn.Linear(d_model, 1)
        
    def forward(self, state: torch.Tensor, memory: torch.Tensor, action: torch.Tensor):
        """
        Sequence-aware forward pass.
        
        Args:
            state: Tensor of shape (batch, seq_len, latent_dim) OR (batch, latent_dim)
            memory: Fast weight memory S of shape (batch, d_model, d_model). 
                    Pass torch.zeros during first step.
            action: Tensor of shape (batch, seq_len, action_dim) OR (batch, action_dim)
            
        Returns:
            next_state: Tensor of shape (batch, seq_len, latent_dim)
            next_memory: The updated fast-weight memory
            reward: Tensor of shape (batch, seq_len, 1)
        """
        # Auto-unsqueeze for MCTS single-step inference compatibility
        is_single_step = (state.dim() == 2)
        if is_single_step:
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
            
        b, seq_len, _ = state.shape
        
        # Combine latent limits with action sequence
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.input_proj(x))
        
        q = self.q_proj(x)
        k = F.elu(self.k_proj(x)) + 1.0 
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))
        
        S = memory
        states_out = []
        
        # Causal sequence unrolling (Fast auto-regressive evaluation)
        # Note: In a highly optimized CUDA implementation, this explicit loop 
        # is replaced by a chunkwise parallel associative scan (like in Mamba/DeltaNet).
        for t in range(seq_len):
            qt = q[:, t, :]
            kt = k[:, t, :]
            vt = v[:, t, :]
            betat = beta[:, t, :]

            # Retrieve latent feature from memory
            ht = einsum(S, qt, 'b d_k d_v, b d_k -> b d_v')
            states_out.append(ht)

            # Delta Memory Update
            read_kt = einsum(S, kt, 'b d_k d_v, b d_k -> b d_v')
            delta_v = betat * (vt - read_kt)
            update = einsum(delta_v, kt, 'b d_v, b d_k -> b d_k d_v')
            S = S + update
            
        # Repackage the sequence
        h_seq = torch.stack(states_out, dim=1)
        
        next_states = self.out_state_proj(h_seq)
        
        # Standardize state bounds (L2 Norm to combat exploding recursive states)
        next_states_norm = torch.norm(next_states, p=2, dim=-1, keepdim=True) + 1e-5
        max_norm = torch.max(next_states_norm, torch.ones_like(next_states_norm))
        next_states = next_states / max_norm 
        
        rewards = self.reward_proj(h_seq)
        
        if is_single_step:
            return next_states.squeeze(1), S, rewards.squeeze(1)
            
        return next_states, S, rewards
