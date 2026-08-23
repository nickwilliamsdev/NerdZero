import torch
import torch.optim as optim
import numpy as np
from nerd_muzero.mcts.search import MCTS, MinMaxStats

class MuZeroAgent:
    """
    Inner loop actor and optimizer for MuZero. 
    Handles interaction with the environment through MCTS.
    """
    def __init__(self, encoder, dynamics, prediction, env, config):
        self.encoder = encoder
        self.dynamics = dynamics
        self.prediction = prediction
        self.env = env
        self.config = config
        
        # Join parameters for inner-loop gradient-based training
        params = list(encoder.parameters()) + list(dynamics.parameters()) + list(prediction.parameters())
        self.optimizer = optim.Adam(params, lr=config.get("lr", 1e-3))
        
        self.mcts = MCTS(dynamics, prediction, num_simulations=config.get("num_simulations", 30))
        
    def act(self, obs, min_max_stats, temperature=1.0):
        # Flatten observation for embedding
        obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            latent_state = self.encoder(obs_flat)
            
        root = self.mcts.run(latent_state, min_max_stats)
        
        action_visits = []
        actions = []
        for action, child in root.children.items():
            actions.append(action)
            action_visits.append(child.visit_count)
            
        if sum(action_visits) == 0:
            # Fallback to random uniform if no visits (shouldn't happen with proper MCTS)
            return np.random.randint(self.prediction.num_actions), root
            
        action_probs = np.array(action_visits) / sum(action_visits)
        
        if temperature == 0:
            action = actions[np.argmax(action_probs)]
        else:
            action = np.random.choice(actions, p=action_probs)
            
        return action, root
        
    def play_episode(self, temperature=1.0):
        obs, info = self.env.reset()
        done = False
        min_max_stats = MinMaxStats()
        
        history = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "policies": []
        }
        
        steps = 0
        max_steps = self.config.get("max_episode_steps", 50)
        
        while not done and steps < max_steps:
            # 1. Run MCTS to get policy and value
            action, root = self.act(obs, min_max_stats, temperature=temperature)
            
            # 2. Decode scalar action back to Grid API format (x, y, color)
            # Action space size is H * W * 10
            h, w = self.env.max_grid_size
            x = (action // (w * 10)) % h
            y = (action // 10) % w
            color = action % 10
            env_action = np.array([x, y, color])
            
            # 3. Environment transition
            next_obs, reward, terminated, truncated, _ = self.env.step(env_action)
            done = terminated or truncated
            
            # 4. Record statistics for inner loop training
            policy_target = np.zeros(self.prediction.num_actions)
            for a, child in root.children.items():
                if root.visit_count > 0:
                    policy_target[a] = child.visit_count / root.visit_count
                
            history["obs"].append(obs.flatten())
            history["actions"].append(action)
            history["rewards"].append(reward)
            history["values"].append(root.value())
            history["policies"].append(policy_target)
            
            obs = next_obs
            steps += 1
            
        return history

    def train_step(self, replay_buffer):
        """
        Placeholder for the PyTorch inner-loop BPTT (Backpropagation Through Time).
        This unravels the dynamics model matching MCTS trajectories.
        """
        self.optimizer.zero_grad()
        # To be implemented: Sample batch -> compute loss -> backward -> step
        # loss = value_loss + policy_loss + reward_loss 
        pass
