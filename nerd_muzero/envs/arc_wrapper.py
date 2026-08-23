import numpy as np
import torch
import einops
from typing import Any, Dict, Tuple, Optional
# Assuming gymnasium is used despite not being explicitly in requirements.txt yet
# If not installed, you can use a custom class with the same signature.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class gym:
        Env = object
    class spaces:
        Box = type('Box', (), {})
        Discrete = type('Discrete', (), {})

import arckit

class ARCWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for the Abstraction and Reasoning Corpus (ARC).
    
    This environment presents an ARC task grid as an observation.
    Actions can involve modifying grid pixels or submitting a final answer.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, data_path: str, max_grid_size: tuple[int, int] = (30, 30), render_mode: Optional[str] = None):
        super().__init__()
        
        self.data_path = data_path
        self.max_grid_size = max_grid_size
        self.render_mode = render_mode
        
        # Load tasks using arckit
        try:
            self.tasks = list(arckit.load_data(data_path))
        except Exception as e:
            # Fallback if ARCKit usage is different or data path is invalid
            print(f"Warning: Could not load tasks from {data_path} with arckit: {e}")
            self.tasks = []
            
        self.current_task = None
        self.current_example = None
        self.current_grid = np.zeros((1, 1), dtype=np.int32)
        
        # Observation space: A grid of size 'max_grid_size' containing colors 0-9 
        # (padded if smaller than max size)
        self.observation_space = spaces.Box(
            low=0, high=9, 
            shape=(self.max_grid_size[0], self.max_grid_size[1]), 
            dtype=np.int32
        )
        
        # Action space: To be defined based on how the agent interacts.
        # e.g., (x, y, color) or submit action
        # This is a placeholder for a flat discrete action or a multidiscrete action.
        self.action_space = spaces.MultiDiscrete([
            self.max_grid_size[0], # x coordinate
            self.max_grid_size[1], # y coordinate
            10                     # color (0-9)
        ])

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to a new training/test example from a new ARC task.
        """
        super().reset(seed=seed)
        
        if not self.tasks:
            self.current_grid = np.zeros(self.max_grid_size, dtype=np.int32)
            obs = np.zeros(self.max_grid_size, dtype=np.int32)
            return obs, {"error": "No tasks loaded"}
        
        # Select a random task
        self.current_task = np.random.choice(self.tasks)
        
        # Select a random train or test example from the task
        # Format depends on arckit internals, assuming standard train/test splits
        examples = getattr(self.current_task, 'train', [])
        if examples:
            self.current_example = np.random.choice(examples)
            # Assuming 'input' is the input grid as a list of lists or numpy array
            input_grid = np.array(getattr(self.current_example, 'input', []))
            self.current_grid = input_grid.copy()
        else:
            self.current_grid = np.zeros((1, 1), dtype=np.int32)
            
        obs = self._get_padded_obs()
        info = {"task_id": getattr(self.current_task, 'id', 'unknown')}
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes an action and updates the environment state.
        
        Args:
            action (np.ndarray): The multi-discrete action [x, y, color]
        """
        x, y, color = action
        
        # Bound check (Ensure we only modify within the active, unpadded grid)
        valid_x = 0 <= x < self.current_grid.shape[0]
        valid_y = 0 <= y < self.current_grid.shape[1]
        
        if valid_x and valid_y:
            self.current_grid[x, y] = color
            
        # Define reward and termination conditions
        # In a real ARC scenario, the episode terminates when an agent issues a "submit" 
        # action or after a max number of steps.
        terminated = False 
        truncated = False
        reward = 0.0
        
        # E.g., if the user submitted and it exactly matches the output grid
        # target_grid = np.array(self.current_example.output)
        # if np.array_equal(self.current_grid, target_grid):
        #     reward = 1.0
        #     terminated = True

        obs = self._get_padded_obs()
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _get_padded_obs(self) -> np.ndarray:
        """
        Pads the current grid to match the expected observation space max dimension.
        """
        obs = np.zeros(self.max_grid_size, dtype=np.int32)
        if self.current_grid is not None and self.current_grid.size > 0:
            h, w = self.current_grid.shape
            obs[:h, :w] = self.current_grid
        return obs

    def render(self):
        """
        Renders the current grid state.
        """
        if self.render_mode == "human":
            print(self.current_grid)
        elif self.render_mode == "rgb_array":
            # Map 0-9 colors to standard ARC RGB values here
            pass

