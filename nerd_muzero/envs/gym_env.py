import gymnasium as gym
import numpy as np
import arckit
from gymnasium import spaces

class ARCEnv(gym.Env):
    """
    OpenAI Gym / Gymnasium environment wrapper for ARC (Abstraction and Reasoning Corpus).
    Supports loading tasks from ArcAGI2 using arckit.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, task_id=None, split="train", max_steps=100):
        super(ARCEnv, self).__init__()
        self.max_steps = max_steps
        self.current_step = 0
        
        # Load dataset
        # arckit load_data returns tuple (train_set, eval_set)
        train_set, _ = arckit.load_data()
        self.dataset = train_set.tasks
        
        self.task_id = task_id
        if self.task_id is None:
            # Defer task selection to reset() for per-episode randomization.
            self.task = self.dataset[0]
        else:
            self.task = next(t for t in self.dataset if t.id == self.task_id)
        
        # ARC grid values are integers from 0 to 9 representing colors
        # Environments typically output varying grid sizes, but for ML we often pad or flatten.
        # Assuming maximum grid size of 30x30
        self.max_grid_size = (30, 30)
        
        # Observation space: the grid (padded to 30x30)
        self.observation_space = spaces.Box(
            low=0, high=9, shape=self.max_grid_size, dtype=np.int32
        )
        
        # Action space: for a generic ARC solver agent, it could be pixel-by-pixel color selection,
        # or higher level edits. Here we use a generic discrete action space placeholder.
        # e.g., (x, y, color) -> 30 * 30 * 10 = 9000 actions
        self.action_space = spaces.Discrete(self.max_grid_size[0] * self.max_grid_size[1] * 10)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # If task_id is not fixed, sample a random ARC task each episode.
        if self.task_id is None:
            task_idx = self.np_random.integers(0, len(self.dataset))
            self.task = self.dataset[task_idx]
            self.current_task_id = self.task.id
        else:
            self.current_task_id = self.task_id

        # Select a random training pair from the selected task.
        example_idx = self.np_random.integers(0, len(self.task.train))
        self.current_example = self.task.train[example_idx]

        input_grid, output_grid = self.current_example
        self.state = np.array(input_grid)
        self.target = np.array(output_grid)

        return self._get_obs(), {"task_id": self.current_task_id}
    
    def _score_partial(self, state, target):
        hs, ws = state.shape
        ht, wt = target.shape

        # overlap region
        h = min(hs, ht)
        w = min(ws, wt)

        # 1) cell-level overlap match
        if h > 0 and w > 0:
            cell_match = float(np.mean(state[:h, :w] == target[:h, :w]))
        else:
            cell_match = 0.0

        # 2) color histogram similarity (10 ARC colors)
        s_hist = np.bincount(state.flatten(), minlength=10).astype(np.float32)
        t_hist = np.bincount(target.flatten(), minlength=10).astype(np.float32)
        s_hist /= max(1.0, s_hist.sum())
        t_hist /= max(1.0, t_hist.sum())
        hist_sim = 1.0 - 0.5 * float(np.abs(s_hist - t_hist).sum())  # in [0,1]

        # 3) shape distance penalty (normalized)
        shape_dist = (abs(hs - ht) + abs(ws - wt)) / float(max(ht + wt, 1))

        # weighted potential
        w_cell, w_color, w_shape = 0.65, 0.30, 0.05
        return w_cell * cell_match + w_color * hist_sim - w_shape * shape_dist
    
    def _get_obs(self):
        # Pad state to max_grid_size
        padded_state = np.zeros(self.max_grid_size, dtype=np.int32)
        h, w = self.state.shape
        padded_state[:h, :w] = self.state
        return padded_state
        
    def step(self, action):
        self.current_step += 1

        # Decode action to (x, y, color)
        color = action % 10
        rem = action // 10
        y = rem % self.max_grid_size[1]
        x = rem // self.max_grid_size[1]

        # Match score before edit
        prev_phi = self._score_partial(self.state, self.target)

        # Apply action if within current grid bounds
        h, w = self.state.shape
        if np.all(x < h) and np.all(y < w):
            self.state[x, y] = color

        # Match score after edit
        curr_phi = self._score_partial(self.state, self.target)

        reward = (curr_phi - prev_phi) - 0.001  # tiny step cost
        done = False

        if self.state.shape == self.target.shape and np.array_equal(self.state, self.target):
            reward += 1.0
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        truncated = False
        info = {"task_id": getattr(self, "current_task_id", self.task_id)}

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        pass # Optional visualization using arckit plotting
