"""Optional Gymnasium wrapper for ARC.

ARC-v1 meta-training does NOT use the 9000-action pixel-edit interface because
that would bypass the learned operator/program hierarchy.  Keep this wrapper
for later RL/controller experiments and interactive evaluation.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # allow importing the rest of sefer without gymnasium
    gym = None
    spaces = None


if gym is not None:
    class ARCEnv(gym.Env):
        metadata = {"render_modes": ["human", "rgb_array"]}

        def __init__(self, task_id=None, split="train", max_steps=100):
            super().__init__()
            import arckit
            self.max_steps = max_steps
            self.current_step = 0
            train_set, eval_set = arckit.load_data()
            ds = train_set if split.lower() in {"train", "training"} else eval_set
            self.dataset = list(ds.tasks)
            self.task_id = task_id
            self.task = self.dataset[0] if task_id is None else next(t for t in self.dataset if t.id == task_id)
            self.max_grid_size = (30, 30)
            self.observation_space = spaces.Box(low=0, high=9, shape=self.max_grid_size, dtype=np.int32)
            self.action_space = spaces.Discrete(30 * 30 * 10)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            if self.task_id is None:
                self.task = self.dataset[int(self.np_random.integers(0, len(self.dataset)))]
            self.current_task_id = self.task.id
            example_idx = int(self.np_random.integers(0, len(self.task.train)))
            input_grid, output_grid = self.task.train[example_idx]
            self.state = np.asarray(input_grid, dtype=np.int32).copy()
            self.target = np.asarray(output_grid, dtype=np.int32)
            return self._get_obs(), {"task_id": self.current_task_id}

        def _score_partial(self, state, target):
            hs, ws = state.shape
            ht, wt = target.shape
            h, w = min(hs, ht), min(ws, wt)
            cell_match = float(np.mean(state[:h, :w] == target[:h, :w])) if h and w else 0.0
            s_hist = np.bincount(state.ravel(), minlength=10).astype(np.float32)
            t_hist = np.bincount(target.ravel(), minlength=10).astype(np.float32)
            s_hist /= max(1.0, s_hist.sum())
            t_hist /= max(1.0, t_hist.sum())
            hist_sim = 1.0 - 0.5 * float(np.abs(s_hist - t_hist).sum())
            shape_dist = (abs(hs - ht) + abs(ws - wt)) / float(max(ht + wt, 1))
            return 0.65 * cell_match + 0.30 * hist_sim - 0.05 * shape_dist

        def _get_obs(self):
            out = np.zeros(self.max_grid_size, dtype=np.int32)
            h, w = self.state.shape
            out[:h, :w] = self.state
            return out

        def step(self, action):
            self.current_step += 1
            color = int(action % 10)
            rem = int(action // 10)
            y = rem % 30
            x = rem // 30
            prev_phi = self._score_partial(self.state, self.target)
            h, w = self.state.shape
            if x < h and y < w:
                self.state[x, y] = color
            curr_phi = self._score_partial(self.state, self.target)
            terminated = bool(np.array_equal(self.state, self.target))
            truncated = self.current_step >= self.max_steps and not terminated
            if terminated:
                reward = 10.0
            elif truncated:
                reward = -1.0
            else:
                reward = (curr_phi - prev_phi) * 10.0 - 0.001
            return self._get_obs(), reward, terminated, truncated, {"task_id": self.current_task_id}

        def render(self):
            return None
else:
    class ARCEnv:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("ARCEnv requires gymnasium")
