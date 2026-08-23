with open("nerd_muzero/envs/arc_wrapper.py", "r") as f:
    text = f.read()

# Replace missing data path logic to ensure dummy data grid gets bound
text = text.replace(
    'self.current_example = None\n        self.current_grid = None',
    'self.current_example = None\n        self.current_grid = np.zeros((1, 1), dtype=np.int32)'
)

text = text.replace(
    'if not self.tasks:\n            obs = np.zeros(self.max_grid_size, dtype=np.int32)\n            return obs, {"error": "No tasks loaded"}',
    'if not self.tasks:\n            self.current_grid = np.zeros(self.max_grid_size, dtype=np.int32)\n            obs = np.zeros(self.max_grid_size, dtype=np.int32)\n            return obs, {"error": "No tasks loaded"}'
)

with open("nerd_muzero/envs/arc_wrapper.py", "w") as f:
    f.write(text)
