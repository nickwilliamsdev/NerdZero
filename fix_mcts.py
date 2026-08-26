with open("nerd_muzero/mcts/search.py", "r") as f:
    text = f.read()

# Make the MCTS tree nodes aware of the fast-weight memory state S
text = text.replace(
    'self.hidden_state = None\n        self.reward = 0',
    'self.hidden_state = None\n        self.memory_state = None\n        self.reward = 0'
)

# Initialize the root node's memory properly
text = text.replace(
    'root = Node(0)\n        root.hidden_state = initial_hidden_state',
    'root = Node(0)\n        root.hidden_state = initial_hidden_state\n        # Initialize fast weight S matrix mapped to DeltaNet dimensions\n        root.memory_state = torch.zeros(initial_hidden_state.shape[0], 128, 128, device=initial_hidden_state.device)'
)

# During MCTS expansion, pass the parent memory state and store the child memory state
text = text.replace(
    'next_hidden_state, reward = self.dynamics_network(parent.hidden_state, action_tensor)\n            node.hidden_state = next_hidden_state\n            node.reward = reward.item()',
    'next_hidden_state, next_memory, reward = self.dynamics_network(parent.hidden_state, parent.memory_state, action_tensor)\n            node.hidden_state = next_hidden_state\n            node.memory_state = next_memory\n            node.reward = reward.item()'
)

with open("nerd_muzero/mcts/search.py", "w") as f:
    f.write(text)
