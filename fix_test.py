with open("tests/test_recursive.py", "r") as f:
    text = f.read()

text = text.replace(
    'next_state, reward = dynamics(dummy_state, dummy_action)',
    'dummy_memory = torch.zeros(batch_size, 128, 128)\n    next_state, next_memory, reward = dynamics(dummy_state, dummy_memory, dummy_action)'
)

with open("tests/test_recursive.py", "w") as f:
    f.write(text)
