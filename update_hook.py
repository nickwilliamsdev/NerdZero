with open("nerd_muzero/training/outer_loop_neat.py", "r") as f:
    text = f.read()

text = text.replace(
    'wq, wk, wv, wbeta = cppn_net(x1, y1, x2, y2)',
    'out = cppn_net(x1, y1, x2, y2)\n            wq, wk, wv, wbeta = out'
)

with open("nerd_muzero/training/outer_loop_neat.py", "w") as f:
    f.write(text)
