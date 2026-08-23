with open("nerd_muzero/training/outer_loop_neat.py", "r") as f:
    text = f.read()

text = text.replace(
    'encoder.beta_proj.weight.data = wbeta.view(input_dim, 1).t()',
    '# beta_proj maps from input_dim (100) -> 1\n            # We evaluated over input_dim * latent_dim grid pairs for the other weights\n            # For beta_proj we only need a size of [1, input_dim]. Let\'s slice wbeta or evaluate it differently.\n            encoder.beta_proj.weight.data = wbeta[:input_dim].view(input_dim, 1).t()'
)

with open("nerd_muzero/training/outer_loop_neat.py", "w") as f:
    f.write(text)
