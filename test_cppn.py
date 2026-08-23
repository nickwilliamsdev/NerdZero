import neat
import torch
import os
import sys

from nerd_muzero.tensorneat_pt.cppn import build_cppn

config_path = os.path.abspath('neat_config.txt')
config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    config_path
)

p = neat.Population(config)
genome = list(p.population.values())[0]

cppn = build_cppn(genome, config, ['x1', 'y1', 'x2', 'y2'], ['w1', 'w2', 'w3', 'w4'])
print("CPPN type:", type(cppn))
print("CPPN:", cppn)
