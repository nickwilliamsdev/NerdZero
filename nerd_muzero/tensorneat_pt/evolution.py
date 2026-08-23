import neat
import os
import torch
from .cppn import build_cppn
from .substrate import query_cppn_for_weights

def evaluate_genomes(genomes, config, env):
    """
    NEAT-Python generation evaluation function.
    Iterates over all genomes, builds CPPNs, maps them to PyTorch substrates,
    and runs the environment to assign fitness.
    """
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        
        # 1. Build CPPN using PyTorch-NEAT
        # The leaf nodes mapped to the spatial coordinates of the substrate connection.
        leaf_names = ['x1', 'y1', 'x2', 'y2']
        node_names = ['weight']
        
        cppn = build_cppn(
            genome=genome, 
            config=config, 
            leaf_names=leaf_names, 
            node_names=node_names
        )
        
        # 2. Build Substrate weights using the CPPN
        # (This will be dynamically replaced when delta_net and recursive models are built)
        # dummy_coords = torch.rand(10, 4) 
        # generated_weights = query_cppn_for_weights(cppn, dummy_coords)
        
        # 3. Instantiate your MuZero models with `generated_weights`
        # ...
        
        # 4. Evaluate via the ARC environment and assign fitness
        # obs, info = env.reset()
        # while not done (play episode):
        #    ...
        
        # Placeholder minimal fitness
        genome.fitness += 1.0 

def run_neat_evolution(config_file: str, env, generations: int = 100):
    """
    Main loop for running neat-python with the PyTorch-NEAT adapter structure.
    """
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Inject the ARCWrapper environment context into the evaluator
    evaluator = lambda genomes, conf: evaluate_genomes(genomes, conf, env)

    winner = p.run(evaluator, generations)
    
    print("\nBest genome:\n{!s}".format(winner))
    
    return winner
