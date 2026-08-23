import sys
import os
import torch
import neat

# Dynamically add the local PyTorch-NEAT clone to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_neat_path = os.path.abspath(os.path.join(current_dir, "../../PyTorch-NEAT"))
if pytorch_neat_path not in sys.path:
    sys.path.append(pytorch_neat_path)

from pytorch_neat.cppn import create_cppn

def build_cppn(genome: neat.DefaultGenome, config: neat.Config, leaf_names: list[str], node_names: list[str]):
    """
    Builds a PyTorch CPPN from a given neat-python genome using the PyTorch-NEAT library.
    
    Args:
        genome: The NEAT genome instance
        config: The NEAT configuration instance
        leaf_names: The names of the input coordinates (e.g., ['x1', 'y1', 'x2', 'y2'])
        node_names: The names of the output weights (e.g., ['weight', 'bias'])
        
    Returns:
        A PyTorch-NEAT CPPN network.
    """
    cppn_net = create_cppn(
        genome=genome,
        config=config,
        leaf_names=leaf_names,
        node_names=node_names
    )
    return cppn_net
