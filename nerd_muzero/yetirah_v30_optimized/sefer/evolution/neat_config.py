from __future__ import annotations

import math
import random
import os
import pickle
import tempfile
import textwrap
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def neat_config_text(num_inputs: int, pop_size: int, seed: int = 0) -> str:
    """Current NEAT-Python compatible feed-forward CPPN configuration."""
    return textwrap.dedent(f"""
    [NEAT]
    fitness_criterion = max
    fitness_threshold = 1e9
    pop_size = {pop_size}
    reset_on_extinction = False
    no_fitness_termination = True
    seed = {seed}

    [DefaultGenome]
    activation_default = tanh
    activation_mutate_rate = 0.15
    activation_options = tanh sin gauss relu sigmoid identity
    aggregation_default = sum
    aggregation_mutate_rate = 0.0
    aggregation_options = sum
    bias_init_mean = 0.0
    bias_init_stdev = 1.0
    bias_init_type = gaussian
    bias_max_value = 10.0
    bias_min_value = -10.0
    bias_mutate_power = 0.5
    bias_mutate_rate = 0.7
    bias_replace_rate = 0.1
    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient = 0.2
    conn_add_prob = 0.35
    conn_delete_prob = 0.10
    enabled_default = True
    enabled_mutate_rate = 0.01
    enabled_rate_to_true_add = 0.0
    enabled_rate_to_false_add = 0.0
    feed_forward = True
    initial_connection = full_direct
    node_add_prob = 0.20
    node_delete_prob = 0.05
    num_hidden = 22
    num_inputs = {num_inputs}
    num_outputs = 1
    response_init_mean = 1.0
    response_init_stdev = 0.0
    response_init_type = gaussian
    response_max_value = 10.0
    response_min_value = -10.0
    response_mutate_power = 0.0
    response_mutate_rate = 0.0
    response_replace_rate = 0.0
    weight_init_mean = 0.0
    weight_init_stdev = 1.0
    weight_init_type = gaussian
    weight_max_value = 10.0
    weight_min_value = -10.0
    weight_mutate_power = 0.5
    weight_mutate_rate = 0.8
    weight_replace_rate = 0.1
    single_structural_mutation = false
    structural_mutation_surer = default

    [DefaultSpeciesSet]
    compatibility_threshold = 4.5

    [DefaultStagnation]
    species_fitness_func = max
    max_stagnation = 15
    species_elitism = 2

    [DefaultReproduction]
    elitism = 1
    survival_threshold = 0.25
    min_species_size = 1
    """).strip() + "\n"
