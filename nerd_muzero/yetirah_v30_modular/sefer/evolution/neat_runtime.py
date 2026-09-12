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

NEAT_INPUT_NAMES = (
    [f"dst_{i}" for i in range(5)]
    + [f"src_{i}" for i in range(5)]
    + [f"diff_{i}" for i in range(5)]
    + [f"prod_{i}" for i in range(5)]
    + ["dist", "dst_idx", "src_idx", "idx_diff", "idx_sum", "is_diag",
       "dst_phase_sin", "dst_phase_cos", "src_phase_sin", "src_phase_cos",
       "rel_phase_sin", "rel_phase_cos"]
    + [f"op_{i}" for i in range(8)]
)
NEAT_OUTPUT_NAMES = ["edge_logit"]


def _add_local_pytorch_neat_repo() -> Optional[Path]:
    """Add a sibling PyTorch-NEAT checkout to sys.path when present.

    Expected user layout::

        <workspace>/
            nerd_muzero/
                sefer/
                    yetirah_v0_neat_cppn_v21.py
            PyTorch-NEAT/
                pytorch_neat/

    From this file that repository is ../../PyTorch-NEAT.  A few additional
    candidates are checked so the script also works when launched/copied from
    another working directory.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        *(parent / "PyTorch-NEAT" for parent in [script_dir, *script_dir.parents]),
        Path.cwd() / "PyTorch-NEAT",
        Path.cwd().parent / "PyTorch-NEAT",
    ]

    for repo in candidates:
        repo = repo.resolve()
        if (repo / "pytorch_neat" / "__init__.py").is_file():
            repo_str = str(repo)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            return repo
    return None

def require_pytorch_neat():
    """Load NEAT-Python plus a local Uber PyTorch-NEAT checkout.

    PyTorch-NEAT does not need to be installed when its repository is a sibling
    of nerd_muzero; we add that checkout directly to ``sys.path``.  NEAT-Python
    itself is still a separate dependency because PyTorch-NEAT imports ``neat``.
    """
    local_repo = _add_local_pytorch_neat_repo()

    try:
        import neat  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "v21 found/uses the local PyTorch-NEAT checkout, but NEAT-Python is "
            "a separate dependency. Install it with:\n"
            "  pip install neat-python\n"
            "If this old PyTorch-NEAT checkout requires the historical API, try:\n"
            "  pip install neat-python==0.92"
        ) from exc

    try:
        from pytorch_neat.cppn import create_cppn  # type: ignore
    except Exception as exc:
        expected = (Path(__file__).resolve().parent.parent.parent / "PyTorch-NEAT").resolve()
        raise RuntimeError(
            "Could not import pytorch_neat. v21 expects a local checkout at:\n"
            f"  {expected}\n"
            "with a pytorch_neat/ package inside it. "
            f"Detected local repo: {local_repo!s}"
        ) from exc

    print(f"PyTorch-NEAT source: {local_repo or 'Python environment'}")
    print(f"NEAT-Python source: {getattr(neat, '__file__', '<unknown>')}")
    return neat, create_cppn

class EvolvedTorchCPPN:
    """Materialized PyTorch-NEAT CPPN winner used as a HyperNEAT edge law.

    The genome is evolved by NEAT-Python. PyTorch-NEAT converts that genome to
    a graph of torch-callable CPPN nodes. It is deliberately frozen during Adam
    training: topology and CPPN weights come from evolution, while the existing
    feature-affine law and controller remain gradient-trained.
    """
    def __init__(self, genome, config, create_cppn_fn):
        self.genome = genome
        self.config = config
        [self.edge_node] = create_cppn_fn(
            genome, config, list(NEAT_INPUT_NAMES), list(NEAT_OUTPUT_NAMES)
        )

    def edge_logits(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.edge_node(**features)
        if not torch.is_tensor(out):
            out = torch.as_tensor(out, dtype=next(iter(features.values())).dtype,
                                  device=next(iter(features.values())).device)
        return out

class DifferentiableGenomeCPPN(nn.Module):
    """Differentiable mirror of a feed-forward NEAT genome.

    Uber PyTorch-NEAT materializes connection weights as Python floats, which is
    excellent for inference but means an optimizer cannot update the original
    genome.  This module mirrors exactly the enabled feed-forward topology with
    torch Parameters, supports the same CPPN activations used by this script,
    and can write trained weights/biases/responses back into the NEAT genome.
    """
    def __init__(self, genome, config):
        super().__init__()
        self.genome = genome
        self.config = config
        gc = config.genome_config
        self.input_keys = list(gc.input_keys)
        self.output_keys = list(gc.output_keys)
        self.input_name_by_key = {k: n for k, n in zip(self.input_keys, NEAT_INPUT_NAMES)}

        self.incoming = {}
        for key, cg in genome.connections.items():
            if not cg.enabled:
                continue
            i, o = key
            self.incoming.setdefault(o, []).append((i, key))

        needed = set(self.output_keys)
        frontier = list(self.output_keys)
        while frontier:
            o = frontier.pop()
            for i, _ in self.incoming.get(o, []):
                if i not in self.input_keys and i not in needed:
                    needed.add(i)
                    frontier.append(i)
        self.node_keys = [k for k in needed if k not in self.input_keys]

        # Feed-forward topological order over only nodes required by outputs.
        order, done = [], set(self.input_keys)
        remaining = set(self.node_keys)
        while remaining:
            progressed = False
            for o in list(remaining):
                ins = [i for i, _ in self.incoming.get(o, [])]
                if all(i in done for i in ins):
                    order.append(o); done.add(o); remaining.remove(o); progressed = True
            if not progressed:
                raise RuntimeError('Genome is not feed-forward or contains an unresolved dependency')
        self.order = order

        self.conn_params = nn.ParameterDict()
        self.conn_param_to_gene = {}
        for o in self.order:
            for i, key in self.incoming.get(o, []):
                if i not in done:
                    continue
                name = self._conn_name(key)
                self.conn_params[name] = nn.Parameter(torch.tensor(float(genome.connections[key].weight)))
                self.conn_param_to_gene[name] = key

        self.bias_params = nn.ParameterDict()
        self.response_params = nn.ParameterDict()
        for k in self.order:
            gene = genome.nodes[k]
            nk = self._node_name(k)
            self.bias_params[nk] = nn.Parameter(torch.tensor(float(gene.bias)))
            self.response_params[nk] = nn.Parameter(torch.tensor(float(gene.response)))

    @staticmethod
    def _node_name(k):
        return ('nneg_' + str(-k)) if k < 0 else ('npos_' + str(k))

    @staticmethod
    def _conn_name(key):
        i, o = key
        def enc(x): return ('neg' + str(-x)) if x < 0 else ('pos' + str(x))
        return f'c_{enc(i)}__{enc(o)}'

    @staticmethod
    def _activate(name, x):
        if name == 'sigmoid': return torch.sigmoid(5.0 * x)
        if name == 'tanh': return torch.tanh(2.5 * x)
        if name == 'abs': return torch.abs(x)
        if name == 'gauss': return torch.exp(-5.0 * x.pow(2))
        if name == 'identity': return x
        if name == 'sin': return torch.sin(x)
        if name == 'relu': return F.relu(x)
        raise ValueError(f'Unsupported CPPN activation: {name}')

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        vals = {k: features[self.input_name_by_key[k]] for k in self.input_keys}
        ref = next(iter(features.values()))
        for o in self.order:
            gene = self.genome.nodes[o]
            terms = []
            for i, key in self.incoming.get(o, []):
                if i not in vals:
                    continue
                w = self.conn_params[self._conn_name(key)]
                terms.append(w * vals[i])
            if terms:
                pre = torch.stack(terms, dim=0).sum(dim=0)
            else:
                pre = torch.zeros_like(ref)
            nk = self._node_name(o)
            z = self.response_params[nk] * pre + self.bias_params[nk]
            vals[o] = self._activate(gene.activation, z)
        return vals[self.output_keys[0]]

    @torch.no_grad()
    def write_back_(self):
        """Lamarckian inheritance: copy trained torch parameters into genes."""
        gc = self.config.genome_config
        for name, key in self.conn_param_to_gene.items():
            v = float(self.conn_params[name].detach().cpu().item())
            v = max(float(gc.weight_min_value), min(float(gc.weight_max_value), v))
            self.genome.connections[key].weight = v
        for k in self.order:
            nk = self._node_name(k)
            b = float(self.bias_params[nk].detach().cpu().item())
            r = float(self.response_params[nk].detach().cpu().item())
            b = max(float(gc.bias_min_value), min(float(gc.bias_max_value), b))
            r = max(float(gc.response_min_value), min(float(gc.response_max_value), r))
            self.genome.nodes[k].bias = b
            self.genome.nodes[k].response = r
