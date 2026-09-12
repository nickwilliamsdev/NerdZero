from pathlib import Path
import torch

def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path

def load_checkpoint(path, map_location="cpu"):
    return torch.load(Path(path), map_location=map_location)
