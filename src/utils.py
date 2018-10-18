import logging
from pathlib import Path

import torch


def save_checkpoint(checkpoint_dir, save_dict, file_name=None, best=False):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if file_name is not None:
        save_path = checkpoint_dir / file_name
    else:
        suffix = "latest.pkl" if not best else "best.pkl"
        save_path = checkpoint_dir / suffix
    torch.save(save_dict, str(save_path))
    logging.debug(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_dir, best=False):
    checkpoint_dir = Path(checkpoint_dir)
    suffix = "latest.pkl" if not best else "best.pkl"
    load_path = checkpoint_dir / suffix
    loaded = torch.load(load_path)
    logging.debug(f"Loaded checkpoint from {load_path.resolve().as_uri()}")
    return loaded
