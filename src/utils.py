import logging
from pathlib import Path

import torch


def save_checkpoint(checkpoint_dir, save_dict, model_name=None, file_name=None, best=False):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        logging.log(logging.INFO, f"Checkpoint path {checkpoint_dir} does not exist. Creating it.")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if file_name is not None:
        save_path = checkpoint_dir / file_name
    elif model_name is not None:
        suffix = "latest.pkl" if not best else "best.pkl"
        save_path = checkpoint_dir / f"{model_name}.{suffix}"
    else:
        save_path = checkpoint_dir / "model_chk.latest.pkl"
    torch.save(save_dict, str(save_path))
    logging.log(logging.INFO, f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_dir, model_name=None, checkpoint_file=None, best=False):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_file is not None:
        load_path = checkpoint_dir / checkpoint_file
    elif model_name is not None:
        suffix = "latest.pkl" if not best else "best.pkl"
        load_path = checkpoint_dir / f"{model_name}.{suffix}"
    else:
        raise ValueError("load checkpoint requires model name or file name")
    loaded = torch.load(load_path)
    logging.info(f"Loaded checkpoint from {load_path.resolve().as_uri()}")
    return loaded
