"""Reproducibility controls for Python, NumPy, PyTorch, and Lightning."""

import random

import lightning
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed supported random generators and request deterministic torch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    lightning.seed_everything(seed, workers=True)
