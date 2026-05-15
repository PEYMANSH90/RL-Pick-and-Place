"""Random seed utilities for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Call once at the start of a script before any stochastic operations.

    Parameters
    ----------
    seed:
        Integer seed value.  Use the same seed to reproduce a run exactly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
