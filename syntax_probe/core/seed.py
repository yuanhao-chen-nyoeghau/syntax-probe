"""Reproducibility: seed everything that matters."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. Call at the start of every script entry."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Seeded all RNGs with seed=%d", seed)


__all__ = ["seed_everything"]
