"""Utilities: seeds, device, logger."""
from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Seeds & device
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU+CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return cuda if available else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOGGER_CONFIGURED = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a stdout logger; configure root handler once."""
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(handler)
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name if name else "alignment_mlp")
