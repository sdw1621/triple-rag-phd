"""
Deterministic seeding for reproducible thesis experiments.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 6 Section 1 (reproducibility), Table 6-1

All thesis numbers are reported as mean ± std over seeds {42, 123, 999}.
Call :func:`set_seed` at the top of every script entry point and inside
each PPO seed run.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

THESIS_SEEDS: tuple[int, ...] = (42, 123, 999)


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Sets:
        - ``PYTHONHASHSEED`` env var (affects dict iteration in subprocesses).
        - ``random.seed`` / ``numpy.random.seed`` / ``torch.manual_seed``.
        - ``torch.cuda.manual_seed_all`` (all GPUs).
        - When ``deterministic_torch`` is True:
          ``torch.backends.cudnn.deterministic = True`` and
          ``torch.backends.cudnn.benchmark = False``.

    Args:
        seed: Integer seed.
        deterministic_torch: If False, leaves cuDNN tuning enabled (faster
            but non-deterministic). Thesis runs use True.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("set_seed(%d, deterministic=%s)", seed, deterministic_torch)
