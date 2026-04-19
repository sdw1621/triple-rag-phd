"""Tests for src.utils.seed.set_seed."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from src.utils.seed import THESIS_SEEDS, set_seed


def test_thesis_seeds_constant() -> None:
    assert THESIS_SEEDS == (42, 123, 999)


def test_invalid_seed_raises() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)
    with pytest.raises(ValueError):
        set_seed("42")  # type: ignore[arg-type]


def test_python_random_is_deterministic() -> None:
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_random_is_deterministic() -> None:
    set_seed(42)
    a = np.random.rand(3, 3)
    set_seed(42)
    b = np.random.rand(3, 3)
    assert np.array_equal(a, b)


def test_torch_random_is_deterministic() -> None:
    set_seed(42)
    a = torch.randn(3, 3)
    set_seed(42)
    b = torch.randn(3, 3)
    assert torch.equal(a, b)


def test_different_seeds_produce_different_streams() -> None:
    set_seed(42)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_pythonhashseed_env_var_set() -> None:
    set_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_cudnn_deterministic_flag_when_requested() -> None:
    set_seed(42, deterministic_torch=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
