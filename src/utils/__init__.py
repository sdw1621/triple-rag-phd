"""Utility modules: deterministic seeding, offline reward cache."""

from src.utils.offline_cache import (
    DEFAULT_GRID,
    DEFAULT_STEP,
    OfflineCache,
    RewardComponents,
    discretize,
    discretize_weights,
    enumerate_simplex,
    simplex_size,
)

__all__ = [
    "set_seed",
    "THESIS_SEEDS",
    "OfflineCache",
    "RewardComponents",
    "discretize",
    "discretize_weights",
    "enumerate_simplex",
    "simplex_size",
    "DEFAULT_GRID",
    "DEFAULT_STEP",
]


def __getattr__(name):
    if name in ("set_seed", "THESIS_SEEDS"):
        from src.utils.seed import THESIS_SEEDS, set_seed  # noqa: F401
        return THESIS_SEEDS if name == "THESIS_SEEDS" else set_seed
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")
