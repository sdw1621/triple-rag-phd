"""
Abstract base class for Dynamic Weighting Algorithms (DWA).

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 4 (R-DWA), Chapter 5 (L-DWA)

Both R-DWA (rule-based) and L-DWA (PPO-learned) implement this interface so
the TripleHybridRAG pipeline can swap them transparently.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.intent.rule_based import QueryIntent

logger = logging.getLogger(__name__)

_SIMPLEX_TOL: float = 1e-5


@dataclass(frozen=True)
class DWAWeights:
    """Source weights on the simplex (α + β + γ = 1).

    Attributes:
        alpha: Vector retrieval weight in [0, 1].
        beta: Graph retrieval weight in [0, 1].
        gamma: Ontology retrieval weight in [0, 1].
    """

    alpha: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        for name, val in (("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)):
            if not 0.0 <= val <= 1.0 + _SIMPLEX_TOL:
                raise ValueError(f"{name}={val} out of [0, 1]")
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > _SIMPLEX_TOL:
            raise ValueError(f"Weights do not sum to 1.0 (got {total:.6f})")

    def as_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.alpha, self.beta, self.gamma)

    def __repr__(self) -> str:
        return f"DWAWeights(α={self.alpha:.3f}, β={self.beta:.3f}, γ={self.gamma:.3f})"


class BaseDWA(ABC):
    """Abstract Dynamic Weighting Algorithm.

    Implementations decide source weights given the user query and its analyzed
    intent. The pipeline calls `compute()` once per query.
    """

    @abstractmethod
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        """Compute (α, β, γ) on the simplex for a single query.

        Args:
            query: Original natural-language query.
            intent: Output of an intent analyzer (rule-based or BERT-based).

        Returns:
            DWAWeights normalized to sum to 1.
        """
