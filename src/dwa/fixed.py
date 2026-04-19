"""
FixedWeightsDWA — used by the offline cache builder.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>

The cache build path needs to evaluate the pipeline at every (query × weight)
pair. R-DWA chooses weights from intent and L-DWA chooses from a learned
policy — neither lets us *force* a weight. FixedWeightsDWA does exactly that:
``compute()`` ignores its inputs and returns the constructor-supplied weights.
"""

from __future__ import annotations

from src.dwa.base import BaseDWA, DWAWeights
from src.intent.rule_based import QueryIntent


class FixedWeightsDWA(BaseDWA):
    """Always returns a pre-set :class:`DWAWeights`.

    Args:
        weights: The fixed weights to return on every :meth:`compute` call.

    Example:
        >>> dwa = FixedWeightsDWA(DWAWeights(0.5, 0.3, 0.2))
        >>> dwa.compute("any query", QueryIntent(  # doctest: +SKIP
        ...     query_type="simple", entities=[], relations=[], constraints=[],
        ...     complexity_score=0.0, density=(0.0, 0.0, 0.0)))
        DWAWeights(α=0.500, β=0.300, γ=0.200)
    """

    def __init__(self, weights: DWAWeights) -> None:
        self.weights = weights

    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        del query, intent  # ignored by design
        return self.weights
