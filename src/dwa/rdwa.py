"""
Rule-based Dynamic Weighting Algorithm (R-DWA).

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 4 (Eq. 4-1 to 4-5, Table 4-1)

Two-stage rule:
  Stage 1 — query type → base weights (Table 4-1).
  Stage 2 — density-based adjustment using s_r, s_c (Eq. 4-2 to 4-4).
  Stage 3 — simplex normalization (Eq. 4-5).
λ = 0.3 (selected via grid search in JKSCI 2025).

Ported from: hybrid-rag-comparsion/src/dwa.py.
"""

from __future__ import annotations

import logging
from typing import Mapping

from src.dwa.base import BaseDWA, DWAWeights
from src.intent.rule_based import QueryIntent, QueryType

logger = logging.getLogger(__name__)

# Thesis Table 4-1.
BASE_WEIGHTS: Mapping[QueryType, tuple[float, float, float]] = {
    "simple": (0.6, 0.2, 0.2),
    "multi_hop": (0.2, 0.6, 0.2),
    "conditional": (0.2, 0.2, 0.6),
}


class RuleBasedDWA(BaseDWA):
    """R-DWA — the JKSCI 2025 baseline.

    Args:
        lambda_: Adjustment strength λ in (0, 1]. Default 0.3 (paper-tuned).

    Example:
        >>> from src.intent.rule_based import RuleBasedIntent
        >>> intent = RuleBasedIntent().analyze("40세 이하 컴공과 교수는?")
        >>> w = RuleBasedDWA().compute("40세 이하 컴공과 교수는?", intent)
        >>> abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6
        True
    """

    def __init__(self, lambda_: float = 0.3) -> None:
        if not 0.0 < lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in (0, 1], got {lambda_}")
        self.lambda_ = lambda_
        logger.debug("Initialized RuleBasedDWA(lambda_=%s)", lambda_)

    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        """Compute weights from query type and density signals.

        Args:
            query: Unused by R-DWA (kept for interface symmetry with L-DWA).
            intent: Provides query_type and (s_r, s_c).

        Returns:
            Simplex-normalized DWAWeights.
        """
        del query  # R-DWA does not use the raw query string.
        a0, b0, g0 = BASE_WEIGHTS[intent.query_type]
        lam = self.lambda_
        s_r, s_c = intent.s_r, intent.s_c

        # Eq. 4-2 to 4-4 — density-based adjustment.
        a = a0 * (1.0 - lam * (s_r + s_c) / 2.0)
        b = b0 + lam * s_r * (1.0 - b0)
        g = g0 + lam * s_c * (1.0 - g0)

        # Eq. 4-5 — normalize to simplex.
        total = a + b + g
        return DWAWeights(alpha=a / total, beta=b / total, gamma=g / total)

    def explain(self, intent: QueryIntent) -> str:
        """Human-readable trace of the computation (for debugging/papers)."""
        a0, b0, g0 = BASE_WEIGHTS[intent.query_type]
        w = self.compute("", intent)
        return (
            f"[Stage 1] type={intent.query_type} → α={a0:.2f}, β={b0:.2f}, γ={g0:.2f}\n"
            f"[Stage 2] s_r={intent.s_r:.2f} s_c={intent.s_c:.2f} λ={self.lambda_}\n"
            f"[Final  ] α={w.alpha:.3f} β={w.beta:.3f} γ={w.gamma:.3f}"
        )
