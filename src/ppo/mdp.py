"""
MDP formulation for Triple-Hybrid RAG weight decision.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 5 Section 1 (Eq. 5-1 to 5-7)

State (18-dim, Eq. 5-1):
    [density (3) | intent_logits (3) | source_stats (9) | query_meta (3)]

    density       — (s_e, s_r, s_c) from RuleBasedIntent (entity/relation/
                    constraint density).
    intent_logits — BERT multi-label pre-sigmoid scores for
                    (simple, multi_hop, conditional).
    source_stats  — for each of (vector, graph, ontology):
                    (top_score, mean_score, hit_count_norm), 3 × 3 = 9.
    query_meta    — (log_length_norm, n_entities_norm, has_negation_flag).

Action (Eq. 5-3):
    Continuous (α, β, γ) on the 3-simplex (α + β + γ = 1) — sampled from a
    Dirichlet distribution parameterized by the Actor head.

Reward (Eq. 5-7):
    R = 0.5 · F1 + 0.3 · EM + 0.2 · Faithfulness − 0.1 · max(0, latency − 5.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from src.dwa.base import DWAWeights
from src.intent.rule_based import QueryIntent

STATE_DIM: int = 18
DENSITY_DIM: int = 3
INTENT_DIM: int = 3
SOURCE_STATS_DIM: int = 9  # 3 sources × 3 stats
QUERY_META_DIM: int = 3
ACTION_DIM: int = 3

LATENCY_BUDGET_SEC: float = 5.0  # thesis Eq. 5-7


@dataclass(frozen=True)
class State:
    """18-dim PPO state vector.

    All scalars should be in roughly [-3, 3] for stable training; consumers
    are responsible for normalizing source scores into [0, 1].
    """

    density: tuple[float, float, float]
    intent_logits: tuple[float, float, float]
    source_stats: tuple[float, ...]  # length 9
    query_meta: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.density) != DENSITY_DIM:
            raise ValueError(f"density must be {DENSITY_DIM}-tuple")
        if len(self.intent_logits) != INTENT_DIM:
            raise ValueError(f"intent_logits must be {INTENT_DIM}-tuple")
        if len(self.source_stats) != SOURCE_STATS_DIM:
            raise ValueError(
                f"source_stats must be length {SOURCE_STATS_DIM}, "
                f"got {len(self.source_stats)}"
            )
        if len(self.query_meta) != QUERY_META_DIM:
            raise ValueError(f"query_meta must be {QUERY_META_DIM}-tuple")

    def to_tensor(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """Concatenate all components into an 18-dim float32 tensor."""
        flat = (
            list(self.density)
            + list(self.intent_logits)
            + list(self.source_stats)
            + list(self.query_meta)
        )
        return torch.tensor(flat, dtype=torch.float32, device=device)

    def to_list(self) -> list[float]:
        return self.to_tensor().tolist()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "State":
        """Inverse of :meth:`to_tensor` for one state vector."""
        if tensor.shape[-1] != STATE_DIM:
            raise ValueError(f"tensor last-dim must be {STATE_DIM}, got {tensor.shape}")
        flat = tensor.detach().cpu().tolist()
        d = tuple(flat[0:3])  # type: ignore[arg-type]
        i = tuple(flat[3:6])  # type: ignore[arg-type]
        s = tuple(flat[6:15])
        m = tuple(flat[15:18])  # type: ignore[arg-type]
        return cls(density=d, intent_logits=i, source_stats=s, query_meta=m)


@dataclass(frozen=True)
class Action:
    """Continuous action on the 3-simplex (α + β + γ = 1)."""

    alpha: float
    beta: float
    gamma: float

    def to_weights(self) -> DWAWeights:
        """Convert to a :class:`DWAWeights` (raises if off-simplex)."""
        return DWAWeights(self.alpha, self.beta, self.gamma)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.alpha, self.beta, self.gamma)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "Action":
        if tensor.shape[-1] != ACTION_DIM:
            raise ValueError(f"action tensor must have last dim {ACTION_DIM}")
        a, b, g = tensor.detach().cpu().tolist()
        return cls(alpha=a, beta=b, gamma=g)


# ---------- reward ----------

def compute_reward(
    f1: float,
    em: float,
    faithfulness: float,
    latency: float,
    latency_budget: float = LATENCY_BUDGET_SEC,
) -> float:
    """Thesis Eq. 5-7.

    Args:
        f1: Token F1 in [0, 1].
        em: Exact match in {0, 1}.
        faithfulness: Faithfulness in [0, 1].
        latency: End-to-end seconds.
        latency_budget: Free latency band before the penalty kicks in.
    """
    return (
        0.5 * f1
        + 0.3 * em
        + 0.2 * faithfulness
        - 0.1 * max(0.0, latency - latency_budget)
    )


# ---------- state extractors ----------

def _normalize_source_scores(
    scores: Sequence[float],
) -> tuple[float, float, float]:
    """Return (top, mean, count_norm) for one source's score list.

    ``count_norm`` is ``min(len(scores) / 3.0, 1.0)`` (3 = top_k cap).
    """
    if not scores:
        return (0.0, 0.0, 0.0)
    top = max(scores)
    mean = sum(scores) / len(scores)
    count_norm = min(len(scores) / 3.0, 1.0)
    return (float(top), float(mean), count_norm)


def extract_source_stats(
    vector_scores: Sequence[float],
    graph_scores: Sequence[float],
    ontology_scores: Sequence[float],
) -> tuple[float, ...]:
    """Build the 9-dim source_stats slice of the state vector."""
    v = _normalize_source_scores(vector_scores)
    g = _normalize_source_scores(graph_scores)
    o = _normalize_source_scores(ontology_scores)
    return v + g + o


_NEGATION_TOKENS: tuple[str, ...] = ("아니", "않", "없", "제외", "non", "not", "no ")


def extract_query_meta(
    query: str,
    intent: QueryIntent,
    max_log_length: float = 6.0,  # log(403) ≈ 6 — long Korean query
    max_entities: int = 5,
) -> tuple[float, float, float]:
    """Build the 3-dim query_meta slice.

    Returns ``(log_length_norm, n_entities_norm, has_negation_flag)``.
    """
    log_length = math.log1p(len(query))
    log_length_norm = min(log_length / max_log_length, 1.0)
    n_entities_norm = min(len(intent.entities) / max_entities, 1.0)
    has_negation = float(any(tok in query for tok in _NEGATION_TOKENS))
    return (log_length_norm, n_entities_norm, has_negation)


def build_state(
    intent: QueryIntent,
    intent_logits: tuple[float, float, float],
    vector_scores: Sequence[float],
    graph_scores: Sequence[float],
    ontology_scores: Sequence[float],
    query: str,
) -> State:
    """Assemble a :class:`State` from RAG pipeline outputs.

    Args:
        intent: From :class:`RuleBasedIntent` (provides density + entities).
        intent_logits: From :class:`BertIntentClassifier.predict_logits`.
        vector_scores: Top-k cosine similarities from VectorStore.
        graph_scores: Per-path relevance scores from GraphStore. (For BFS
            paths without intrinsic scoring, supply 1.0 per path.)
        ontology_scores: Per-fact match scores. (Supply 1.0 per fact when
            the fallback rule path is used.)
        query: Original natural-language query.
    """
    return State(
        density=intent.density,
        intent_logits=intent_logits,
        source_stats=extract_source_stats(vector_scores, graph_scores, ontology_scores),
        query_meta=extract_query_meta(query, intent),
    )
