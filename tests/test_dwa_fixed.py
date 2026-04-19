"""Tests for src.dwa.fixed.FixedWeightsDWA."""

from __future__ import annotations

import pytest

from src.dwa.base import BaseDWA, DWAWeights
from src.dwa.fixed import FixedWeightsDWA
from src.intent.rule_based import QueryIntent


def _intent() -> QueryIntent:
    return QueryIntent(
        query_type="simple",
        entities=[],
        relations=[],
        constraints=[],
        complexity_score=0.0,
        density=(0.1, 0.2, 0.3),
    )


def test_implements_base_dwa() -> None:
    dwa = FixedWeightsDWA(DWAWeights(0.5, 0.3, 0.2))
    assert isinstance(dwa, BaseDWA)


def test_compute_returns_constructor_weights() -> None:
    target = DWAWeights(0.5, 0.3, 0.2)
    dwa = FixedWeightsDWA(target)
    assert dwa.compute("anything", _intent()) == target


def test_compute_ignores_query_and_intent() -> None:
    target = DWAWeights(0.0, 0.0, 1.0)
    dwa = FixedWeightsDWA(target)
    a = dwa.compute("foo", _intent())
    b = dwa.compute("bar", QueryIntent(
        query_type="multi_hop", entities=["x"] * 10,
        relations=["r"] * 10, constraints=["c"] * 10,
        complexity_score=1.0, density=(1.0, 1.0, 1.0),
    ))
    assert a == b == target
