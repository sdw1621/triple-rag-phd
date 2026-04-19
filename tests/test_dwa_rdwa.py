"""Tests for src.dwa.base.DWAWeights and src.dwa.rdwa.RuleBasedDWA."""

from __future__ import annotations

import pytest

from src.dwa.base import DWAWeights
from src.dwa.rdwa import BASE_WEIGHTS, RuleBasedDWA
from src.intent.rule_based import QueryIntent


# ---------- DWAWeights ----------

def test_weights_must_sum_to_one() -> None:
    DWAWeights(0.5, 0.3, 0.2)  # ok
    with pytest.raises(ValueError, match="sum to 1.0"):
        DWAWeights(0.5, 0.3, 0.3)


def test_weights_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="out of"):
        DWAWeights(-0.1, 0.6, 0.5)
    with pytest.raises(ValueError, match="out of"):
        DWAWeights(1.1, 0.0, -0.1)


def test_weights_as_dict_and_tuple() -> None:
    w = DWAWeights(0.6, 0.2, 0.2)
    assert w.as_dict() == {"alpha": 0.6, "beta": 0.2, "gamma": 0.2}
    assert w.as_tuple() == (0.6, 0.2, 0.2)


# ---------- RuleBasedDWA ----------

def _intent(query_type, s_e=0.0, s_r=0.0, s_c=0.0) -> QueryIntent:
    return QueryIntent(
        query_type=query_type,
        entities=[],
        relations=[],
        constraints=[],
        complexity_score=0.0,
        density=(s_e, s_r, s_c),
    )


def test_lambda_must_be_in_open_zero_one() -> None:
    RuleBasedDWA(lambda_=0.3)
    with pytest.raises(ValueError):
        RuleBasedDWA(lambda_=0.0)
    with pytest.raises(ValueError):
        RuleBasedDWA(lambda_=-0.1)
    with pytest.raises(ValueError):
        RuleBasedDWA(lambda_=1.5)


def test_simple_zero_density_returns_base_weights() -> None:
    """With s_r=s_c=0, output should equal base weights for the query type."""
    dwa = RuleBasedDWA(lambda_=0.3)
    for qtype, base in BASE_WEIGHTS.items():
        intent = _intent(qtype, s_r=0.0, s_c=0.0)
        w = dwa.compute("", intent)
        assert pytest.approx(w.alpha, abs=1e-9) == base[0]
        assert pytest.approx(w.beta, abs=1e-9) == base[1]
        assert pytest.approx(w.gamma, abs=1e-9) == base[2]


def test_high_relation_density_increases_beta() -> None:
    dwa = RuleBasedDWA(lambda_=0.3)
    base = dwa.compute("", _intent("simple", s_r=0.0, s_c=0.0))
    boosted = dwa.compute("", _intent("simple", s_r=1.0, s_c=0.0))
    assert boosted.beta > base.beta
    assert boosted.alpha < base.alpha  # alpha decreases


def test_high_constraint_density_increases_gamma() -> None:
    dwa = RuleBasedDWA(lambda_=0.3)
    base = dwa.compute("", _intent("simple", s_r=0.0, s_c=0.0))
    boosted = dwa.compute("", _intent("simple", s_r=0.0, s_c=1.0))
    assert boosted.gamma > base.gamma
    assert boosted.alpha < base.alpha


def test_output_always_on_simplex() -> None:
    dwa = RuleBasedDWA()
    for qtype in ("simple", "multi_hop", "conditional"):
        for s_r in (0.0, 0.5, 1.0):
            for s_c in (0.0, 0.5, 1.0):
                w = dwa.compute("", _intent(qtype, s_r=s_r, s_c=s_c))
                assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-9
                assert all(0.0 <= x <= 1.0 for x in w.as_tuple())


def test_explain_returns_string() -> None:
    dwa = RuleBasedDWA()
    text = dwa.explain(_intent("conditional", s_r=0.3, s_c=0.6))
    assert "Stage 1" in text
    assert "Stage 2" in text
    assert "Final" in text
