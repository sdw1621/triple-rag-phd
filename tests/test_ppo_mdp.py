"""Tests for src.ppo.mdp."""

from __future__ import annotations

import math

import pytest
import torch

from src.dwa.base import DWAWeights
from src.intent.rule_based import QueryIntent, RuleBasedIntent
from src.ppo.mdp import (
    ACTION_DIM,
    LATENCY_BUDGET_SEC,
    STATE_DIM,
    Action,
    State,
    build_state,
    compute_reward,
    extract_query_meta,
    extract_source_stats,
)


# ---------- State ----------

def test_state_constants() -> None:
    assert STATE_DIM == 18
    assert ACTION_DIM == 3


def test_state_shape_validation_density() -> None:
    with pytest.raises(ValueError, match="density"):
        State(
            density=(0.1, 0.2),  # type: ignore[arg-type]
            intent_logits=(0.0, 0.0, 0.0),
            source_stats=tuple([0.0] * 9),
            query_meta=(0.0, 0.0, 0.0),
        )


def test_state_shape_validation_source_stats() -> None:
    with pytest.raises(ValueError, match="source_stats"):
        State(
            density=(0.1, 0.2, 0.3),
            intent_logits=(0.0, 0.0, 0.0),
            source_stats=tuple([0.0] * 8),
            query_meta=(0.0, 0.0, 0.0),
        )


def test_state_to_tensor_returns_18_dim_float32() -> None:
    s = State(
        density=(0.1, 0.2, 0.3),
        intent_logits=(0.4, 0.5, 0.6),
        source_stats=tuple(0.1 * i for i in range(9)),
        query_meta=(0.7, 0.8, 1.0),
    )
    t = s.to_tensor()
    assert t.shape == (STATE_DIM,)
    assert t.dtype == torch.float32
    # Order check: density slice 0:3
    assert torch.allclose(t[0:3], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(t[3:6], torch.tensor([0.4, 0.5, 0.6]))
    assert torch.allclose(t[15:18], torch.tensor([0.7, 0.8, 1.0]))


def test_state_to_tensor_from_tensor_roundtrip() -> None:
    s = State(
        density=(0.1, 0.2, 0.3),
        intent_logits=(0.4, 0.5, 0.6),
        source_stats=tuple(0.1 * i for i in range(9)),
        query_meta=(0.7, 0.8, 0.0),
    )
    s2 = State.from_tensor(s.to_tensor())
    assert s2.density == pytest.approx(s.density)
    assert s2.intent_logits == pytest.approx(s.intent_logits)
    assert s2.source_stats == pytest.approx(s.source_stats)
    assert s2.query_meta == pytest.approx(s.query_meta)


def test_from_tensor_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="last-dim"):
        State.from_tensor(torch.zeros(10))


# ---------- Action ----------

def test_action_to_weights_validates_simplex() -> None:
    a = Action(0.5, 0.3, 0.2)
    w = a.to_weights()
    assert isinstance(w, DWAWeights)
    assert (w.alpha, w.beta, w.gamma) == (0.5, 0.3, 0.2)


def test_action_off_simplex_raises_via_weights() -> None:
    a = Action(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        a.to_weights()


def test_action_from_tensor_roundtrip() -> None:
    a = Action.from_tensor(torch.tensor([0.5, 0.3, 0.2]))
    assert a.to_tuple() == pytest.approx((0.5, 0.3, 0.2))


def test_action_from_tensor_wrong_dim() -> None:
    with pytest.raises(ValueError):
        Action.from_tensor(torch.zeros(2))


# ---------- compute_reward ----------

def test_reward_perfect_no_latency_penalty() -> None:
    # F1=1, EM=1, Faith=1, latency below budget → 0.5+0.3+0.2 = 1.0
    assert compute_reward(1.0, 1.0, 1.0, 2.0) == pytest.approx(1.0)


def test_reward_zero_metrics_with_latency_under_budget() -> None:
    assert compute_reward(0.0, 0.0, 0.0, LATENCY_BUDGET_SEC) == pytest.approx(0.0)


def test_reward_latency_penalty_kicks_in() -> None:
    # All metrics zero, latency 10 → -0.1 * (10 - 5) = -0.5
    assert compute_reward(0.0, 0.0, 0.0, 10.0) == pytest.approx(-0.5)


def test_reward_partial_metrics_with_latency() -> None:
    # F1=0.8, EM=1, Faith=0.9, latency=6 → 0.4 + 0.3 + 0.18 - 0.1 = 0.78
    assert compute_reward(0.8, 1.0, 0.9, 6.0) == pytest.approx(0.78)


# ---------- extract_source_stats ----------

def test_extract_source_stats_dimensions() -> None:
    stats = extract_source_stats([0.9, 0.8, 0.7], [1.0, 0.5], [0.3])
    assert len(stats) == 9


def test_extract_source_stats_empty_source() -> None:
    stats = extract_source_stats([], [0.5], [])
    # First 3 = vector zeros, next 3 = graph (0.5, 0.5, 1/3), last 3 = ontology zeros
    assert stats[:3] == (0.0, 0.0, 0.0)
    assert stats[3] == 0.5  # top
    assert stats[4] == 0.5  # mean
    assert stats[5] == pytest.approx(1 / 3)  # count_norm
    assert stats[6:] == (0.0, 0.0, 0.0)


def test_extract_source_stats_count_norm_capped_at_one() -> None:
    stats = extract_source_stats([0.1] * 10, [], [])
    assert stats[2] == 1.0  # count_norm capped


# ---------- extract_query_meta ----------

def test_query_meta_negation_detected() -> None:
    intent = RuleBasedIntent().analyze("이영희 교수가 아니라 박민수 교수")
    meta = extract_query_meta("이영희 교수가 아니라 박민수 교수", intent)
    assert meta[2] == 1.0  # has_negation


def test_query_meta_no_negation() -> None:
    intent = RuleBasedIntent().analyze("김철수 교수")
    meta = extract_query_meta("김철수 교수", intent)
    assert meta[2] == 0.0


def test_query_meta_log_length_norm_in_unit_interval() -> None:
    intent = RuleBasedIntent().analyze("test")
    short = extract_query_meta("a", intent)
    long_q = "x" * 1000
    long_meta = extract_query_meta(long_q, intent)
    assert 0.0 <= short[0] <= 1.0
    assert 0.0 <= long_meta[0] <= 1.0
    assert long_meta[0] >= short[0]


# ---------- build_state ----------

def test_build_state_returns_18_dim_state() -> None:
    intent = RuleBasedIntent().analyze("40세 이하 컴퓨터공학과 교수는?")
    state = build_state(
        intent=intent,
        intent_logits=(0.1, 0.7, 0.9),
        vector_scores=[0.85, 0.7, 0.6],
        graph_scores=[1.0, 1.0, 1.0],
        ontology_scores=[0.9],
        query="40세 이하 컴퓨터공학과 교수는?",
    )
    t = state.to_tensor()
    assert t.shape == (STATE_DIM,)
    # Density slice equals intent.density
    assert tuple(t[0:3].tolist()) == pytest.approx(intent.density)
    # Intent logits slice
    assert tuple(t[3:6].tolist()) == pytest.approx((0.1, 0.7, 0.9))
