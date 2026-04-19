"""Tests for src.intent.rule_based.RuleBasedIntent."""

from __future__ import annotations

import pytest

from src.intent.rule_based import QueryIntent, RuleBasedIntent


@pytest.fixture
def analyzer() -> RuleBasedIntent:
    return RuleBasedIntent()


def test_simple_query_classified_as_simple(analyzer: RuleBasedIntent) -> None:
    intent = analyzer.analyze("김철수 교수는 어느 학과 소속인가?")
    # Has 1 entity (김철수 교수) and 1 relation (소속) → not multi_hop
    # No constraints → not conditional
    assert intent.query_type == "simple"
    assert "김철수 교수" in intent.entities
    assert "소속" in intent.relations
    assert intent.constraints == []


def test_multi_hop_query_classified(analyzer: RuleBasedIntent) -> None:
    intent = analyzer.analyze(
        "이영희 교수와 박민수 교수가 함께 담당하는 과목은 무엇이며 그 과목을 수강하는 학생은?"
    )
    assert intent.query_type == "multi_hop"
    assert len(intent.entities) >= 2 or len(intent.relations) >= 2


def test_conditional_query_classified(analyzer: RuleBasedIntent) -> None:
    intent = analyzer.analyze("40세 이하 컴퓨터공학과 교수 중 인공지능을 연구하는 사람은?")
    assert intent.query_type == "conditional"
    assert any("이하" in c for c in intent.constraints)


def test_density_signals_in_unit_interval(analyzer: RuleBasedIntent) -> None:
    intent = analyzer.analyze("AI, ML, NLP, CV, DL을 모두 다루는 교수는?")
    s_e, s_r, s_c = intent.density
    assert 0.0 <= s_e <= 1.0
    assert 0.0 <= s_r <= 1.0
    assert 0.0 <= s_c <= 1.0
    assert s_e == intent.s_e
    assert s_r == intent.s_r
    assert s_c == intent.s_c


def test_empty_query_returns_simple_with_zero_density(analyzer: RuleBasedIntent) -> None:
    intent = analyzer.analyze("")
    assert intent.query_type == "simple"
    assert intent.entities == []
    assert intent.relations == []
    assert intent.constraints == []
    assert intent.density == (0.0, 0.0, 0.0)


def test_entities_deduplicated(analyzer: RuleBasedIntent) -> None:
    intent = analyzer.analyze("김철수 교수와 김철수 교수의 연구 분야는?")
    # "김철수 교수" should appear only once
    assert intent.entities.count("김철수 교수") == 1


def test_query_intent_is_dataclass() -> None:
    intent = QueryIntent(
        query_type="simple",
        entities=["김철수 교수"],
        relations=["소속"],
        constraints=[],
        complexity_score=0.1,
        density=(0.2, 0.25, 0.0),
    )
    assert intent.s_e == 0.2
    assert intent.s_r == 0.25
    assert intent.s_c == 0.0
