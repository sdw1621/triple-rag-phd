"""Tests for src.rag.ontology_store.OntologyStore."""

from __future__ import annotations

import pytest

from src.rag.ontology_store import (
    DEFAULT_INSTANCES,
    OntologyStore,
    PersonInstance,
)


@pytest.fixture
def store() -> OntologyStore:
    return OntologyStore(try_owlready=False)


def test_default_instance_count(store: OntologyStore) -> None:
    assert store.n_instances == len(DEFAULT_INSTANCES) == 4


def test_search_empty_query_returns_empty(store: OntologyStore) -> None:
    assert store.search("") == []


def test_search_by_name(store: OntologyStore) -> None:
    results = store.search("김철수", top_k=3)
    assert results
    assert any("김철수" in r and "컴퓨터공학과" in r for r in results)


def test_search_by_course(store: OntologyStore) -> None:
    results = store.search("자연어처리", top_k=3)
    # 박민수 담당
    assert any("박민수" in r for r in results)


def test_search_unknown_returns_default_briefs(store: OntologyStore) -> None:
    results = store.search("xyz", top_k=2)
    assert len(results) == 2
    # Each brief should look like "name: dept, age세"
    assert all(":" in r and "세" in r for r in results)


def test_constraint_age_iha_satisfied(store: OntologyStore) -> None:
    # 정수진은 36세 → "40세 이하" 만족
    assert store.satisfies_constraint("정수진", "40세 이하 교수") is True


def test_constraint_age_iha_violated(store: OntologyStore) -> None:
    # 박민수는 52세 → "40세 이하" 미만족
    assert store.satisfies_constraint("박민수", "40세 이하 교수") is False


def test_constraint_age_isang_satisfied(store: OntologyStore) -> None:
    # 김철수는 45세 → "40세 이상" 만족
    assert store.satisfies_constraint("김철수", "40세 이상 교수") is True


def test_constraint_no_constraint_phrase_returns_true(store: OntologyStore) -> None:
    assert store.satisfies_constraint("김철수", "어느 학과 소속인가?") is True


def test_constraint_unknown_entity_returns_true(store: OntologyStore) -> None:
    # Unknown entity → can't disprove → True (preserves prior repo semantics).
    assert store.satisfies_constraint("없는사람", "30세 이하") is True


def test_custom_instances() -> None:
    custom = (
        PersonInstance("홍길동", "FullProfessor", 30, "전산학과", ["알고리즘"]),
    )
    store = OntologyStore(instances=custom, try_owlready=False)
    assert store.n_instances == 1
    results = store.search("홍길동", top_k=1)
    assert "홍길동" in results[0]
