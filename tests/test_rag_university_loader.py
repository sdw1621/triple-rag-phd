"""Tests for src.rag.university_loader (extended corpus from dataset_generator)."""

from __future__ import annotations

import pytest

from src.rag.graph_store import GraphStore
from src.rag.ontology_store import OntologyStore
from src.rag.university_loader import (
    DEFAULT_SEED,
    build_documents,
    build_graph,
    build_ontology,
    load_university_data,
    stats,
)


# ---------- load_university_data ----------

def test_load_returns_expected_keys() -> None:
    d = load_university_data()
    expected_keys = {"depts", "professors", "courses", "projects",
                     "dept_profs", "dept_courses", "course_profs"}
    assert expected_keys.issubset(set(d.keys()))


def test_seed_caching_returns_same_object() -> None:
    """load_university_data caches per-seed; same seed → same dict object."""
    d1 = load_university_data(seed=DEFAULT_SEED)
    d2 = load_university_data(seed=DEFAULT_SEED)
    assert d1 is d2


def test_extended_corpus_dimensions() -> None:
    """seed=42 must produce the published thesis dimensions."""
    s = stats()
    assert s["n_departments"] == 60
    assert s["n_professors"] == 577
    assert s["n_courses"] == 1505
    assert s["n_projects"] == 400
    # Collaborations: 1-3 per prof. Lower bound is 577.
    assert s["n_collaborations"] >= 577


# ---------- build_documents ----------

def test_build_documents_count() -> None:
    docs = build_documents()
    # 577 prof + 60 dept + 1505 course + 400 project = 2542
    assert len(docs) == 577 + 60 + 1505 + 400


def test_build_documents_contain_expected_keywords() -> None:
    docs = build_documents()
    assert any("교수는" in d and "소속이다" in d for d in docs)
    assert any("프로젝트에는" in d for d in docs)


# ---------- build_graph ----------

def test_build_graph_node_count_matches_corpus() -> None:
    g = build_graph()
    assert isinstance(g, GraphStore)
    s = g.get_stats()
    # 60 dept + 577 prof + 1505 course + 400 project = 2542 nodes
    assert s["n_nodes"] == 60 + 577 + 1505 + 400


def test_build_graph_edge_floor() -> None:
    """Edges: 소속 (577) + 개설 (≤1505) + 담당 (≥577×3) + 협력 (≥1) + 참여 (≥800).
    Just verify the lower bound is sensibly large."""
    g = build_graph()
    assert g.get_stats()["n_edges"] >= 3000


def test_build_graph_search_finds_first_professor() -> None:
    d = load_university_data()
    first_prof_name = d["professors"][0]["name"]
    g = build_graph()
    paths = g.search(first_prof_name, top_k=5)
    assert paths
    assert any(first_prof_name in p for p in paths)


# ---------- build_ontology ----------

def test_build_ontology_one_instance_per_professor() -> None:
    onto = build_ontology()
    assert isinstance(onto, OntologyStore)
    assert onto.n_instances == 577


def test_build_ontology_max_instances_cap() -> None:
    onto = build_ontology(max_instances=10)
    assert onto.n_instances == 10


def test_build_ontology_constraint_check_uses_real_age() -> None:
    d = load_university_data()
    onto = build_ontology()
    # Pick a professor with known age and exercise the constraint check.
    prof = d["professors"][0]
    over_age = prof["age"] + 5
    under_age = max(prof["age"] - 5, 1)
    assert onto.satisfies_constraint(prof["name"], f"{over_age}세 이하 교수") is True
    assert onto.satisfies_constraint(prof["name"], f"{under_age}세 이상 교수") is True
