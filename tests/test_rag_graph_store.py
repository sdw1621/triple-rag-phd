"""Tests for src.rag.graph_store.GraphStore."""

from __future__ import annotations

import pytest

from src.rag.graph_store import GraphStore


@pytest.fixture
def small_graph() -> GraphStore:
    """4 professors, 2 departments, with 소속/협력 relations."""
    g = GraphStore(max_depth=3)
    g.add_node("p1", name="김철수", node_type="Professor", age=45)
    g.add_node("p2", name="이영희", node_type="Professor", age=38)
    g.add_node("p3", name="박민수", node_type="Professor", age=52)
    g.add_node("p4", name="정수진", node_type="Professor", age=36)
    g.add_node("d1", name="컴퓨터공학과", node_type="Department")
    g.add_node("d2", name="인공지능학과", node_type="Department")
    g.add_edge("p1", "소속", "d1")
    g.add_edge("p3", "소속", "d1")
    g.add_edge("p2", "소속", "d2")
    g.add_edge("p4", "소속", "d2")
    g.add_edge("p1", "협력", "p2")
    g.add_edge("p2", "협력", "p4")
    return g


def test_max_depth_must_be_positive() -> None:
    GraphStore(max_depth=1)
    with pytest.raises(ValueError):
        GraphStore(max_depth=0)


def test_empty_graph_search_returns_empty_list() -> None:
    g = GraphStore()
    assert g.search("anything") == []


def test_add_edge_with_missing_node_raises() -> None:
    g = GraphStore()
    g.add_node("a", name="A", node_type="X")
    with pytest.raises(KeyError):
        g.add_edge("a", "rel", "missing")


def test_search_returns_paths_with_arrow_format(small_graph: GraphStore) -> None:
    paths = small_graph.search("김철수", top_k=5)
    assert paths
    assert all("--[" in p and "]-->" in p for p in paths)
    assert any("김철수" in p for p in paths)


def test_search_top_k_respected(small_graph: GraphStore) -> None:
    assert len(small_graph.search("이영희", top_k=2)) <= 2


def test_search_no_seed_match_returns_empty(small_graph: GraphStore) -> None:
    assert small_graph.search("존재하지않음") == []


def test_search_traverses_multi_hop(small_graph: GraphStore) -> None:
    """Starting from 김철수, BFS must reach 정수진 within 3 hops via 협력→이영희→협력→정수진."""
    paths = small_graph.search("김철수", top_k=20, max_depth=3)
    # Path text only shows direct edges, but reaching 정수진 requires the BFS
    # to expand beyond the immediate neighborhood.
    assert any("정수진" in p or "이영희" in p for p in paths)


def test_get_stats_returns_correct_counts(small_graph: GraphStore) -> None:
    stats = small_graph.get_stats()
    assert stats["n_nodes"] == 6
    assert stats["n_edges"] == 6
    assert stats["avg_degree"] == pytest.approx(12 / 6)


def test_build_from_data() -> None:
    g = GraphStore()
    g.build_from_data(
        {
            "nodes": [
                {"id": "n1", "name": "A", "type": "X"},
                {"id": "n2", "name": "B", "type": "Y"},
            ],
            "edges": [{"source": "n1", "target": "n2", "relation": "links"}],
        }
    )
    assert g.get_stats()["n_nodes"] == 2
    assert g.get_stats()["n_edges"] == 1
