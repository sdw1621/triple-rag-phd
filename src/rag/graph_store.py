"""
GraphStore — in-memory directed graph with BFS k-hop retrieval.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 3 Section 2.2

Backed by NetworkX MultiDiGraph for richer relation types in future, but the
hot retrieval path uses a hand-rolled adjacency list and BFS for speed (the
prior repo benchmarked NetworkX BFS as the bottleneck on the 5K QA set).

Ported from: hybrid-rag-comparsion/src/knowledge_graph.py — Neo4j fallback
removed (the PhD experiments stay in-memory; Neo4j path was unused in the
final evaluation).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable

import networkx as nx

logger = logging.getLogger(__name__)


class GraphStore:
    """In-memory knowledge graph with BFS k-hop retrieval.

    Args:
        max_depth: Maximum BFS depth. Default 3 (thesis Sec 3.2.2).

    Example:
        >>> g = GraphStore()
        >>> g.add_node("p1", name="김철수", node_type="Professor")
        >>> g.add_node("d1", name="컴퓨터공학과", node_type="Department")
        >>> g.add_edge("p1", "소속", "d1")
        >>> paths = g.search("김철수", top_k=3)
        >>> any("김철수" in p for p in paths)
        True
    """

    def __init__(self, max_depth: int = 3) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.max_depth = max_depth
        # NetworkX graph for stats / external use; adj for BFS hot path.
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._adj: dict[str, list[tuple[str, str]]] = {}

    # ---------- mutation ----------

    def add_node(self, node_id: str, name: str, node_type: str, **props: object) -> None:
        """Add or replace a node."""
        self._g.add_node(node_id, name=name, type=node_type, **props)
        self._adj.setdefault(node_id, [])

    def add_edge(self, src: str, relation: str, dst: str) -> None:
        """Add a directed edge (and a synthetic inverse for BFS reachability)."""
        if src not in self._g:
            raise KeyError(f"Source node not found: {src}")
        if dst not in self._g:
            raise KeyError(f"Target node not found: {dst}")
        self._g.add_edge(src, dst, key=relation, relation=relation)
        self._adj.setdefault(src, []).append((relation, dst))
        self._adj.setdefault(dst, []).append((f"inv_{relation}", src))

    def build_from_data(self, data: dict) -> None:
        """Bulk-load from a dict of {nodes, edges}.

        Args:
            data: ``{"nodes": [{"id", "name", "type", **attrs}, ...],
                     "edges": [{"source", "target", "relation"}, ...]}``
        """
        for node in data.get("nodes", []):
            attrs = {k: v for k, v in node.items() if k not in ("id", "name", "type")}
            self.add_node(node["id"], node["name"], node["type"], **attrs)
        for edge in data.get("edges", []):
            self.add_edge(edge["source"], edge["relation"], edge["target"])

    # ---------- retrieval ----------

    def search(
        self,
        query: str,
        top_k: int = 3,
        max_depth: int | None = None,
    ) -> list[str]:
        """BFS retrieval starting from nodes whose name appears in the query.

        Args:
            query: Natural-language query (substring match against node names).
            top_k: Maximum number of path strings to return.
            max_depth: Override instance max_depth for this call.

        Returns:
            List of human-readable path strings ``"A --[rel]--> B"``.
            Empty graph or no match returns an empty list.
        """
        depth_limit = max_depth if max_depth is not None else self.max_depth

        seeds = self._find_seeds(query)
        if not seeds:
            return []

        results: list[str] = []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)

        # Cap exploration at top_k * 10 expansions to avoid runaway BFS.
        budget = max(top_k * 10, top_k)

        while queue and len(results) < budget:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > depth_limit:
                continue
            visited.add(node_id)
            node_name = self._g.nodes[node_id].get("name", "?")
            for rel, neighbor in self._adj.get(node_id, []):
                if rel.startswith("inv_"):
                    continue
                neighbor_name = self._g.nodes.get(neighbor, {}).get("name", "?")
                results.append(f"{node_name} --[{rel}]--> {neighbor_name}")
            if depth < depth_limit:
                for _, neighbor in self._adj.get(node_id, []):
                    queue.append((neighbor, depth + 1))

        return results[:top_k]

    def _find_seeds(self, query: str) -> list[str]:
        if not query:
            return []
        return [
            nid
            for nid, attrs in self._g.nodes(data=True)
            if attrs.get("name") and (attrs["name"] in query or query in attrs["name"])
        ]

    # ---------- introspection ----------

    def get_stats(self) -> dict[str, float]:
        """Return basic graph statistics."""
        n_nodes = self._g.number_of_nodes()
        n_edges = self._g.number_of_edges()
        avg_degree = (2 * n_edges) / n_nodes if n_nodes else 0.0
        return {"n_nodes": n_nodes, "n_edges": n_edges, "avg_degree": avg_degree}

    @property
    def nodes(self) -> Iterable[str]:
        return self._g.nodes()

    @property
    def edges(self) -> Iterable[tuple[str, str, str]]:
        return ((u, v, k) for u, v, k in self._g.edges(keys=True))
