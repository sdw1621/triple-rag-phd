"""Integration tests for src.rag.triple_hybrid_rag.TripleHybridRAG.

Uses a HashEmbedder + EchoLLM to keep the pipeline fully offline.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.dwa.rdwa import RuleBasedDWA
from src.rag.graph_store import GraphStore
from src.rag.ontology_store import OntologyStore
from src.rag.triple_hybrid_rag import RAGResult, TripleHybridRAG
from src.rag.vector_store import VectorStore


class HashEmbedder:
    DIM = 64

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.DIM, dtype="float32")
        for tok in text.split():
            h = hashlib.md5(tok.encode("utf-8")).digest()
            for i in range(0, len(h), 4):
                vec[int.from_bytes(h[i : i + 4], "big") % self.DIM] += 1.0
        if not vec.any():
            vec[0] = 1.0
        return vec.tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._embed(d) for d in documents]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


class EchoLLM:
    """Deterministic LLM stand-in: returns the prompt's question line."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Extract the question after "Question: "
        for line in prompt.splitlines():
            if line.startswith("Question:"):
                return line[len("Question:"):].strip()
        return prompt[:50]


@pytest.fixture
def pipeline() -> TripleHybridRAG:
    vec = VectorStore(embedder=HashEmbedder())
    vec.add_documents([
        "김철수는 컴퓨터공학과 소속의 인공지능 연구자이다.",
        "이영희는 인공지능학과 소속의 딥러닝 연구자이다.",
        "박민수는 컴퓨터공학과 소속의 자연어처리 연구자이다.",
    ])

    graph = GraphStore(max_depth=3)
    graph.add_node("p1", name="김철수", node_type="Professor")
    graph.add_node("p2", name="이영희", node_type="Professor")
    graph.add_node("d1", name="컴퓨터공학과", node_type="Department")
    graph.add_node("d2", name="인공지능학과", node_type="Department")
    graph.add_edge("p1", "소속", "d1")
    graph.add_edge("p2", "소속", "d2")
    graph.add_edge("p1", "협력", "p2")

    ontology = OntologyStore(try_owlready=False)
    return TripleHybridRAG(
        vector_store=vec,
        graph_store=graph,
        ontology_store=ontology,
        dwa=RuleBasedDWA(),
        llm=EchoLLM(),
        top_k=3,
    )


def test_query_returns_rag_result(pipeline: TripleHybridRAG) -> None:
    result = pipeline.query("김철수 교수는 어느 학과 소속인가?")
    assert isinstance(result, RAGResult)
    assert result.answer  # non-empty
    assert result.elapsed >= 0.0
    assert abs(result.weights.alpha + result.weights.beta + result.weights.gamma - 1.0) < 1e-6


def test_query_collects_contexts_from_all_sources(pipeline: TripleHybridRAG) -> None:
    result = pipeline.query("김철수")
    # Vector + ontology should match; graph may match if seed found.
    assert any("김철수" in c for c in result.vector_contexts)
    assert any("김철수" in c for c in result.onto_contexts)
    assert any("김철수" in c for c in result.graph_contexts)


def test_intent_simple_query_classified(pipeline: TripleHybridRAG) -> None:
    result = pipeline.query("김철수 교수는 어느 학과 소속인가?")
    # 1 entity (김철수 교수) + 1 relation (소속) + 0 constraints → simple
    assert result.intent.query_type == "simple"


def test_intent_conditional_query_classified(pipeline: TripleHybridRAG) -> None:
    result = pipeline.query("40세 이하 컴퓨터공학과 교수는?")
    assert result.intent.query_type == "conditional"
    # Conditional → gamma should dominate (base 0.6 for ontology).
    assert result.weights.gamma > result.weights.alpha
    assert result.weights.gamma > result.weights.beta


def test_prompt_contains_query_and_weights(pipeline: TripleHybridRAG) -> None:
    result = pipeline.query("이영희 교수")
    assert "이영희 교수" in result.prompt
    assert "Question:" in result.prompt
    assert "Vector(" in result.prompt or "Graph(" in result.prompt or "Ontology(" in result.prompt


def test_pluggable_dwa() -> None:
    """Pipeline accepts any BaseDWA — verify interface contract."""
    from src.dwa.base import BaseDWA, DWAWeights

    class FixedDWA(BaseDWA):
        def compute(self, query, intent):  # noqa: D401
            return DWAWeights(0.4, 0.3, 0.3)

    vec = VectorStore(embedder=HashEmbedder())
    vec.add_documents(["doc1"])
    pipeline = TripleHybridRAG(
        vector_store=vec,
        graph_store=GraphStore(),
        ontology_store=OntologyStore(try_owlready=False),
        dwa=FixedDWA(),
        llm=EchoLLM(),
    )
    result = pipeline.query("test")
    assert result.weights == DWAWeights(0.4, 0.3, 0.3)
