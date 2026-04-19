"""Tests for src.rag.vector_store.VectorStore using a deterministic in-memory embedder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.rag.vector_store import VectorStore


class HashEmbedder:
    """Deterministic embedder: maps each token's hash to dimensions of a 64-d vector.

    Documents sharing tokens will have similar embeddings (positive overlap),
    so cosine similarity is meaningful.
    """

    DIM: int = 64

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.DIM, dtype="float32")
        for tok in text.split():
            h = hashlib.md5(tok.encode("utf-8")).digest()
            for i in range(0, len(h), 4):
                idx = int.from_bytes(h[i : i + 4], "big") % self.DIM
                vec[idx] += 1.0
        # Avoid all-zero vectors (degenerate cosine).
        if not vec.any():
            vec[0] = 1.0
        return vec.tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._embed_one(d) for d in documents]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)


@pytest.fixture
def empty_store() -> VectorStore:
    return VectorStore(embedder=HashEmbedder())


@pytest.fixture
def populated_store() -> VectorStore:
    s = VectorStore(embedder=HashEmbedder())
    s.add_documents([
        "김철수 컴퓨터공학과 인공지능",
        "이영희 인공지능학과 딥러닝",
        "박민수 컴퓨터공학과 자연어처리",
        "정수진 인공지능학과 컴퓨터비전",
    ])
    return s


def test_empty_store_search_returns_empty(empty_store: VectorStore) -> None:
    assert empty_store.search("anything", top_k=3) == []


def test_add_documents_increments_count(empty_store: VectorStore) -> None:
    empty_store.add_documents(["a b c", "d e f"])
    assert empty_store.n_documents == 2


def test_add_empty_list_is_noop(empty_store: VectorStore) -> None:
    empty_store.add_documents([])
    assert empty_store.n_documents == 0


def test_search_returns_top_k_with_scores(populated_store: VectorStore) -> None:
    hits = populated_store.search("자연어처리", top_k=3)
    assert len(hits) == 3
    docs, scores = zip(*hits)
    # Top hit should mention 자연어처리.
    assert "자연어처리" in docs[0]
    # Scores in [0, 1] (cosine on non-negative vectors).
    assert all(0.0 <= s <= 1.0 + 1e-6 for s in scores)
    # Scores in descending order.
    assert list(scores) == sorted(scores, reverse=True)


def test_search_top_k_capped_to_corpus_size(populated_store: VectorStore) -> None:
    hits = populated_store.search("query", top_k=100)
    assert len(hits) == populated_store.n_documents


def test_dim_mismatch_on_extension_raises(empty_store: VectorStore) -> None:
    empty_store.add_documents(["abc"])

    class WrongDim:
        def embed_documents(self, documents: list[str]) -> list[list[float]]:
            return [[0.1] * 32 for _ in documents]

        def embed_query(self, query: str) -> list[float]:
            return [0.1] * 32

    empty_store._embedder = WrongDim()  # inject mismatched embedder
    with pytest.raises(ValueError, match="dim mismatch"):
        empty_store.add_documents(["xyz"])


def test_save_and_load_roundtrip(populated_store: VectorStore, tmp_path: Path) -> None:
    target = tmp_path / "vs"
    populated_store.save(target)

    restored = VectorStore(embedder=HashEmbedder())
    restored.load(target)

    assert restored.n_documents == populated_store.n_documents
    hits_a = populated_store.search("딥러닝", top_k=2)
    hits_b = restored.search("딥러닝", top_k=2)
    assert [d for d, _ in hits_a] == [d for d, _ in hits_b]


def test_save_empty_store_raises(empty_store: VectorStore, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="empty"):
        empty_store.save(tmp_path / "vs")
