"""
VectorStore — FAISS-based vector retrieval.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 3 Section 2.1, Chapter 6 Section 1.2

OpenAI text-embedding-3-small (1536-dim) → FAISS IndexFlatIP (cosine via L2
normalization), top-k=3.

Embedder is injected via the :class:`Embedder` protocol so unit tests can
substitute a deterministic in-memory embedder without API calls.

Ported from: hybrid-rag-comparsion/src/vector_store.py.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM: int = 1536  # text-embedding-3-small


class Embedder(Protocol):
    """Minimal embedder interface — anything implementing both methods works."""

    def embed_documents(self, documents: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class _OpenAIEmbedder:
    """Thin wrapper around langchain_openai.OpenAIEmbeddings (lazy import)."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        from langchain_openai import OpenAIEmbeddings  # lazy: avoid import cost in tests

        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAIEmbeddings(**kwargs)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._client.embed_documents(documents)

    def embed_query(self, query: str) -> list[float]:
        return self._client.embed_query(query)


class VectorStore:
    """FAISS IndexFlatIP vector store with cosine similarity.

    Args:
        embedding_model: OpenAI embedding model name. Used only when
            ``embedder`` is None.
        embedder: Optional injected embedder (e.g. for tests). When provided,
            ``embedding_model`` and ``api_key`` are ignored.
        api_key: OpenAI API key. Falls back to the OPENAI_API_KEY env var.

    Example:
        >>> store = VectorStore(embedder=MyEmbedder())
        >>> store.add_documents(["김철수는 컴공", "이영희는 AI학과"])
        >>> hits = store.search("컴공", top_k=1)
        >>> hits[0][0]
        '김철수는 컴공'
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedder: Embedder | None = None,
        api_key: str | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self._embedder: Embedder | None = embedder
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._index: faiss.Index | None = None
        self._documents: list[str] = []
        self._dim: int | None = None

    # ---------- lifecycle ----------

    @property
    def embedder(self) -> Embedder:
        """Return the injected embedder, lazily creating an OpenAI one if needed."""
        if self._embedder is None:
            self._embedder = _OpenAIEmbedder(self.embedding_model, self._api_key)
        return self._embedder

    @property
    def n_documents(self) -> int:
        return len(self._documents)

    # ---------- mutation ----------

    def add_documents(self, documents: list[str]) -> None:
        """Embed and add documents. Subsequent calls extend the index.

        Args:
            documents: Non-empty list of text chunks.
        """
        if not documents:
            return
        vectors = self.embedder.embed_documents(documents)
        mat = np.asarray(vectors, dtype="float32")
        if mat.ndim != 2:
            raise ValueError(f"Embedder returned non-2D matrix: shape={mat.shape}")
        faiss.normalize_L2(mat)

        if self._index is None:
            self._dim = mat.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)
        elif mat.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: existing={self._dim}, new={mat.shape[1]}"
            )
        self._index.add(mat)
        self._documents.extend(documents)
        logger.info("VectorStore add_documents: +%d (total=%d, dim=%d)",
                    len(documents), len(self._documents), self._dim)

    # ---------- retrieval ----------

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (document, cosine_similarity) tuples.

        Returns empty list if the store is empty.
        """
        if self._index is None or not self._documents:
            return []
        q_vec = np.asarray(self.embedder.embed_query(query), dtype="float32").reshape(1, -1)
        if q_vec.shape[1] != self._dim:
            raise ValueError(
                f"Query embedding dim mismatch: store={self._dim}, query={q_vec.shape[1]}"
            )
        faiss.normalize_L2(q_vec)
        k = min(top_k, len(self._documents))
        scores, idxs = self._index.search(q_vec, k)
        return [
            (self._documents[i], float(scores[0][j]))
            for j, i in enumerate(idxs[0])
            if i >= 0
        ]

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        """Persist index + documents to ``path`` (creates parent dirs)."""
        if self._index is None:
            raise RuntimeError("Cannot save empty VectorStore")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with path.with_suffix(".pkl").open("wb") as f:
            pickle.dump(
                {
                    "documents": self._documents,
                    "dim": self._dim,
                    "embedding_model": self.embedding_model,
                },
                f,
            )

    def load(self, path: Path) -> None:
        """Restore index + documents from ``path``."""
        path = Path(path)
        self._index = faiss.read_index(str(path.with_suffix(".faiss")))
        with path.with_suffix(".pkl").open("rb") as f:
            payload = pickle.load(f)
        self._documents = payload["documents"]
        self._dim = payload["dim"]
        self.embedding_model = payload.get("embedding_model", self.embedding_model)
