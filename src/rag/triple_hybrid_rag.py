"""
TripleHybridRAG — Vector + Graph + Ontology pipeline with pluggable DWA.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 3 (architecture), Chapter 5 (DWA integration)

Score-fusion formula (thesis Eq. 3-1):
    S_total = α · S_vector + β · S_graph + γ · S_ontology

DWA is injected via :class:`BaseDWA` so the pipeline supports both R-DWA
(thesis Ch. 4) and L-DWA (thesis Ch. 5) without code changes.

LLM is injected via the :class:`LLM` protocol; the test suite supplies a
deterministic mock LLM to keep unit tests offline. Production code passes a
LangChain ChatOpenAI instance.

Adapted from: hybrid-rag-comparsion/src/triple_hybrid_rag.py — DWA and LLM
turned into injection points.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from src.dwa.base import BaseDWA, DWAWeights
from src.intent.rule_based import QueryIntent, RuleBasedIntent
from src.rag.graph_store import GraphStore
from src.rag.ontology_store import OntologyStore
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = (
    "다음 컨텍스트를 기반으로 질문에 정확하게 답변하세요. "
    "컨텍스트에서 답을 찾을 수 없는 경우에만 '정보를 찾을 수 없습니다'라고 답하세요. "
    "Graph 경로 정보(A --[관계]--> B)도 참고하여 관계를 추론하세요. "
    "가능한 한 구체적이고 상세하게 답변하세요.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


def merge_contexts(
    v_ctxs: list[str],
    g_ctxs: list[str],
    o_ctxs: list[str],
    weights: DWAWeights,
    top_k: int = 3,
) -> str:
    """Allocate context slots in proportion to source weights.

    Each source gets at least one slot if it returned anything; total
    budget is ``3 * top_k`` (thesis Sec 3.4). Exposed at module level so
    the offline cache builder can call it without a TripleHybridRAG instance.
    """
    budget = top_k * 3
    n_v = max(1, round(budget * weights.alpha))
    n_g = max(1, round(budget * weights.beta))
    n_o = max(1, round(budget * weights.gamma))

    parts: list[str] = []
    if v_ctxs:
        parts.append(f"[Vector(α={weights.alpha:.2f})]\n" + "\n".join(v_ctxs[:n_v]))
    if g_ctxs:
        parts.append(f"[Graph(β={weights.beta:.2f})]\n" + "\n".join(g_ctxs[:n_g]))
    if o_ctxs:
        parts.append(f"[Ontology(γ={weights.gamma:.2f})]\n" + "\n".join(o_ctxs[:n_o]))
    return "\n\n".join(parts)


class LLM(Protocol):
    """Minimal LLM interface — anything with a ``generate(prompt) -> str`` works."""

    def generate(self, prompt: str) -> str: ...


@dataclass
class RAGResult:
    """Per-query pipeline output."""

    answer: str
    elapsed: float
    weights: DWAWeights
    intent: QueryIntent
    vector_contexts: list[str] = field(default_factory=list)
    graph_contexts: list[str] = field(default_factory=list)
    onto_contexts: list[str] = field(default_factory=list)
    prompt: str = ""

    @property
    def all_contexts(self) -> list[str]:
        return self.vector_contexts + self.graph_contexts + self.onto_contexts


class _LangChainLLM:
    """Thin wrapper around langchain_openai.ChatOpenAI (lazy import)."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        from langchain_openai import ChatOpenAI  # lazy

        self._client = ChatOpenAI(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

    def generate(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


class TripleHybridRAG:
    """End-to-end pipeline.

    Args:
        vector_store: Initialized :class:`VectorStore` (must call
            :meth:`VectorStore.add_documents` first).
        graph_store: Initialized :class:`GraphStore`.
        ontology_store: Initialized :class:`OntologyStore`.
        dwa: A :class:`BaseDWA` (R-DWA or L-DWA).
        intent_analyzer: Defaults to :class:`RuleBasedIntent`. Pass a BERT
            classifier wrapper here to use the thesis Ch.5 intent.
        llm: Injected LLM. If None, a LangChain ChatOpenAI is built lazily on
            first :meth:`query` call using ``llm_model``/``temperature``/``max_tokens``.
        top_k: Per-source top-k cap. Default 3 (thesis Sec 6.1).
        llm_model: OpenAI model id when ``llm`` is None.
        temperature: LLM temperature when ``llm`` is None. Default 0.0
            (deterministic per thesis Sec 6.1).
        max_tokens: LLM max_tokens when ``llm`` is None. Default 500.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        ontology_store: OntologyStore,
        dwa: BaseDWA,
        intent_analyzer: object | None = None,
        llm: LLM | None = None,
        top_k: int = 3,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self.vector = vector_store
        self.graph = graph_store
        self.ontology = ontology_store
        self.dwa = dwa
        self.analyzer = intent_analyzer or RuleBasedIntent()
        self.top_k = top_k
        self._llm: LLM | None = llm
        self._llm_model = llm_model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = _LangChainLLM(self._llm_model, self._temperature, self._max_tokens)
        return self._llm

    # ---------- main entry ----------

    def query(self, question: str) -> RAGResult:
        """Run intent → DWA → 3-source retrieval → fusion → LLM."""
        start = time.time()

        intent = self.analyzer.analyze(question)
        weights = self.dwa.compute(question, intent)

        v_hits = self.vector.search(question, self.top_k)
        v_ctxs = [doc for doc, _ in v_hits]
        g_ctxs = self.graph.search(question, self.top_k)
        o_ctxs = self.ontology.search(question, self.top_k)

        context = self._merge_contexts(v_ctxs, g_ctxs, o_ctxs, weights)
        prompt = PROMPT_TEMPLATE.format(context=context, query=question)
        answer = self.llm.generate(prompt)

        elapsed = time.time() - start
        return RAGResult(
            answer=answer,
            elapsed=elapsed,
            weights=weights,
            intent=intent,
            vector_contexts=v_ctxs,
            graph_contexts=g_ctxs,
            onto_contexts=o_ctxs,
            prompt=prompt,
        )

    # ---------- internals ----------

    def _merge_contexts(
        self,
        v_ctxs: list[str],
        g_ctxs: list[str],
        o_ctxs: list[str],
        weights: DWAWeights,
    ) -> str:
        """Instance-method wrapper around module-level :func:`merge_contexts`."""
        return merge_contexts(v_ctxs, g_ctxs, o_ctxs, weights, self.top_k)
