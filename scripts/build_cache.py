"""
Build the offline reward cache for PPO training.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 5 Section 4 (cache design),
                  Chapter 6 Section 10 (cost reduction)

For each query in the QA file × every (α, β, γ) on the discretized simplex,
this script:
  1. Runs Triple-Hybrid retrieval ONCE per query (weight-independent).
  2. For each weight combo: re-merges contexts under the budget split,
     calls the LLM once, measures F1/EM/Faithfulness vs the gold answer,
     and writes a RewardComponents row to SQLite.

Per-query retrieval is cached in memory so the heavy loop is N_queries ×
N_simplex_points LLM calls, NOT × 3 retrieval calls.

Usage:
    # Mock smoke test (offline, validates wiring):
    python scripts/build_cache.py \
        --qa data/university/gold_qa_5000.json \
        --output cache/smoke_mock.sqlite \
        --grid 2 --limit 100 --mock

    # Real smoke test (small, costs ~$0.20):
    python scripts/build_cache.py \
        --qa data/university/gold_qa_5000.json \
        --output cache/smoke_real.sqlite \
        --grid 2 --limit 100

    # Full overnight run (~$15, 12-15h):
    python scripts/build_cache.py \
        --qa data/university/gold_qa_5000.json \
        --output cache/university.sqlite \
        --grid 10 --workers 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Allow running from any cwd inside the container.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dwa.base import DWAWeights  # noqa: E402
from src.eval.metrics import evaluate_single  # noqa: E402
from src.intent.rule_based import QueryIntent, RuleBasedIntent  # noqa: E402
from src.rag.graph_store import GraphStore  # noqa: E402
from src.rag.ontology_store import OntologyStore  # noqa: E402
from src.rag.triple_hybrid_rag import PROMPT_TEMPLATE, merge_contexts  # noqa: E402
from src.rag.university_loader import (  # noqa: E402
    build_documents,
    build_graph,
    build_ontology,
    stats as corpus_stats,
)
from src.rag.vector_store import VectorStore  # noqa: E402
from src.utils.offline_cache import (  # noqa: E402
    OfflineCache,
    RewardComponents,
    enumerate_simplex,
    simplex_size,
)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("build_cache")

TOP_K = 3


# ---------- mock fakes (--mock mode) ----------

class _HashEmbedder:
    """Deterministic 64-d hash embedder; no API calls."""

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


class _MockLLM:
    """Returns the first non-tag line of the merged context as the answer."""

    def generate(self, prompt: str) -> str:
        in_context = False
        for line in prompt.splitlines():
            if line.startswith("Context:"):
                in_context = True
                continue
            if line.startswith("Question:"):
                break
            if in_context and line.strip() and not line.startswith("[") and "--" not in line[:5]:
                return line.strip()
        return "정보를 찾을 수 없습니다"


# ---------- real LLM wrapper ----------

class _OpenAILLM:
    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        from langchain_openai import ChatOpenAI

        self._client = ChatOpenAI(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

    def generate(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


# ---------- pipeline assembly ----------

def build_components(mock: bool, llm_model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    """Build vector_store, graph, ontology, llm, analyzer."""
    docs = build_documents()
    graph = build_graph()
    ontology = build_ontology()
    analyzer = RuleBasedIntent()

    if mock:
        embedder: Any = _HashEmbedder()
        llm: Any = _MockLLM()
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set (use --mock to skip API calls)")
        embedder = None  # VectorStore lazily creates OpenAI embedder
        llm = _OpenAILLM(llm_model, temperature, max_tokens)

    logger.info("Loading vector store: embedding %d documents...", len(docs))
    t0 = time.time()
    vector = VectorStore(embedder=embedder)
    vector.add_documents(docs)
    logger.info("Vector store built in %.1fs (%d docs, dim=%d)",
                time.time() - t0, vector.n_documents, vector._dim)

    return {
        "vector": vector,
        "graph": graph,
        "ontology": ontology,
        "llm": llm,
        "analyzer": analyzer,
        "n_docs": len(docs),
    }


# ---------- per-query retrieval pre-compute ----------

def precompute_retrievals(
    queries: list[tuple[str, str]],
    components: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """For each query, run V/G/O retrievals once (weight-independent)."""
    vector = components["vector"]
    graph = components["graph"]
    ontology = components["ontology"]
    analyzer = components["analyzer"]

    out: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for i, (qid, query) in enumerate(queries):
        intent = analyzer.analyze(query)
        v_hits = vector.search(query, top_k=TOP_K)
        v_ctxs = [doc for doc, _ in v_hits]
        g_ctxs = graph.search(query, top_k=TOP_K)
        o_ctxs = ontology.search(query, top_k=TOP_K)
        out[qid] = {
            "intent": intent,
            "v_ctxs": v_ctxs,
            "g_ctxs": g_ctxs,
            "o_ctxs": o_ctxs,
        }
        if (i + 1) % 50 == 0 or (i + 1) == len(queries):
            logger.info("  retrievals %d/%d (%.1fs)", i + 1, len(queries), time.time() - t0)
    logger.info("All retrievals done in %.1fs (%.0f qps)",
                time.time() - t0, len(queries) / max(time.time() - t0, 1e-6))
    return out


# ---------- reward fn factory ----------

def make_reward_fn(
    qa_by_id: dict[str, dict],
    retrievals: dict[str, dict[str, Any]],
    llm: Any,
):
    """Closure that computes reward for (query_id, query_text, weights)."""

    def reward_fn(query_id: str, query_text: str, weights: DWAWeights) -> RewardComponents:
        retrieval = retrievals[query_id]
        v_ctxs, g_ctxs, o_ctxs = retrieval["v_ctxs"], retrieval["g_ctxs"], retrieval["o_ctxs"]
        gold = qa_by_id[query_id]["answer"]

        start = time.time()
        context = merge_contexts(v_ctxs, g_ctxs, o_ctxs, weights, TOP_K)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query_text)
        answer = llm.generate(prompt)
        latency = time.time() - start

        ev = evaluate_single(
            pred=answer,
            gold=gold,
            retrieved_docs=v_ctxs + g_ctxs + o_ctxs,
            contexts=v_ctxs,
            k=TOP_K,
        )
        return RewardComponents(
            f1=ev.f1,
            em=ev.em_norm,
            faithfulness=ev.faithfulness,
            latency=latency,
        )

    return reward_fn


# ---------- progress printer ----------

class _ProgressPrinter:
    def __init__(self, total: int, every: int = 50) -> None:
        self.total = total
        self.every = every
        self.start = time.time()

    def __call__(self, done: int, _total: int) -> None:
        if done % self.every == 0 or done == self.total:
            elapsed = time.time() - self.start
            rate = done / max(elapsed, 1e-6)
            eta = (self.total - done) / rate if rate > 0 else float("inf")
            logger.info(
                "  cache  %d/%d (%.1f%%)  rate=%.1f/s  eta=%.0fs",
                done, self.total, 100 * done / self.total, rate, eta,
            )


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa", required=True, type=Path, help="Path to gold QA JSON file.")
    parser.add_argument("--output", required=True, type=Path, help="SQLite cache path.")
    parser.add_argument("--grid", type=int, default=10,
                        help="Discretization grid (10 = 0.1 step → 66 simplex points).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N queries (smoke testing).")
    parser.add_argument("--workers", type=int, default=1,
                        help="OfflineCache.build worker thread count.")
    parser.add_argument("--mock", action="store_true",
                        help="Use HashEmbedder + MockLLM (no API calls).")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="OpenAI model id for real mode.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip (qid, weights) entries already in the DB.")
    args = parser.parse_args(argv)

    set_seed(args.seed)

    logger.info("Mode: %s", "MOCK (offline)" if args.mock else "REAL (OpenAI)")
    logger.info("QA file: %s", args.qa)
    logger.info("Output:  %s", args.output)
    logger.info("Grid:    %d (simplex points = %d)", args.grid, simplex_size(args.grid))

    # Load QA
    with args.qa.open(encoding="utf-8") as f:
        qa_data = json.load(f)
    if args.limit:
        qa_data = qa_data[: args.limit]
    qa_by_id = {str(item["id"]): item for item in qa_data}
    queries = [(str(item["id"]), item["query"]) for item in qa_data]

    total_pairs = len(queries) * simplex_size(args.grid)
    logger.info("Plan: %d queries × %d weights = %d (query, weight) pairs",
                len(queries), simplex_size(args.grid), total_pairs)
    logger.info("Corpus stats: %s", corpus_stats())

    # Build pipeline components
    components = build_components(args.mock, args.llm_model, args.temperature, args.max_tokens)

    # Pre-compute retrievals (one per query)
    logger.info("Phase 1: pre-computing retrievals...")
    retrievals = precompute_retrievals(queries, components)

    # Build cache
    logger.info("Phase 2: computing rewards...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = OfflineCache(args.output)
    reward_fn = make_reward_fn(qa_by_id, retrievals, components["llm"])

    progress = _ProgressPrinter(total=total_pairs, every=max(50, total_pairs // 20))
    t0 = time.time()
    written = cache.build(
        queries=queries,
        reward_fn=reward_fn,
        grid=args.grid,
        n_workers=args.workers,
        skip_existing=args.skip_existing,
        on_progress=progress,
    )
    elapsed = time.time() - t0

    logger.info("Done. Wrote %d new entries in %.1fs (%.1f entries/s)",
                written, elapsed, written / max(elapsed, 1e-6))
    logger.info("Cache stats: %s", cache.stats())
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
