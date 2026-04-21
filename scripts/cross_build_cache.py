"""
Build a reward cache for a cross-domain benchmark (HotpotQA / MuSiQue / PubMedQA).

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 6 Section 7.4 (domain-specific fine-tuning claim)

Unlike the main build_cache.py which loads the shared university corpus,
this builder uses the PER-QUERY contexts provided by each benchmark. For
each query:
  1. Build a small per-query FAISS index from its supplied paragraphs.
  2. For each weight on the 0.1 simplex grid (66 combos), call the LLM
     once with those paragraphs merged at that weight, compute
     RewardComponents (F1_strict × 0.5 + EM × 0.3 + Faith × 0.2 - 0.1·
     max(0, lat−5)), insert into SQLite cache.

Graph and Ontology are empty for these benchmarks — only α is active,
so the effective search space collapses from 66 to 11 points per query.
We still record all 66 entries for schema consistency with M5 cache.

Cost (gpt-4o-mini):
  HotpotQA 300 × 66 = 19,800 entries ≈ \$6, ~1h with 10 workers.
  MuSiQue 300 × 66  = 19,800 entries ≈ \$6, ~1h with 10 workers.
  PubMedQA 250 × 66 = 16,500 entries ≈ \$5, ~50 min with 10 workers.

Usage:
    python scripts/cross_build_cache.py \\
        --benchmark hotpotqa \\
        --input data/hotpotqa/hard_300.json \\
        --output cache/hotpotqa.sqlite \\
        --grid 10 --workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dwa.base import DWAWeights  # noqa: E402
from src.eval.metrics import evaluate_single  # noqa: E402
from src.rag.triple_hybrid_rag import PROMPT_TEMPLATE, merge_contexts  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402
from src.utils.offline_cache import (  # noqa: E402
    OfflineCache,
    RewardComponents,
    enumerate_simplex,
    simplex_size,
)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("cross_build_cache")

TOP_K = 3


# ---------- benchmark loaders ----------

def load_hotpotqa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for r in raw:
        ctx = r.get("context", {})
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        paragraphs = [
            f"{titles[i]}: " + " ".join(sents[i])
            for i in range(min(len(titles), len(sents)))
        ]
        items.append({
            "qid": str(r["id"]),
            "question": r["question"],
            "answer": r["answer"],
            "contexts": paragraphs,
        })
    return items


def load_musique(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for r in raw:
        paragraphs = [
            f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
            for p in r.get("paragraphs", [])
        ]
        items.append({
            "qid": str(r["id"]),
            "question": r["question"],
            "answer": r["answer"],
            "contexts": paragraphs,
        })
    return items


def load_pubmedqa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for r in raw:
        ctx = r.get("context", {})
        contexts = ctx.get("contexts", [])
        gold = r.get("long_answer") or r.get("final_decision", "")
        items.append({
            "qid": str(r["pubid"]),
            "question": r["question"],
            "answer": gold,
            "contexts": contexts,
        })
    return items


LOADERS: dict[str, Callable[[Path], list[dict]]] = {
    "hotpotqa": load_hotpotqa,
    "musique": load_musique,
    "pubmedqa": load_pubmedqa,
}


# ---------- LLM + retrieval pre-compute ----------

class _OpenAILLM:
    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI
        self._client = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=500)

    def generate(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


def precompute_retrievals(items: list[dict]) -> dict[str, list[str]]:
    """Build per-query VectorStore once, save top-K contexts."""
    out: dict[str, list[str]] = {}
    t0 = time.time()
    for i, item in enumerate(items):
        if not item["contexts"]:
            out[item["qid"]] = []
            continue
        vs = VectorStore()
        vs.add_documents(item["contexts"])
        hits = vs.search(item["question"], top_k=TOP_K)
        out[item["qid"]] = [d for d, _ in hits]
        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            logger.info("  retrievals %d/%d (%.1fs)", i + 1, len(items), time.time() - t0)
    return out


# ---------- reward fn ----------

def make_reward_fn(items_by_qid: dict[str, dict], retrievals: dict[str, list[str]],
                   llm: _OpenAILLM):
    def reward_fn(qid: str, query: str, weights: DWAWeights) -> RewardComponents:
        item = items_by_qid[qid]
        gold = item["answer"]
        v_ctxs = retrievals[qid]
        g_ctxs: list[str] = []
        o_ctxs: list[str] = []

        start = time.time()
        context = merge_contexts(v_ctxs, g_ctxs, o_ctxs, weights, TOP_K)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        answer = llm.generate(prompt)
        latency = time.time() - start

        ev = evaluate_single(
            pred=answer, gold=gold, retrieved_docs=v_ctxs,
            contexts=v_ctxs, k=TOP_K,
        )
        return RewardComponents(
            f1=ev.f1, em=ev.em_norm, faithfulness=ev.faithfulness, latency=latency,
        )
    return reward_fn


class _ProgressPrinter:
    def __init__(self, total: int, every: int = 500) -> None:
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

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=list(LOADERS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_seed(args.seed)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    items = LOADERS[args.benchmark](args.input)
    if args.limit:
        items = items[: args.limit]
    items_by_qid = {it["qid"]: it for it in items}

    total = len(items) * simplex_size(args.grid)
    logger.info(
        "Benchmark: %s | %d queries × %d weights = %d entries | workers=%d",
        args.benchmark, len(items), simplex_size(args.grid), total, args.workers,
    )

    logger.info("Phase 1: per-query retrieval pre-compute...")
    retrievals = precompute_retrievals(items)

    logger.info("Phase 2: reward computation (LLM calls)...")
    llm = _OpenAILLM()
    reward_fn = make_reward_fn(items_by_qid, retrievals, llm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = OfflineCache(args.output)
    progress = _ProgressPrinter(total=total, every=max(500, total // 20))

    t0 = time.time()
    written = cache.build(
        queries=[(it["qid"], it["question"]) for it in items],
        reward_fn=reward_fn,
        grid=args.grid,
        n_workers=args.workers,
        skip_existing=True,
        on_progress=progress,
    )
    elapsed = time.time() - t0

    logger.info("Done. Wrote %d entries in %.1fs (%.1f/s)",
                written, elapsed, written / max(elapsed, 1e-6))
    logger.info("Cache stats: %s", cache.stats())
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
