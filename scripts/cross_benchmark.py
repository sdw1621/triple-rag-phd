"""
Cross-benchmark evaluation on HotpotQA / MuSiQue / PubMedQA.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 6 Section 8 (cross-domain generalization)

Each benchmark ships its own per-query context paragraphs. We build a
small per-query FAISS index from those contexts, then run the pipeline
under each policy (R-DWA, L-DWA, Uniform) and measure F1_strict,
F1_substring, EM, and Faithfulness.

Caveats (documented in thesis Ch.6 §8):
- L-DWA was trained on Korean university-domain states; state features
  like Korean-regex entity density return ~0 for English queries.
  This is a *transfer* experiment, not a domain-native L-DWA.
- Graph and Ontology are empty (no domain-specific structures), so only
  α-weighted Vector retrieval contributes meaningful context.
  This effectively probes whether L-DWA's α component stays reasonable.

Usage:
    python scripts/cross_benchmark.py \\
        --benchmark hotpotqa \\
        --input data/hotpotqa/hard_300.json \\
        --policy rdwa \\
        --output results/cross_hotpotqa_rdwa.json \\
        --workers 5
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

from src.dwa.base import BaseDWA, DWAWeights  # noqa: E402
from src.dwa.rdwa import RuleBasedDWA  # noqa: E402
from src.eval.metrics import exact_match, f1_score, f1_substring, faithfulness  # noqa: E402
from src.intent.rule_based import QueryIntent, RuleBasedIntent  # noqa: E402
from src.rag.triple_hybrid_rag import PROMPT_TEMPLATE, merge_contexts  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K = 3


# ---------- benchmark loaders ----------

def load_hotpotqa(path: Path) -> list[dict]:
    """Return a list of {qid, question, answer, contexts, type}."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for r in raw:
        # context: dict with "title" (list) and "sentences" (list of sent lists)
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
            "type": r.get("type", "unknown"),
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
            "type": "multi_hop",  # MuSiQue is all multi-hop
        })
    return items


def load_pubmedqa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for r in raw:
        ctx = r.get("context", {})
        contexts = ctx.get("contexts", [])
        # long_answer as gold
        gold = r.get("long_answer") or r.get("final_decision", "")
        items.append({
            "qid": str(r["pubid"]),
            "question": r["question"],
            "answer": gold,
            "contexts": contexts,
            "type": r.get("final_decision", "unknown"),
        })
    return items


BENCHMARK_LOADERS: dict[str, Callable[[Path], list[dict]]] = {
    "hotpotqa": load_hotpotqa,
    "musique": load_musique,
    "pubmedqa": load_pubmedqa,
}


# ---------- policy resolution ----------

def snap_to_simplex_int(weights: DWAWeights, grid: int = 10) -> tuple[int, int, int]:
    a = round(weights.alpha * grid)
    b = round(weights.beta * grid)
    g = round(weights.gamma * grid)
    residual = grid - (a + b + g)
    if residual != 0:
        if a >= b and a >= g:
            a += residual
        elif b >= g:
            b += residual
        else:
            g += residual
    return (max(0, min(grid, a)), max(0, min(grid, b)), max(0, min(grid, g)))


class _Uniform(BaseDWA):
    def compute(self, query, intent):
        return DWAWeights(1 / 3, 1 / 3, 1 / 3)


class _VectorOnly(BaseDWA):
    def compute(self, query, intent):
        return DWAWeights(1.0, 0.0, 0.0)


def build_policy(spec: str):
    """Return (policy_callable, needs_pipeline_for_ldwa)."""
    if spec == "rdwa":
        return (RuleBasedDWA().compute, False)
    if spec == "uniform":
        return (_Uniform().compute, False)
    if spec == "vector-only":
        return (_VectorOnly().compute, False)
    if spec.startswith("ldwa:"):
        import torch
        from src.ppo.actor_critic import ActorCritic
        from src.ppo.mdp import State, extract_query_meta, extract_source_stats

        ckpt_path = Path(spec[len("ldwa:"):])
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        ac = ActorCritic()
        ac.load_state_dict(ckpt["actor_critic"])
        ac.eval()

        def policy(query: str, intent: QueryIntent, *, v_hits=None) -> DWAWeights:
            v_scores = [s for _, s in v_hits] if v_hits else []
            state = State(
                density=intent.density,
                intent_logits=(0.0, 0.0, 0.0),
                source_stats=extract_source_stats(v_scores, [], []),
                query_meta=extract_query_meta(query, intent),
            )
            with torch.no_grad():
                action, _ = ac.act_mean(state.to_tensor().unsqueeze(0))
            w = action[0].tolist()
            return DWAWeights(float(w[0]), float(w[1]), float(w[2]))

        policy.needs_v_hits = True
        return (policy, True)
    raise ValueError(f"unknown policy spec: {spec}")


# ---------- LLM wrapper ----------

class _OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 max_tokens: int = 500):
        from langchain_openai import ChatOpenAI
        self._client = ChatOpenAI(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

    def generate(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


# ---------- per-query work ----------

def evaluate_one(item: dict, policy, is_ldwa: bool, analyzer: RuleBasedIntent,
                 llm: _OpenAILLM) -> dict:
    """Build a per-query VectorStore from provided contexts, run pipeline."""
    qid = item["qid"]
    question = item["question"]
    gold = item["answer"]
    contexts = item["contexts"]

    # Per-query vector store (small, ~10 docs → ~10 embedding calls each)
    vector = VectorStore()
    if contexts:
        vector.add_documents(contexts)
    v_hits = vector.search(question, top_k=TOP_K) if contexts else []
    v_ctxs = [d for d, _ in v_hits]
    # graph/ontology are empty in cross-benchmark (no domain structure)
    g_ctxs: list[str] = []
    o_ctxs: list[str] = []

    intent = analyzer.analyze(question)

    if is_ldwa:
        weights = policy(question, intent, v_hits=v_hits)
    else:
        weights = policy(question, intent)

    a_i, b_i, g_i = snap_to_simplex_int(weights)
    snapped = DWAWeights(a_i / 10, b_i / 10, g_i / 10)

    start = time.time()
    context = merge_contexts(v_ctxs, g_ctxs, o_ctxs, snapped, TOP_K)
    prompt = PROMPT_TEMPLATE.format(context=context, query=question)
    answer = llm.generate(prompt)
    latency = time.time() - start

    return {
        "qid": qid,
        "type": item["type"],
        "gold": gold,
        "answer": answer,
        "weights": (a_i, b_i, g_i),
        "metrics": {
            "f1_strict": f1_score(answer, gold),
            "f1_substring": f1_substring(answer, gold),
            "em_norm": exact_match(answer, gold, normalize=True),
            "em_raw": exact_match(answer, gold, normalize=False),
            "faithfulness": faithfulness(answer, v_ctxs),
            "latency": latency,
        },
    }


def aggregate(results: list[dict]) -> dict:
    def stat(arr):
        if not arr:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}

    return {
        "n_queries": len(results),
        "overall": {
            "F1_strict": stat([r["metrics"]["f1_strict"] for r in results]),
            "F1_substring": stat([r["metrics"]["f1_substring"] for r in results]),
            "EM_norm": stat([r["metrics"]["em_norm"] for r in results]),
            "Faithfulness": stat([r["metrics"]["faithfulness"] for r in results]),
            "Latency": stat([r["metrics"]["latency"] for r in results]),
        },
    }


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-samples", type=int, default=20)
    args = parser.parse_args(argv)

    set_seed(args.seed)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    loader = BENCHMARK_LOADERS[args.benchmark]
    items = loader(args.input)
    if args.limit:
        items = items[: args.limit]
    logger.info("Benchmark %s: loaded %d items", args.benchmark, len(items))

    policy, is_ldwa = build_policy(args.policy)
    analyzer = RuleBasedIntent()
    llm = _OpenAILLM()

    logger.info("Running %d workers on %d items (%s)", args.workers, len(items), args.policy)
    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, item, policy, is_ldwa, analyzer, llm): item
            for item in items
        }
        for i, fut in enumerate(as_completed(futures)):
            try:
                results.append(fut.result())
            except Exception as e:
                logger.exception("worker failed on qid=%s: %s", futures[fut]["qid"], e)
            if (i + 1) % 50 == 0 or (i + 1) == len(items):
                logger.info("  evaluated %d/%d (%.1fs)", i + 1, len(items), time.time() - t0)

    agg = aggregate(results)
    o = agg["overall"]
    logger.info(
        "Overall: F1s=%.4f F1sub=%.4f EM=%.4f Faith=%.4f Lat=%.2fs",
        o["F1_strict"]["mean"], o["F1_substring"]["mean"],
        o["EM_norm"]["mean"], o["Faithfulness"]["mean"], o["Latency"]["mean"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": args.benchmark,
        "policy": args.policy,
        "n_queries": len(results),
        "aggregate": agg,
        "samples": [
            {k: r[k] for k in ("qid", "type", "gold", "answer", "weights", "metrics")}
            for r in results[: args.save_samples]
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
