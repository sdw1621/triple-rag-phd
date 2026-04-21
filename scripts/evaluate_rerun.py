"""
Re-run the pipeline for a policy, recording dual F1 (strict + substring).

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>

Produces thesis Ch.6 headline numbers by running the full retrieval + LLM
pipeline once per query under a chosen policy, and computing BOTH:
    - F1_strict (our rigorous token-set F1 with Korean particle stripping)
    - F1_substring (loose list-overlap F1, aligns with JKSCI 2025 style)

The two metrics are reported side-by-side so:
    a) L-DWA's improvement is measurable under a strict evaluator
    b) journal absolute numbers (e.g. F1 ≈ 0.86) are reproducible when
       evaluated under the looser metric

Unlike evaluate_on_cache.py, this script actually calls the LLM again — it
does NOT use the reward cache. This gives us raw LLM outputs for case
studies and dual-metric comparison.

Supported policies (``--policy``):
    rdwa                 Rule-based DWA baseline
    ldwa:<ckpt_path>     L-DWA from PPOTrainer checkpoint
    uniform              (1/3, 1/3, 1/3)
    vector-only          (1, 0, 0)
    graph-only           (0, 1, 0)
    ontology-only        (0, 0, 1)
    fixed:a,b,g          Fixed weight e.g. fixed:0.5,0.3,0.2
    oracle:<cache_path>  Per-query argmax from the reward cache

Usage (example for R-DWA on full 5000 QA with 10 parallel LLM workers):
    python scripts/evaluate_rerun.py \\
        --qa data/university/gold_qa_5000.json \\
        --policy rdwa \\
        --output results/rerun_rdwa.json \\
        --workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
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
from src.dwa.fixed import FixedWeightsDWA  # noqa: E402
from src.dwa.rdwa import RuleBasedDWA  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    exact_match,
    f1_char,
    f1_score,
    f1_substring,
    faithfulness,
)
from src.intent.rule_based import QueryIntent, RuleBasedIntent  # noqa: E402
from src.rag.graph_store import GraphStore  # noqa: E402,F401
from src.rag.ontology_store import OntologyStore  # noqa: E402,F401
from src.rag.triple_hybrid_rag import (  # noqa: E402
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_LIST,
    merge_contexts,
)
from src.rag.university_loader import (  # noqa: E402
    build_documents,
    build_graph,
    build_ontology,
)
from src.rag.vector_store import VectorStore  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("evaluate_rerun")

TOP_K = 3


# ---------- LLM wrapper ----------

class _OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 max_tokens: int = 500) -> None:
        from langchain_openai import ChatOpenAI
        self._client = ChatOpenAI(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

    def generate(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


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


def policy_fn(spec: str) -> Callable[[str, QueryIntent], DWAWeights]:
    """Return a callable (query, intent) -> DWAWeights."""
    analyzer_not_needed = None  # placeholder to make tools happy

    if spec == "rdwa":
        dwa: BaseDWA = RuleBasedDWA()
        return dwa.compute
    if spec == "uniform":
        w = DWAWeights(1 / 3, 1 / 3, 1 / 3)
        return lambda q, i: w
    if spec == "vector-only":
        w = DWAWeights(1.0, 0.0, 0.0)
        return lambda q, i: w
    if spec == "graph-only":
        w = DWAWeights(0.0, 1.0, 0.0)
        return lambda q, i: w
    if spec == "ontology-only":
        w = DWAWeights(0.0, 0.0, 1.0)
        return lambda q, i: w
    if spec.startswith("fixed:"):
        a, b, g = map(float, spec[len("fixed:"):].split(","))
        w = DWAWeights(a, b, g)
        return lambda q, i: w
    if spec.startswith("oracle:"):
        cache_path = Path(spec[len("oracle:"):])
        return _oracle_from_cache(cache_path)
    if spec.startswith("ldwa:"):
        return _ldwa_policy(Path(spec[len("ldwa:"):]))
    raise ValueError(f"unknown policy spec: {spec}")


def _oracle_from_cache(cache_path: Path):
    logger.info("Loading oracle best weights from cache: %s", cache_path)
    conn = sqlite3.connect(str(cache_path))
    best: dict[str, tuple[int, int, int]] = {}
    best_R: dict[str, float] = {}
    cur = conn.execute(
        "SELECT query_id, alpha_int, beta_int, gamma_int, f1, em, faithfulness, latency FROM rewards"
    )
    for qid, a, b, g, f1, em, faith, lat in cur:
        R = 0.5 * f1 + 0.3 * em + 0.2 * faith - 0.1 * max(0.0, lat - 5.0)
        if qid not in best_R or R > best_R[qid]:
            best_R[qid] = R
            best[qid] = (a, b, g)
    conn.close()
    logger.info("Oracle loaded for %d queries", len(best))

    def policy(query: str, intent: QueryIntent) -> DWAWeights:
        # This function is called per query; we need to know which qid it is.
        # We'll override the main loop to use qid-aware lookup instead.
        raise NotImplementedError("Oracle policy must be used via qid_to_weights")

    policy.qid_to_triple = best  # attach the lookup
    return policy


def _ldwa_policy(ckpt_path: Path):
    import torch
    from src.ppo.actor_critic import ActorCritic
    from src.ppo.mdp import State, extract_query_meta, extract_source_stats

    logger.info("Loading L-DWA checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ac = ActorCritic()
    ac.load_state_dict(ckpt["actor_critic"])
    ac.eval()

    # vector/graph/ontology are built in main(); we need them here.
    # We attach a "needs_pipeline" marker; main() will inject.
    def policy(query: str, intent: QueryIntent, *, vector=None, graph=None, ontology=None) -> DWAWeights:
        v_hits = vector.search(query, top_k=TOP_K)
        v_scores = [s for _, s in v_hits]
        g_paths = graph.search(query, top_k=TOP_K)
        o_facts = ontology.search(query, top_k=TOP_K)
        state = State(
            density=intent.density,
            intent_logits=(0.0, 0.0, 0.0),
            source_stats=extract_source_stats(v_scores, [1.0] * len(g_paths), [1.0] * len(o_facts)),
            query_meta=extract_query_meta(query, intent),
        )
        with torch.no_grad():
            action, _ = ac.act_mean(state.to_tensor().unsqueeze(0))
        w = action[0].tolist()
        return DWAWeights(float(w[0]), float(w[1]), float(w[2]))

    policy.needs_pipeline = True
    return policy


# ---------- evaluator worker ----------

def evaluate_one(
    item: dict,
    retrievals: dict,  # {qid: {"intent":..., "v":..., "g":..., "o":...}}
    policy,
    llm: _OpenAILLM,
    qid_oracle: dict | None,
    pipeline_bits: dict | None,
    prompt_template: str,
) -> dict:
    qid = str(item["id"])
    query = item["query"]
    gold = item["answer"]
    qtype = item.get("type", "unknown")
    intent = retrievals[qid]["intent"]
    v_ctxs = retrievals[qid]["v"]
    g_ctxs = retrievals[qid]["g"]
    o_ctxs = retrievals[qid]["o"]

    # Decide weights
    if qid_oracle is not None:
        a, b, g = qid_oracle[qid]
        weights = DWAWeights(a / 10, b / 10, g / 10)
    elif getattr(policy, "needs_pipeline", False):
        weights = policy(query, intent, **pipeline_bits)
    else:
        weights = policy(query, intent)

    # Snap to cache-compatible simplex for consistent reporting; also keeps
    # the (α,β,γ) sensible if a policy returns tiny float drift.
    a_i, b_i, g_i = snap_to_simplex_int(weights)
    snapped = DWAWeights(a_i / 10, b_i / 10, g_i / 10)

    start = time.time()
    context = merge_contexts(v_ctxs, g_ctxs, o_ctxs, snapped, TOP_K)
    prompt = prompt_template.format(context=context, query=query)
    answer = llm.generate(prompt)
    latency = time.time() - start

    all_ctxs = v_ctxs + g_ctxs + o_ctxs
    return {
        "qid": qid,
        "type": qtype,
        "gold": gold,
        "answer": answer,
        "weights": (a_i, b_i, g_i),
        "metrics": {
            "f1_strict": f1_score(answer, gold),
            "f1_substring": f1_substring(answer, gold),
            "f1_char": f1_char(answer, gold),
            "em_norm": exact_match(answer, gold, normalize=True),
            "em_raw": exact_match(answer, gold, normalize=False),
            "faithfulness": faithfulness(answer, v_ctxs),
            "latency": latency,
        },
    }


# ---------- aggregation ----------

def aggregate(results: list[dict]) -> dict:
    def stat(arr):
        if not arr:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}

    f1_s = [r["metrics"]["f1_strict"] for r in results]
    f1_sub = [r["metrics"]["f1_substring"] for r in results]
    f1_c = [r["metrics"].get("f1_char", 0.0) for r in results]
    em_n = [r["metrics"]["em_norm"] for r in results]
    faith = [r["metrics"]["faithfulness"] for r in results]
    lat = [r["metrics"]["latency"] for r in results]

    by_type: dict[str, dict[str, list[float]]] = {}
    for r in results:
        t = r["type"]
        b = by_type.setdefault(
            t, {"f1_s": [], "f1_sub": [], "f1_c": [], "em": [], "faith": []}
        )
        b["f1_s"].append(r["metrics"]["f1_strict"])
        b["f1_sub"].append(r["metrics"]["f1_substring"])
        b["f1_c"].append(r["metrics"].get("f1_char", 0.0))
        b["em"].append(r["metrics"]["em_norm"])
        b["faith"].append(r["metrics"]["faithfulness"])

    from collections import Counter
    weights_c = Counter(tuple(r["weights"]) for r in results)

    return {
        "overall": {
            "F1_strict": stat(f1_s),
            "F1_substring": stat(f1_sub),
            "F1_char": stat(f1_c),
            "EM_norm": stat(em_n),
            "Faithfulness": stat(faith),
            "Latency": stat(lat),
        },
        "by_type": {
            t: {
                "F1_strict": stat(b["f1_s"]),
                "F1_substring": stat(b["f1_sub"]),
                "F1_char": stat(b["f1_c"]),
                "EM_norm": stat(b["em"]),
                "Faithfulness": stat(b["faith"]),
            }
            for t, b in by_type.items()
        },
        "top_weights": [
            {"weight": f"(α={a / 10:.1f}, β={b / 10:.1f}, γ={g / 10:.1f})", "count": n}
            for (a, b, g), n in weights_c.most_common(5)
        ],
    }


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-samples", type=int, default=50,
                        help="Save this many full (query, gold, answer) samples to output JSON.")
    parser.add_argument("--retrieval-cache", type=Path, default=None,
                        help="Pickle path for retrievals — first run creates, subsequent runs load.")
    parser.add_argument(
        "--prompt-style",
        choices=("sentence", "list"),
        default="sentence",
        help=(
            "sentence (default): original free-form prompt used during M5 cache "
            "build and M6 PPO training. list: structured prompt that forces the "
            "LLM to emit comma-separated items, matching list-typed gold answers. "
            "'list' recovers strict-F1 reproducibility with JKSCI 2025."
        ),
    )
    args = parser.parse_args(argv)

    prompt_template = (
        PROMPT_TEMPLATE_LIST if args.prompt_style == "list" else PROMPT_TEMPLATE
    )
    logger.info("Prompt style: %s", args.prompt_style)

    set_seed(args.seed)

    with args.qa.open(encoding="utf-8") as f:
        qa = json.load(f)
    if args.limit:
        qa = qa[: args.limit]
    logger.info("QA: %d queries", len(qa))

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    # Build pipeline (always — needed for LLM even if retrieval cache exists,
    # and L-DWA policy may need vector/graph/ontology)
    import pickle
    logger.info("Building pipeline components...")
    vector = VectorStore()
    vector.add_documents(build_documents())
    graph = build_graph()
    ontology = build_ontology()
    analyzer = RuleBasedIntent()
    llm = _OpenAILLM()

    # Pre-compute retrievals (or load from cache)
    if args.retrieval_cache and args.retrieval_cache.exists():
        logger.info("Loading retrievals from cache: %s", args.retrieval_cache)
        with args.retrieval_cache.open("rb") as f:
            retrievals = pickle.load(f)
        assert len(retrievals) >= len(qa), \
            f"retrieval cache has {len(retrievals)} entries, need ≥ {len(qa)}"
    else:
        logger.info("Pre-computing retrievals for %d queries...", len(qa))
        retrievals: dict[str, dict] = {}
        t0 = time.time()
        for i, item in enumerate(qa):
            qid = str(item["id"])
            q = item["query"]
            intent = analyzer.analyze(q)
            v_hits = vector.search(q, top_k=TOP_K)
            retrievals[qid] = {
                "intent": intent,
                "v": [d for d, _ in v_hits],
                "g": graph.search(q, top_k=TOP_K),
                "o": ontology.search(q, top_k=TOP_K),
            }
            if (i + 1) % 500 == 0 or (i + 1) == len(qa):
                logger.info("  retrievals %d/%d (%.1fs)", i + 1, len(qa), time.time() - t0)
        if args.retrieval_cache:
            args.retrieval_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.retrieval_cache.open("wb") as f:
                pickle.dump(retrievals, f)
            logger.info("Retrievals saved to %s", args.retrieval_cache)

    # Resolve policy
    logger.info("Resolving policy: %s", args.policy)
    policy = policy_fn(args.policy)
    qid_oracle = getattr(policy, "qid_to_triple", None)
    pipeline_bits = (
        {"vector": vector, "graph": graph, "ontology": ontology}
        if getattr(policy, "needs_pipeline", False)
        else None
    )

    # Run LLM evaluation
    logger.info("Running LLM evaluation (workers=%d)...", args.workers)
    results: list[dict] = []
    t0 = time.time()

    def _work(item):
        return evaluate_one(
            item, retrievals, policy, llm, qid_oracle, pipeline_bits, prompt_template
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_work, item): item for item in qa}
        for i, fut in enumerate(as_completed(futures)):
            try:
                results.append(fut.result())
            except Exception as e:
                logger.exception("worker failed on qid=%s: %s", futures[fut]["id"], e)
            if (i + 1) % 500 == 0 or (i + 1) == len(qa):
                logger.info("  evaluated %d/%d (%.1fs)", i + 1, len(qa), time.time() - t0)

    logger.info("All queries done in %.1fs", time.time() - t0)

    # Aggregate
    agg = aggregate(results)
    o = agg["overall"]
    logger.info(
        "Overall: F1s=%.4f±%.4f F1sub=%.4f±%.4f F1char=%.4f±%.4f EM=%.4f Faith=%.4f±%.4f Lat=%.2fs",
        o["F1_strict"]["mean"], o["F1_strict"]["std"],
        o["F1_substring"]["mean"], o["F1_substring"]["std"],
        o["F1_char"]["mean"], o["F1_char"]["std"],
        o["EM_norm"]["mean"],
        o["Faithfulness"]["mean"], o["Faithfulness"]["std"],
        o["Latency"]["mean"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = results[: args.save_samples]
    payload = {
        "policy": args.policy,
        "prompt_style": args.prompt_style,
        "n_queries": len(results),
        "aggregate": agg,
        "samples": [
            {
                "qid": r["qid"], "type": r["type"], "gold": r["gold"], "answer": r["answer"],
                "weights": r["weights"], "metrics": r["metrics"],
            }
            for r in samples
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s (%d queries + %d samples)", args.output, len(results), len(samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
