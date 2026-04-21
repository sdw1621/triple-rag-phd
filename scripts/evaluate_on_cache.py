"""
Evaluate any DWA policy against the offline reward cache — no LLM calls.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>

The 330K-entry cache covers every (query, simplex-point) pair, so evaluating
a new policy reduces to: for each query, decide its weight → look up reward.
This lets us compute F1/EM/Faithfulness for R-DWA, L-DWA (PPO checkpoint),
ablations (vector-only / graph-only / ontology-only), uniform, and the
oracle (per-query argmax) all in seconds with zero API cost.

Supported policies (``--policy``):
    rdwa                 Rule-based DWA (thesis Ch.4 baseline / journal R-DWA)
    ldwa:<ckpt_path>     L-DWA from a PPOTrainer checkpoint
    uniform              (1/3, 1/3, 1/3)
    vector-only          (1, 0, 0)
    graph-only           (0, 1, 0)
    ontology-only        (0, 0, 1)
    fixed:a,b,g          Fixed weight e.g. fixed:0.5,0.3,0.2
    oracle               Per-query argmax over the 66 cache points (upper bound)

Usage:
    python scripts/evaluate_on_cache.py \\
        --cache cache/university.sqlite \\
        --qa data/university/gold_qa_5000.json \\
        --policy rdwa \\
        --output results/eval_rdwa.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dwa.base import BaseDWA, DWAWeights  # noqa: E402
from src.dwa.fixed import FixedWeightsDWA  # noqa: E402
from src.dwa.rdwa import RuleBasedDWA  # noqa: E402
from src.intent.rule_based import QueryIntent, RuleBasedIntent  # noqa: E402
from src.utils.offline_cache import DEFAULT_GRID, RewardComponents  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("evaluate_on_cache")


# ---------- helpers ----------

def snap_to_simplex_int(weights: DWAWeights, grid: int = DEFAULT_GRID) -> tuple[int, int, int]:
    """Snap continuous weights to the nearest integer simplex point (same as trainer)."""
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
    a = max(0, min(grid, a))
    b = max(0, min(grid, b))
    g = max(0, min(grid, g))
    return (a, b, g)


def load_query_rewards(cache_path: Path) -> dict[str, dict[tuple[int, int, int], RewardComponents]]:
    """Load all cache rows into ``{qid: {(a,b,g): RewardComponents}}``."""
    conn = sqlite3.connect(str(cache_path))
    out: dict[str, dict[tuple[int, int, int], RewardComponents]] = {}
    cur = conn.execute(
        "SELECT query_id, alpha_int, beta_int, gamma_int, f1, em, faithfulness, latency FROM rewards"
    )
    for qid, a, b, g, f1, em, faith, lat in cur:
        out.setdefault(qid, {})[(a, b, g)] = RewardComponents(
            f1=f1, em=em, faithfulness=faith, latency=lat,
        )
    conn.close()
    return out


# ---------- policy resolvers ----------

class VectorOnly(BaseDWA):
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        return DWAWeights(1.0, 0.0, 0.0)


class GraphOnly(BaseDWA):
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        return DWAWeights(0.0, 1.0, 0.0)


class OntologyOnly(BaseDWA):
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        return DWAWeights(0.0, 0.0, 1.0)


class Uniform(BaseDWA):
    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        return DWAWeights(1 / 3, 1 / 3, 1 / 3)


def build_policy(spec: str, qa_rewards: dict):
    """Parse ``--policy`` spec into a callable ``(qid, query, intent) -> (a,b,g)``."""
    if spec == "rdwa":
        dwa: BaseDWA = RuleBasedDWA()
        return _from_dwa(dwa)
    if spec == "uniform":
        return _from_dwa(Uniform())
    if spec == "vector-only":
        return _from_dwa(VectorOnly())
    if spec == "graph-only":
        return _from_dwa(GraphOnly())
    if spec == "ontology-only":
        return _from_dwa(OntologyOnly())
    if spec.startswith("fixed:"):
        a, b, g = map(float, spec[len("fixed:"):].split(","))
        return _from_dwa(FixedWeightsDWA(DWAWeights(a, b, g)))
    if spec == "oracle":
        return _make_oracle(qa_rewards)
    if spec.startswith("ldwa:"):
        ckpt = Path(spec[len("ldwa:"):])
        return _make_ldwa_policy(ckpt)
    raise ValueError(f"unknown policy spec: {spec}")


def _from_dwa(dwa: BaseDWA):
    analyzer = RuleBasedIntent()

    def policy(qid: str, query: str) -> tuple[int, int, int]:
        intent = analyzer.analyze(query)
        w = dwa.compute(query, intent)
        return snap_to_simplex_int(w)

    return policy


def _make_oracle(qa_rewards: dict):
    """Oracle: for each query, pick the argmax reward weight."""
    best: dict[str, tuple[int, int, int]] = {}
    for qid, rows in qa_rewards.items():
        best[qid] = max(rows.keys(), key=lambda k: rows[k].total_reward())

    def policy(qid: str, query: str) -> tuple[int, int, int]:
        return best[qid]

    return policy


def _make_ldwa_policy(ckpt_path: Path):
    """Load a PPOTrainer checkpoint and use ActorCritic.act_mean for weights."""
    import torch

    from src.ppo.actor_critic import ActorCritic
    from src.ppo.mdp import State, extract_query_meta, extract_source_stats
    from src.rag.graph_store import GraphStore  # noqa: F401  (to ensure import works)
    from src.rag.ontology_store import OntologyStore  # noqa: F401
    from src.rag.university_loader import build_graph, build_ontology, build_documents
    from src.rag.vector_store import VectorStore

    logger.info("Loading L-DWA checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ac = ActorCritic()
    ac.load_state_dict(ckpt["actor_critic"])
    ac.eval()

    logger.info("L-DWA needs retrievals to build state — initializing pipeline...")
    analyzer = RuleBasedIntent()
    graph = build_graph()
    ontology = build_ontology()
    vector = VectorStore()
    vector.add_documents(build_documents())

    def policy(qid: str, query: str) -> tuple[int, int, int]:
        intent = analyzer.analyze(query)
        v_hits = vector.search(query, top_k=3)
        v_scores = [s for _, s in v_hits]
        g_paths = graph.search(query, top_k=3)
        o_facts = ontology.search(query, top_k=3)
        state = State(
            density=intent.density,
            intent_logits=(0.0, 0.0, 0.0),
            source_stats=extract_source_stats(v_scores, [1.0] * len(g_paths), [1.0] * len(o_facts)),
            query_meta=extract_query_meta(query, intent),
        )
        with torch.no_grad():
            action, _ = ac.act_mean(state.to_tensor().unsqueeze(0))
        w = action[0].tolist()
        return snap_to_simplex_int(DWAWeights(float(w[0]), float(w[1]), float(w[2])))

    return policy


# ---------- evaluation ----------

def evaluate(
    qa: list[dict],
    qa_rewards: dict,
    policy,
) -> dict:
    """Apply policy to each query, look up cache, aggregate metrics."""
    per_type: dict[str, dict[str, list[float]]] = {}
    picked_weights: list[tuple[int, int, int]] = []
    t0 = time.time()

    f1s, ems, faiths, lats, Rs = [], [], [], [], []
    misses = 0

    for i, item in enumerate(qa):
        qid = str(item["id"])
        qtype = item.get("type", "unknown")
        triple = policy(qid, item["query"])
        picked_weights.append(triple)
        rc = qa_rewards.get(qid, {}).get(triple)
        if rc is None:
            misses += 1
            continue
        R = rc.total_reward()
        f1s.append(rc.f1)
        ems.append(rc.em)
        faiths.append(rc.faithfulness)
        lats.append(rc.latency)
        Rs.append(R)
        bucket = per_type.setdefault(qtype, {"f1": [], "em": [], "faith": [], "R": []})
        bucket["f1"].append(rc.f1)
        bucket["em"].append(rc.em)
        bucket["faith"].append(rc.faithfulness)
        bucket["R"].append(R)
        if (i + 1) % 1000 == 0:
            logger.info("  evaluated %d/%d (%.1fs)", i + 1, len(qa), time.time() - t0)

    # Weight distribution
    from collections import Counter
    w_counter = Counter(picked_weights)
    top_weights = w_counter.most_common(5)

    def stat(arr):
        if not arr:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}

    return {
        "n_queries": len(qa),
        "n_misses": misses,
        "overall": {
            "F1": stat(f1s),
            "EM": stat(ems),
            "Faithfulness": stat(faiths),
            "Latency": stat(lats),
            "TotalReward": stat(Rs),
        },
        "by_type": {
            t: {
                "F1": stat(v["f1"]),
                "EM": stat(v["em"]),
                "Faithfulness": stat(v["faith"]),
                "TotalReward": stat(v["R"]),
            }
            for t, v in per_type.items()
        },
        "top_chosen_weights": [
            {"weight": f"(α={a / 10:.1f}, β={b / 10:.1f}, γ={g / 10:.1f})", "count": n}
            for (a, b, g), n in top_weights
        ],
        "mean_chosen_weight": {
            "alpha": float(np.mean([w[0] / 10 for w in picked_weights])),
            "beta": float(np.mean([w[1] / 10 for w in picked_weights])),
            "gamma": float(np.mean([w[2] / 10 for w in picked_weights])),
        },
    }


# ---------- rendering ----------

def render_summary(result: dict, policy_name: str) -> str:
    lines: list[str] = []
    a = lines.append
    o = result["overall"]

    a(f"### Policy: `{policy_name}`  —  n_queries={result['n_queries']:,}  misses={result['n_misses']}")
    a("")
    a("| metric | mean ± std |")
    a("|---|---|")
    a(f"| **F1** | **{o['F1']['mean']:.4f} ± {o['F1']['std']:.4f}** |")
    a(f"| EM | {o['EM']['mean']:.4f} ± {o['EM']['std']:.4f} |")
    a(f"| Faithfulness | {o['Faithfulness']['mean']:.4f} ± {o['Faithfulness']['std']:.4f} |")
    a(f"| Latency (s) | {o['Latency']['mean']:.4f} ± {o['Latency']['std']:.4f} |")
    a(f"| **Total Reward** | **{o['TotalReward']['mean']:.4f} ± {o['TotalReward']['std']:.4f}** |")
    a("")
    mw = result["mean_chosen_weight"]
    a(f"**Mean chosen weight**: (α={mw['alpha']:.3f}, β={mw['beta']:.3f}, γ={mw['gamma']:.3f})")
    a("")
    a("**Top 5 chosen weights**:")
    for item in result["top_chosen_weights"]:
        a(f"  - {item['weight']}: {item['count']}")
    a("")
    a("**Per-type F1 / EM / Faith / R**:")
    a("")
    a("| type | n | F1 | EM | Faith | R |")
    a("|---|---|---|---|---|---|")
    for t, v in result["by_type"].items():
        a(
            f"| {t} | {v['F1']['n']} | "
            f"{v['F1']['mean']:.3f}±{v['F1']['std']:.3f} | "
            f"{v['EM']['mean']:.3f}±{v['EM']['std']:.3f} | "
            f"{v['Faithfulness']['mean']:.3f}±{v['Faithfulness']['std']:.3f} | "
            f"{v['TotalReward']['mean']:.3f}±{v['TotalReward']['std']:.3f} |"
        )
    return "\n".join(lines)


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--policy", required=True, help="rdwa | uniform | vector-only | ldwa:<path> | ...")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logger.info("Loading cache: %s", args.cache)
    qa_rewards = load_query_rewards(args.cache)
    logger.info("Cache loaded: %d queries, total rows %d",
                len(qa_rewards), sum(len(v) for v in qa_rewards.values()))

    with args.qa.open(encoding="utf-8") as f:
        qa = json.load(f)
    if args.limit:
        qa = qa[: args.limit]
    logger.info("QA set: %d queries", len(qa))

    logger.info("Building policy: %s", args.policy)
    policy = build_policy(args.policy, qa_rewards)

    logger.info("Evaluating...")
    result = evaluate(qa, qa_rewards, policy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"policy": args.policy, **result}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote JSON: %s", args.output)

    md_path = args.output.with_suffix(".md")
    md_path.write_text(render_summary(result, args.policy), encoding="utf-8")
    logger.info("Wrote summary: %s", md_path)

    o = result["overall"]
    logger.info(
        "Summary → F1=%.4f±%.4f  EM=%.4f±%.4f  Faith=%.4f±%.4f  R=%.4f±%.4f  (misses=%d)",
        o["F1"]["mean"], o["F1"]["std"],
        o["EM"]["mean"], o["EM"]["std"],
        o["Faithfulness"]["mean"], o["Faithfulness"]["std"],
        o["TotalReward"]["mean"], o["TotalReward"]["std"],
        result["n_misses"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
