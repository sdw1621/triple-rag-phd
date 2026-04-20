"""
Offline reward cache — quality report for PPO readiness.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>

Validates whether the cache built by ``build_cache.py`` has enough signal
for PPO to learn from. Key questions:

    1. Signal variance per query — does reward actually change across weights?
    2. Flat-reward ratio — what fraction of queries give identical reward for
       every weight combo? (These are "uninformative" for PPO; high ratio =
       potential problem.)
    3. Best-weight distribution — where does the reward landscape peak on
       the simplex? Useful as a sanity check against thesis expectations
       (e.g., conditional queries should peak at γ-dominant weights).
    4. Per-type breakdown — simple / multi_hop / conditional reward quality.
    5. Total reward landscape — apply thesis Eq. 5-7 and report dispersion.

Outputs a markdown report at ``--output`` (default results/cache_quality.md).

Usage:
    python scripts/cache_quality_report.py \
        --cache cache/university.sqlite \
        --qa data/university/gold_qa_5000.json \
        --output results/cache_quality.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.offline_cache import RewardComponents  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("cache_quality")


# ---------- data loading ----------

def load_cache_rows(cache_path: Path) -> dict[str, list[dict]]:
    """Return {query_id: [row, row, ...]} where each row has all weight combos."""
    conn = sqlite3.connect(str(cache_path))
    by_query: dict[str, list[dict]] = defaultdict(list)
    cur = conn.execute(
        "SELECT query_id, alpha_int, beta_int, gamma_int, f1, em, faithfulness, latency FROM rewards"
    )
    for qid, a, b, g, f1, em, faith, lat in cur:
        by_query[qid].append(
            {"a": a, "b": b, "g": g, "f1": f1, "em": em, "faith": faith, "lat": lat}
        )
    conn.close()
    return by_query


def load_qa_types(qa_path: Path) -> dict[str, str]:
    with qa_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["id"]): item["type"] for item in data}


def total_reward(row: dict) -> float:
    return RewardComponents(
        f1=row["f1"], em=row["em"], faithfulness=row["faith"], latency=row["lat"]
    ).total_reward()


# ---------- analyses ----------

def summarize_global(by_query: dict[str, list[dict]]) -> dict:
    all_f1, all_em, all_faith, all_lat, all_R = [], [], [], [], []
    for rows in by_query.values():
        for r in rows:
            all_f1.append(r["f1"])
            all_em.append(r["em"])
            all_faith.append(r["faith"])
            all_lat.append(r["lat"])
            all_R.append(total_reward(r))
    return {
        "total_entries": len(all_f1),
        "n_queries": len(by_query),
        "avg_f1": float(np.mean(all_f1)),
        "avg_em": float(np.mean(all_em)),
        "avg_faith": float(np.mean(all_faith)),
        "avg_latency": float(np.mean(all_lat)),
        "avg_R": float(np.mean(all_R)),
        "std_R_global": float(np.std(all_R)),
    }


def variance_per_query(by_query: dict[str, list[dict]]) -> dict:
    """For each query: std & range of F1 and total reward across its 66 weight combos."""
    f1_stds = []
    R_stds = []
    R_ranges = []
    flat_f1_count = 0
    flat_R_count = 0
    eps = 1e-6

    per_query: list[dict] = []
    for qid, rows in by_query.items():
        f1s = np.array([r["f1"] for r in rows])
        Rs = np.array([total_reward(r) for r in rows])
        f1_std = float(f1s.std())
        R_std = float(Rs.std())
        R_rng = float(Rs.max() - Rs.min())
        f1_stds.append(f1_std)
        R_stds.append(R_std)
        R_ranges.append(R_rng)
        if f1_std < eps:
            flat_f1_count += 1
        if R_std < eps:
            flat_R_count += 1
        per_query.append(
            {"qid": qid, "f1_std": f1_std, "R_std": R_std, "R_range": R_rng}
        )

    return {
        "n_queries": len(by_query),
        "f1_std_mean": float(np.mean(f1_stds)),
        "f1_std_median": float(np.median(f1_stds)),
        "R_std_mean": float(np.mean(R_stds)),
        "R_std_median": float(np.median(R_stds)),
        "R_range_mean": float(np.mean(R_ranges)),
        "R_range_median": float(np.median(R_ranges)),
        "R_range_p90": float(np.percentile(R_ranges, 90)),
        "R_range_p99": float(np.percentile(R_ranges, 99)),
        "flat_f1_queries": flat_f1_count,
        "flat_f1_ratio": flat_f1_count / len(by_query),
        "flat_R_queries": flat_R_count,
        "flat_R_ratio": flat_R_count / len(by_query),
        "_per_query": per_query,
    }


def best_weight_distribution(by_query: dict[str, list[dict]]) -> dict:
    """Histogram of argmax-reward weight for each query."""
    best_weights = Counter()
    best_a_vals = []
    best_b_vals = []
    best_g_vals = []
    for rows in by_query.values():
        Rs = [(total_reward(r), r) for r in rows]
        Rs.sort(key=lambda x: x[0], reverse=True)
        best = Rs[0][1]
        a, b, g = best["a"], best["b"], best["g"]
        best_weights[(a, b, g)] += 1
        best_a_vals.append(a / 10)
        best_b_vals.append(b / 10)
        best_g_vals.append(g / 10)

    top5 = best_weights.most_common(5)
    return {
        "top5_best_weights": [
            {"weight": f"(α={a / 10:.1f}, β={b / 10:.1f}, γ={g / 10:.1f})", "count": n}
            for (a, b, g), n in top5
        ],
        "mean_best_alpha": float(np.mean(best_a_vals)),
        "mean_best_beta": float(np.mean(best_b_vals)),
        "mean_best_gamma": float(np.mean(best_g_vals)),
    }


def per_type_breakdown(
    by_query: dict[str, list[dict]],
    qa_types: dict[str, str],
) -> dict:
    """Group by query type (simple / multi_hop / conditional)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    R_ranges_by_type: dict[str, list[float]] = defaultdict(list)
    best_weights_by_type: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for qid, rows in by_query.items():
        qtype = qa_types.get(qid, "unknown")
        Rs = np.array([total_reward(r) for r in rows])
        buckets[qtype].append(float(Rs.mean()))
        R_ranges_by_type[qtype].append(float(Rs.max() - Rs.min()))
        best = max(rows, key=total_reward)
        best_weights_by_type[qtype].append((best["a"] / 10, best["b"] / 10, best["g"] / 10))

    out = {}
    for t, vals in buckets.items():
        ranges = R_ranges_by_type[t]
        bws = best_weights_by_type[t]
        out[t] = {
            "n": len(vals),
            "avg_reward": float(np.mean(vals)),
            "R_range_mean": float(np.mean(ranges)),
            "R_range_median": float(np.median(ranges)),
            "mean_best_weight": (
                float(np.mean([w[0] for w in bws])),
                float(np.mean([w[1] for w in bws])),
                float(np.mean([w[2] for w in bws])),
            ),
        }
    return out


def correlation_analysis(by_query: dict[str, list[dict]]) -> dict:
    """Pearson correlation between F1 and Faithfulness across all entries."""
    f1s, faiths = [], []
    for rows in by_query.values():
        for r in rows:
            f1s.append(r["f1"])
            faiths.append(r["faith"])
    f1_arr = np.array(f1s)
    faith_arr = np.array(faiths)
    if f1_arr.std() > 0 and faith_arr.std() > 0:
        corr = float(np.corrcoef(f1_arr, faith_arr)[0, 1])
    else:
        corr = float("nan")
    return {"f1_faith_corr": corr, "f1_std": float(f1_arr.std()), "faith_std": float(faith_arr.std())}


# ---------- rendering ----------

def render_markdown(
    global_stats: dict,
    variance: dict,
    best_weights: dict,
    per_type: dict,
    corr: dict,
) -> str:
    lines = []
    a = lines.append

    a("# Offline Reward Cache — Quality Report")
    a("")
    a("Produced by `scripts/cache_quality_report.py`. Used to gate M6 PPO training.")
    a("")

    a("## 1. Global statistics")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| total_entries | {global_stats['total_entries']:,} |")
    a(f"| n_queries | {global_stats['n_queries']:,} |")
    a(f"| avg F1 | {global_stats['avg_f1']:.4f} |")
    a(f"| avg EM | {global_stats['avg_em']:.4f} |")
    a(f"| avg Faithfulness | {global_stats['avg_faith']:.4f} |")
    a(f"| avg latency (s) | {global_stats['avg_latency']:.4f} |")
    a(f"| avg total R | {global_stats['avg_R']:.4f} |")
    a(f"| std total R (global) | {global_stats['std_R_global']:.4f} |")
    a("")

    a("## 2. Signal variance per query (PPO-readiness)")
    a("")
    a(f"- **flat F1 queries** (std < 1e-6): {variance['flat_f1_queries']:,} / {variance['n_queries']:,} "
      f"= **{100 * variance['flat_f1_ratio']:.1f}%**")
    a(f"- **flat R queries** (std < 1e-6): {variance['flat_R_queries']:,} / {variance['n_queries']:,} "
      f"= **{100 * variance['flat_R_ratio']:.1f}%**")
    a("")
    a("| R-std / R-range | mean | median | p90 | p99 |")
    a("|---|---|---|---|---|")
    a(f"| F1 std per query | {variance['f1_std_mean']:.4f} | {variance['f1_std_median']:.4f} | — | — |")
    a(f"| R std per query | {variance['R_std_mean']:.4f} | {variance['R_std_median']:.4f} | — | — |")
    a(f"| R range per query | {variance['R_range_mean']:.4f} | {variance['R_range_median']:.4f} | "
      f"{variance['R_range_p90']:.4f} | {variance['R_range_p99']:.4f} |")
    a("")
    a("> **Interpretation**: PPO learns gradients from reward variance. ")
    a("> Flat-R ratio > 50% → most queries give no learning signal → PPO may not converge. ")
    a("> Median R-range > 0.05 → usable signal for at least half the queries.")
    a("")

    a("## 3. Best-weight distribution (argmax of total reward per query)")
    a("")
    a(f"- mean best α (Vector): **{best_weights['mean_best_alpha']:.3f}**")
    a(f"- mean best β (Graph):  **{best_weights['mean_best_beta']:.3f}**")
    a(f"- mean best γ (Ontology): **{best_weights['mean_best_gamma']:.3f}**")
    a("")
    a("Top 5 most-common best weights:")
    a("")
    a("| Weight | # queries |")
    a("|---|---|")
    for item in best_weights["top5_best_weights"]:
        a(f"| {item['weight']} | {item['count']} |")
    a("")

    a("## 4. Per-type breakdown")
    a("")
    a("| Type | n | avg R | R-range mean | R-range median | mean best (α, β, γ) |")
    a("|---|---|---|---|---|---|")
    for t, v in per_type.items():
        a(
            f"| {t} | {v['n']:,} | {v['avg_reward']:.4f} | "
            f"{v['R_range_mean']:.4f} | {v['R_range_median']:.4f} | "
            f"({v['mean_best_weight'][0]:.2f}, {v['mean_best_weight'][1]:.2f}, "
            f"{v['mean_best_weight'][2]:.2f}) |"
        )
    a("")
    a("> **Thesis expectation** (Ch.4 Table 4-1, qualitative): ")
    a("> simple queries prefer α-dominant, multi_hop β-dominant, conditional γ-dominant. ")
    a("> If observed pattern matches, the reward signal aligns with prior-knowledge priors.")
    a("")

    a("## 5. Correlation / variance of components")
    a("")
    a(f"- corr(F1, Faithfulness) = **{corr['f1_faith_corr']:.4f}**")
    a(f"- global std(F1) = {corr['f1_std']:.4f}")
    a(f"- global std(Faithfulness) = {corr['faith_std']:.4f}")
    a("")
    a("> High positive correlation → the two components are redundant; low/negative → independent signal.")
    a("")

    a("## 6. PPO-readiness verdict")
    a("")
    # Heuristic: usable if flat ratio < 30% AND median R-range > 0.02
    flat = variance["flat_R_ratio"]
    med = variance["R_range_median"]
    ready = flat < 0.30 and med > 0.02
    a(f"- flat-R ratio: {100 * flat:.1f}% (threshold < 30%)")
    a(f"- R-range median: {med:.4f} (threshold > 0.02)")
    a("")
    if ready:
        a("✅ **PPO should be able to learn.** Proceed to M6 with default config.")
    elif flat > 0.50:
        a("🔴 **High risk**: majority of queries give flat reward. Consider:")
        a("  1. Re-weighting Eq.5-7 (EM=0 kills 0.3 of the signal — reallocate to F1 or Faith).")
        a("  2. Subsetting to non-flat queries for training (filter by R_std > 1e-6).")
        a("  3. Adding a shaping term (e.g., + penalty for extreme weight).")
    else:
        a("🟡 **Borderline**: PPO may converge slowly. Recommend:")
        a("  1. Start with seed 42, monitor mean_reward trajectory for 1K episodes.")
        a("  2. If mean_reward does not improve by 200 episodes, abort and reshape reward.")
    a("")

    return "\n".join(lines)


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    logger.info("Loading cache: %s", args.cache)
    by_query = load_cache_rows(args.cache)
    logger.info("Loaded %d queries × avg %d weights",
                len(by_query), len(next(iter(by_query.values()))) if by_query else 0)

    logger.info("Loading QA types: %s", args.qa)
    qa_types = load_qa_types(args.qa)

    logger.info("Analysis 1/5: global stats...")
    global_stats = summarize_global(by_query)

    logger.info("Analysis 2/5: variance per query...")
    variance = variance_per_query(by_query)

    logger.info("Analysis 3/5: best-weight distribution...")
    best_weights = best_weight_distribution(by_query)

    logger.info("Analysis 4/5: per-type breakdown...")
    per_type = per_type_breakdown(by_query, qa_types)

    logger.info("Analysis 5/5: correlations...")
    corr = correlation_analysis(by_query)

    # Remove the heavy _per_query list before rendering (keep for JSON dump).
    per_query_detail = variance.pop("_per_query")

    md = render_markdown(global_stats, variance, best_weights, per_type, corr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    logger.info("Report written: %s (%d chars)", args.output, len(md))

    # Also dump JSON with full detail for downstream use.
    json_path = args.output.with_suffix(".json")
    payload = {
        "global": global_stats,
        "variance": variance,
        "best_weights": best_weights,
        "per_type": per_type,
        "correlation": corr,
        "n_per_query_detail_rows": len(per_query_detail),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON written: %s", json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
