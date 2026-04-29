"""
Paired bootstrap (BCa) of policy A vs policy B over per-query metrics.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>

Why this script exists
----------------------
Headline tables in 통합본_v19 report mean ± std but no significance test, so
a thesis committee cannot tell whether the L-DWA → R-DWA gap (or the
L-DWA → Oracle gap) is statistical noise. We compute paired bootstrap CIs
on the within-query difference d_i = m_A(q_i) − m_B(q_i), reporting:
    * mean Δ
    * 95% CI (BCa, 5,000 resamples)
    * two-sided bootstrap p-value (proportion of resampled means crossing 0)
    * Cohen's d_z (paired)

This consumes ``per_query`` blocks from ``evaluate_on_cache.py`` outputs
(after the 2026-04-29 patch that always emits ``per_query``). The cache
evaluator uses the *same* (possibly buggy) reward function across all
policies, so within-policy ranking remains valid even though absolute
F1 numbers differ from the corrected ``rerun_*_list.json`` headline.

Usage
-----
    python scripts/paired_bootstrap.py \\
        --policies \\
            rdwa=results/eval_rdwa.json \\
            uniform=results/eval_uniform.json \\
            ldwa_seed42=results/eval_ldwa_seed42_cache.json \\
            ldwa_seed123=results/eval_ldwa_seed123_cache.json \\
            ldwa_seed999=results/eval_ldwa_seed999_cache.json \\
            oracle=results/eval_oracle.json \\
        --baseline rdwa \\
        --output results/paired_bootstrap.md \\
        --metric f1

The ``--baseline`` policy is the LHS in every comparison "policy − baseline".
For thesis defense we typically run twice:
    1) --baseline rdwa     (Q: does L-DWA beat the rule baseline?)
    2) --baseline oracle   (Q: does L-DWA close / cross the Oracle bound?)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("paired_bootstrap")

METRIC_KEY = {
    "f1": "f1",
    "em": "em",
    "faith": "faith",
    "faithfulness": "faith",
    "r": "R",
    "reward": "R",
    "latency": "latency",
}


# ---------- io ----------

def load_per_query(path: Path) -> dict[str, dict]:
    """Return ``{qid: row}`` from an evaluate_on_cache.py JSON dump."""
    with path.open(encoding="utf-8") as f:
        d = json.load(f)
    if "per_query" not in d:
        raise SystemExit(
            f"{path} has no 'per_query' field — re-run evaluate_on_cache.py "
            f"after the 2026-04-29 patch."
        )
    return {row["qid"]: row for row in d["per_query"] if not row.get("miss")}


def aligned_pairs(a: dict[str, dict], b: dict[str, dict], metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(arr_a, arr_b, qids)`` aligned on shared qids."""
    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        raise SystemExit("no common qids between the two policies")
    key = METRIC_KEY[metric.lower()]
    arr_a = np.array([a[q][key] for q in common], dtype=float)
    arr_b = np.array([b[q][key] for q in common], dtype=float)
    return arr_a, arr_b, common


# ---------- bootstrap ----------

def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 42, alpha: float = 0.05,
) -> dict:
    """
    Paired bootstrap of mean(a − b) with BCa-style 95% CI.

    BCa correction handles bias and skewness in the bootstrap distribution.
    Implementation follows Efron & Tibshirani (1993) §14.3.
    """
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    mean_d = float(np.mean(d))

    # Bootstrap resamples
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    boot_means_sorted = np.sort(boot_means)

    # BCa correction
    # bias-correction z0
    p_below = float((boot_means < mean_d).mean())
    if p_below in (0.0, 1.0):
        z0 = 0.0
    else:
        z0 = float(_norm_ppf(p_below))

    # Acceleration via jackknife
    jack = np.empty(n)
    for i in range(n):
        jack[i] = (np.sum(d) - d[i]) / (n - 1)
    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    accel = num / den if den != 0 else 0.0

    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)
    a1 = _norm_cdf(z0 + (z0 + z_lo) / (1 - accel * (z0 + z_lo)))
    a2 = _norm_cdf(z0 + (z0 + z_hi) / (1 - accel * (z0 + z_hi)))
    a1 = float(np.clip(a1, 0.001, 0.999))
    a2 = float(np.clip(a2, 0.001, 0.999))

    ci_lo = float(np.quantile(boot_means_sorted, a1))
    ci_hi = float(np.quantile(boot_means_sorted, a2))

    # Two-sided bootstrap p-value: 2 × min(P(boot ≥ 0), P(boot ≤ 0))
    p_ge = float((boot_means >= 0).mean())
    p_le = float((boot_means <= 0).mean())
    p_two = float(min(1.0, 2.0 * min(p_ge, p_le)))

    # Cohen's d_z = mean(d) / std(d)
    sd = float(np.std(d, ddof=1))
    cohens_dz = mean_d / sd if sd > 0 else float("inf") if mean_d != 0 else 0.0

    return {
        "n": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "delta": mean_d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_two_sided": p_two,
        "cohens_dz": cohens_dz,
        "n_boot": n_boot,
    }


def _norm_cdf(z: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (Acklam 2003 rational approximation)."""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ---------- per-type ----------

def per_type_bootstrap(
    a: dict[str, dict], b: dict[str, dict], metric: str, n_boot: int, seed: int,
) -> dict[str, dict]:
    """Bootstrap each query type separately."""
    common = sorted(set(a.keys()) & set(b.keys()))
    types = sorted({a[q]["type"] for q in common})
    out: dict[str, dict] = {}
    key = METRIC_KEY[metric.lower()]
    for t in types:
        qids = [q for q in common if a[q]["type"] == t]
        if not qids:
            continue
        arr_a = np.array([a[q][key] for q in qids], dtype=float)
        arr_b = np.array([b[q][key] for q in qids], dtype=float)
        out[t] = paired_bootstrap_diff(arr_a, arr_b, n_boot=n_boot, seed=seed)
    return out


# ---------- rendering ----------

def _format_p(p: float) -> str:
    if p < 1e-4:
        return "< 1e-4"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _format_d(dz: float) -> str:
    if math.isinf(dz):
        return "∞"
    return f"{dz:+.3f}"


def render_markdown(
    metric: str, baseline: str, comparisons: list[tuple[str, dict, dict]],
) -> str:
    """Build a thesis-ready Markdown report."""
    lines: list[str] = []
    a = lines.append

    a(f"# Paired bootstrap — {metric.upper()} ({baseline} as baseline)")
    a("")
    a("- Method: 5,000 paired resamples, BCa-corrected 95% CI, two-sided bootstrap p-value")
    a("- Sample: query-level metric from cache eval (consistent reward function across policies)")
    a("- Effect size: Cohen's d_z (paired); |0.2|=small, |0.5|=medium, |0.8|=large")
    a("- Δ = (policy) − (baseline). Positive Δ ⇒ policy beats baseline.")
    a("")
    a("## Overall (5,000 QA)")
    a("")
    a(f"| policy | n | mean (policy) | mean ({baseline}) | Δ | 95% CI (BCa) | p (2-sided) | Cohen's d_z | verdict |")
    a("|---|---|---|---|---|---|---|---|---|")
    for name, overall, _ in comparisons:
        verdict = _verdict(overall)
        a(
            f"| **{name}** | {overall['n']:,} | "
            f"{overall['mean_a']:.4f} | {overall['mean_b']:.4f} | "
            f"{overall['delta']:+.4f} | "
            f"[{overall['ci_lo']:+.4f}, {overall['ci_hi']:+.4f}] | "
            f"{_format_p(overall['p_two_sided'])} | "
            f"{_format_d(overall['cohens_dz'])} | {verdict} |"
        )

    a("")
    a("## Per-type breakdown")
    a("")
    types = set()
    for _, _, by_type in comparisons:
        types |= set(by_type.keys())
    for t in sorted(types):
        a(f"### type = `{t}`")
        a("")
        a(f"| policy | n | Δ | 95% CI | p | d_z |")
        a("|---|---|---|---|---|---|")
        for name, _, by_type in comparisons:
            if t not in by_type:
                continue
            r = by_type[t]
            a(
                f"| {name} | {r['n']} | {r['delta']:+.4f} | "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                f"{_format_p(r['p_two_sided'])} | {_format_d(r['cohens_dz'])} |"
            )
        a("")
    return "\n".join(lines)


def _verdict(r: dict) -> str:
    if r["p_two_sided"] >= 0.05:
        return "n.s."
    if r["delta"] > 0:
        if r["ci_lo"] > 0:
            return "✅ better"
        return "≈ tie (CI crosses 0)"
    if r["ci_hi"] < 0:
        return "❌ worse"
    return "≈ tie (CI crosses 0)"


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policies", nargs="+", required=True,
        help="Pairs name=path/to/eval_*.json (must contain per_query)",
    )
    parser.add_argument("--baseline", required=True, help="Name of baseline policy (LHS)")
    parser.add_argument("--metric", default="f1", choices=list(METRIC_KEY.keys()))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    # Parse "name=path"
    pol: dict[str, Path] = {}
    for spec in args.policies:
        if "=" not in spec:
            raise SystemExit(f"--policies must be name=path, got: {spec}")
        name, p = spec.split("=", 1)
        pol[name] = Path(p)

    if args.baseline not in pol:
        raise SystemExit(f"--baseline {args.baseline!r} not in --policies names {list(pol)}")

    logger.info("Loading per-query data...")
    data: dict[str, dict[str, dict]] = {n: load_per_query(p) for n, p in pol.items()}
    logger.info("Loaded: " + ", ".join(f"{n}={len(d):,}" for n, d in data.items()))

    base = data[args.baseline]
    comparisons: list[tuple[str, dict, dict]] = []
    for name, rows in data.items():
        if name == args.baseline:
            continue
        a, b, _ = aligned_pairs(rows, base, args.metric)
        overall = paired_bootstrap_diff(a, b, n_boot=args.n_boot, seed=args.seed)
        by_type = per_type_bootstrap(rows, base, args.metric, args.n_boot, args.seed)
        comparisons.append((name, overall, by_type))
        logger.info(
            "%s vs %s: Δ=%+.4f CI=[%+.4f, %+.4f] p=%s d_z=%s",
            name, args.baseline, overall["delta"], overall["ci_lo"], overall["ci_hi"],
            _format_p(overall["p_two_sided"]), _format_d(overall["cohens_dz"]),
        )

    md = render_markdown(args.metric, args.baseline, comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")

    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "metric": args.metric,
                "baseline": args.baseline,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "comparisons": [
                    {"name": n, "overall": o, "by_type": t} for n, o, t in comparisons
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", args.output, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
