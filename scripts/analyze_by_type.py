"""
Per-type breakdown analysis for evaluate_rerun JSON outputs.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 6 Section 7 (boundary query claim)

Given one or more rerun JSONs produced by ``scripts/evaluate_rerun.py``,
this script aggregates metrics by query type (simple / multi_hop /
conditional) and writes a comparison markdown.

Usage:
    python scripts/analyze_by_type.py \\
        --input results/rerun_rdwa.json results/rerun_ldwa_seed42.json results/rerun_oracle.json \\
        --names "R-DWA" "L-DWA (ours)" "Oracle" \\
        --output results/per_type_comparison.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TYPES = ("simple", "multi_hop", "conditional")


def mean_std(arr: list[float]) -> tuple[float, float, int]:
    if not arr:
        return 0.0, 0.0, 0
    return float(np.mean(arr)), float(np.std(arr)), len(arr)


def aggregate_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Older format has "aggregate" with pre-computed per-type; full samples
    # have raw per-query metrics. Regenerate from samples for consistency.
    # evaluate_rerun saves the first N samples in "samples"; we rely on
    # "aggregate.by_type" when present.
    agg = data.get("aggregate", {})
    return {
        "policy": data.get("policy", path.stem),
        "n_queries": data.get("n_queries", 0),
        "overall": agg.get("overall", {}),
        "by_type": agg.get("by_type", {}),
    }


def render(agg_list: list[dict], names: list[str]) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Per-type breakdown — F1_strict / F1_substring / Faithfulness")
    a("")
    a(f"Policies: {', '.join(names)}")
    a(f"Queries per policy: {agg_list[0]['n_queries']:,}")
    a("")

    # For each type × metric, show a comparison row
    for t in TYPES:
        a(f"## {t.upper()}")
        a("")
        a("| Policy | F1_strict | F1_substring | EM | Faithfulness |")
        a("|---|---|---|---|---|")
        for agg, name in zip(agg_list, names):
            row = agg["by_type"].get(t, {})
            f1s = row.get("F1_strict", {})
            f1sub = row.get("F1_substring", {})
            em = row.get("EM_norm", {})
            faith = row.get("Faithfulness", {})
            a(
                f"| {name} | "
                f"{f1s.get('mean', 0):.4f} ± {f1s.get('std', 0):.3f} (n={f1s.get('n', 0)}) | "
                f"{f1sub.get('mean', 0):.4f} ± {f1sub.get('std', 0):.3f} | "
                f"{em.get('mean', 0):.4f} | "
                f"{faith.get('mean', 0):.4f} ± {faith.get('std', 0):.3f} |"
            )
        # Compute delta L-DWA vs R-DWA if names line up
        if len(agg_list) >= 2 and "L-DWA" in names[1]:
            rdwa_row = agg_list[0]["by_type"].get(t, {})
            ldwa_row = agg_list[1]["by_type"].get(t, {})
            def delta(m):
                r = rdwa_row.get(m, {}).get("mean", 0)
                l = ldwa_row.get(m, {}).get("mean", 0)
                abs_d = l - r
                rel = (abs_d / r * 100.0) if r > 0 else 0.0
                return abs_d, rel
            f1s_abs, f1s_rel = delta("F1_strict")
            f1sub_abs, f1sub_rel = delta("F1_substring")
            faith_abs, faith_rel = delta("Faithfulness")
            a("")
            a(f"**L-DWA vs R-DWA ({t})**: "
              f"F1_strict {f1s_abs:+.4f} ({f1s_rel:+.1f}%), "
              f"F1_substring {f1sub_abs:+.4f} ({f1sub_rel:+.1f}%), "
              f"Faithfulness {faith_abs:+.4f} ({faith_rel:+.1f}%)")
        a("")

    # Boundary claim section
    a("## Boundary-query implications (thesis Ch.6 §7)")
    a("")
    a("The thesis's boundary-query claim targets queries that combine multi-hop")
    a("reasoning with conditional constraints. In our benchmark, multi_hop and")
    a("conditional are separate categories — the strongest evidence for the")
    a("boundary claim is the L-DWA improvement on **multi_hop** and **conditional**")
    a("relative to their respective R-DWA baselines.")
    a("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if len(args.input) != len(args.names):
        raise SystemExit("--input and --names must have same length")

    aggs = [aggregate_json(p) for p in args.input]
    md = render(aggs, args.names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    logger.info("Wrote %s (%d chars)", args.output, len(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
