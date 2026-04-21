"""
Figure 6-2: Per-type F1_strict comparison (R-DWA / L-DWA / Oracle).

Reads results/rerun_rdwa_list.json, rerun_ldwa_seed42_list.json,
rerun_oracle_list.json, and produces a grouped bar chart grouping by
query type (simple / multi_hop / conditional) × policy.

Thesis reference: Ch.6 §4 질의 유형별 세분 분석.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_OUT = ROOT / "docs" / "figures" / "fig6_2_per_type.png"

POLICIES = [
    ("R-DWA", "rerun_rdwa_list.json", "#4F86C6"),
    ("L-DWA (seed 42)", "rerun_ldwa_seed42_list.json", "#E8756E"),
    ("Oracle", "rerun_oracle_list.json", "#8FB573"),
]

TYPES = ["simple", "multi_hop", "conditional"]
TYPE_LABELS = {"simple": "Simple\n(n=2,000)", "multi_hop": "Multi-hop\n(n=1,750)", "conditional": "Conditional\n(n=1,250)"}


def load_per_type(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)["aggregate"]["by_type"]


def main() -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), sharey=False, constrained_layout=True
    )

    # Left panel — F1_strict by type × policy
    x = np.arange(len(TYPES))
    width = 0.25
    for i, (name, fname, color) in enumerate(POLICIES):
        data = load_per_type(RESULTS / fname)
        vals = [data[t]["F1_strict"]["mean"] for t in TYPES]
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, vals, width, label=name, color=color,
                       edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([TYPE_LABELS[t] for t in TYPES], fontsize=10)
    ax1.set_ylabel("F1$_{strict}$ (mean)", fontsize=10)
    ax1.set_title("F1$_{strict}$ by query type (list prompt)", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9)

    # Right panel — L-DWA Δ over R-DWA (%)
    delta_pct = []
    for t in TYPES:
        rdwa = load_per_type(RESULTS / "rerun_rdwa_list.json")[t]["F1_strict"]["mean"]
        ldwa = load_per_type(RESULTS / "rerun_ldwa_seed42_list.json")[t]["F1_strict"]["mean"]
        delta_pct.append((ldwa - rdwa) / rdwa * 100 if rdwa > 0 else 0.0)
    colors = ["#5B9BD5", "#5B9BD5", "#C00000"]  # highlight conditional in red
    bars = ax2.bar([TYPE_LABELS[t] for t in TYPES], delta_pct,
                   color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, delta_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                 f"{v:+.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax2.set_ylabel("L-DWA Δ over R-DWA (%)", fontsize=10)
    ax2.set_title("Relative improvement per type (F1$_{strict}$)", fontsize=11)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylim(0, max(delta_pct) * 1.2 if delta_pct else 50)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Figure 6-2. Per-type F1$_{strict}$ on 5,000 QA (list prompt)",
        fontsize=12,
    )
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
