"""
Figure 6-5: Cross-domain performance radar.

Reads `results/cross_{benchmark}_{policy}.json` for 3 benchmarks ×
3 policies and plots two panels:
  - Radar chart: F1_strict for each policy on each benchmark
  - Grouped bars: side-by-side 4 metrics × 3 policies for each benchmark

Thesis reference: Ch.6 §7 교차 도메인 실험.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_OUT = ROOT / "docs" / "figures" / "fig6_5_cross_domain_radar.png"

_FONT_PATH = ROOT / "cache" / "malgun.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


BENCHMARKS = ["hotpotqa", "musique", "pubmedqa"]
BENCHMARK_LABELS = {
    "hotpotqa": "HotpotQA\nHard 300",
    "musique": "MuSiQue\nDev 300",
    "pubmedqa": "PubMedQA\nPharma 300",
}

# Policy → filename stub
POLICIES = [
    ("Vector-only", "vector-only", "#4F86C6"),
    ("R-DWA", "rdwa", "#8FB573"),
    ("L-DWA (univ-trained)", "ldwa_cache_ppo_checkpoints_seed_42_final.pt", "#E8756E"),
]


def load_overall(bench: str, policy_stub: str) -> dict | None:
    p = RESULTS / f"cross_{bench}_{policy_stub}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["aggregate"]["overall"]


def main() -> int:
    # Collect
    grid: dict[str, dict[str, dict]] = {}  # policy -> bench -> metrics
    for name, stub, _ in POLICIES:
        grid[name] = {}
        for b in BENCHMARKS:
            o = load_overall(b, stub)
            if o is None:
                grid[name][b] = {"F1_strict": 0.0, "F1_sub": 0.0, "Faith": 0.0, "EM": 0.0}
            else:
                grid[name][b] = {
                    "F1_strict": o["F1_strict"]["mean"],
                    "F1_sub": o["F1_substring"]["mean"],
                    "Faith": o["Faithfulness"]["mean"],
                    "EM": o["EM_norm"]["mean"],
                }

    fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])

    # Left — radar
    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(BENCHMARKS), endpoint=False).tolist()
    angles += angles[:1]

    for name, _, color in POLICIES:
        vals = [grid[name][b]["F1_strict"] for b in BENCHMARKS]
        vals += vals[:1]
        ax1.plot(angles, vals, marker="o", linewidth=2, color=color, label=name)
        ax1.fill(angles, vals, alpha=0.15, color=color)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([BENCHMARK_LABELS[b] for b in BENCHMARKS], fontsize=10)
    ax1.set_ylim(0, 0.25)
    ax1.set_title("F1$_{strict}$ across benchmarks", fontsize=11, pad=20)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)
    ax1.grid(alpha=0.4)

    # Right — grouped bars (4 metrics × 3 policies per benchmark)
    ax2 = fig.add_subplot(gs[0, 1])

    metrics = ["F1_strict", "F1_sub", "Faith"]
    n_bench = len(BENCHMARKS)
    n_pol = len(POLICIES)
    width = 0.12

    offsets = np.linspace(-(n_pol - 1) / 2, (n_pol - 1) / 2, n_pol) * width * 1.2
    x_groups = np.arange(n_bench)
    # For each metric, separate blocks offset by metric
    metric_offset = np.linspace(
        -(len(metrics) - 1) / 2, (len(metrics) - 1) / 2, len(metrics)
    ) * width * n_pol * 1.5

    bars_info = []
    for mi, metric in enumerate(metrics):
        for pi, (name, _, color) in enumerate(POLICIES):
            vals = [grid[name][b][metric] for b in BENCHMARKS]
            positions = x_groups + metric_offset[mi] + offsets[pi]
            bars_info.append((positions, vals, name, metric, color))

    # Simpler: one metric per subplot → 3 subplots
    # Redraw with clearer layout
    ax2.clear()
    x_base = np.arange(n_bench)
    group_width = 0.8
    sub_width = group_width / n_pol

    for pi, (name, _, color) in enumerate(POLICIES):
        vals = [grid[name][b]["F1_strict"] for b in BENCHMARKS]
        positions = x_base + (pi - (n_pol - 1) / 2) * sub_width
        bars = ax2.bar(positions, vals, sub_width * 0.92, label=name,
                       color=color, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x_base)
    ax2.set_xticklabels([BENCHMARK_LABELS[b] for b in BENCHMARKS], fontsize=10)
    ax2.set_ylabel("F1$_{strict}$ (mean)")
    ax2.set_title("Detail — F1$_{strict}$ per benchmark × policy", fontsize=11)
    ax2.set_ylim(0, max(
        max(grid[name][b]["F1_strict"] for b in BENCHMARKS)
        for name, _, _ in POLICIES
    ) * 1.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Figure 6-5. Cross-domain generalization — univ-trained L-DWA on English benchmarks",
        fontsize=12,
    )
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
