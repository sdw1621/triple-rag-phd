"""
Generate Figure 6-1: Sentence vs List prompt delta bar chart.

Reads the six rerun files (sentence+list × R-DWA/Oracle/L-DWA seed 42)
and produces docs/figures/fig6_1_prompt_delta.png. Matplotlib only —
no seaborn dependency.

Thesis reference: Ch.6 §3.4, figure caption template at bottom.
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
FIG_OUT = ROOT / "docs" / "figures" / "fig6_1_prompt_delta.png"

# (policy display name, sentence file, list file)
POLICIES = [
    ("R-DWA", "rerun_rdwa_fixed.json", "rerun_rdwa_list.json"),
    ("Oracle", "rerun_oracle_fixed.json", "rerun_oracle_list.json"),
    ("L-DWA (seed 42)", "rerun_ldwa_seed42_fixed.json", "rerun_ldwa_seed42_list.json"),
]

# Drop F1_char — pre-existing sentence-run JSONs (from dff7dc1) predate the
# f1_char metric, so the "sentence" side would be 0 (not measured, not
# actually zero). Would mislead the reader. F1_char delta is presented in
# the text (Ch.6 Table 6-2 footnote) instead.
METRICS = [
    ("F1_strict", "F1$_{strict}$"),
    ("F1_substring", "F1$_{sub}$"),
    ("EM_norm", "EM"),
    ("Faithfulness", "Faith"),
]


def load_overall(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)["aggregate"]["overall"]


def main() -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, len(POLICIES), figsize=(14, 5), sharey=True, constrained_layout=True
    )

    x = np.arange(len(METRICS))
    width = 0.38

    for ax, (name, sent_file, list_file) in zip(axes, POLICIES):
        sent = load_overall(RESULTS / sent_file)
        lst = load_overall(RESULTS / list_file)
        sent_vals = [sent.get(m, {"mean": 0.0})["mean"] for m, _ in METRICS]
        list_vals = [lst.get(m, {"mean": 0.0})["mean"] for m, _ in METRICS]

        b1 = ax.bar(x - width / 2, sent_vals, width, label="sentence prompt",
                    color="#4F86C6", edgecolor="black", linewidth=0.5)
        b2 = ax.bar(x + width / 2, list_vals, width, label="list prompt",
                    color="#E8756E", edgecolor="black", linewidth=0.5)

        for bar, val in zip(b1, sent_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(b2, list_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in METRICS], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ax is axes[0]:
            ax.set_ylabel("Score (mean, 5,000 QA)")
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Figure 6-1. Prompt-style effect on evaluation metrics "
        "(sentence vs list, 5,000 QA)",
        fontsize=12,
    )

    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
