"""
Figure 6-3: L-DWA vs R-DWA vs Oracle weight distribution over 5,000 QA.

Reads `top_weights` from the aggregate section of each policy's rerun
JSON and produces a scatter plot on the 2-simplex (barycentric) showing
the most-frequent weight combinations each policy selected.

Thesis reference: Ch.6 §5.3 정책 선택 가중치 분포.
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
FIG_OUT = ROOT / "docs" / "figures" / "fig6_3_weight_distribution.png"

_FONT_PATH = ROOT / "cache" / "malgun.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


POLICIES = [
    ("R-DWA", "rerun_rdwa_list.json", "#4F86C6"),
    ("L-DWA (seed 42)", "rerun_ldwa_seed42_list.json", "#E8756E"),
    ("Oracle", "rerun_oracle_list.json", "#8FB573"),
]


def simplex_to_xy(alpha: float, beta: float, gamma: float) -> tuple[float, float]:
    """Barycentric to Cartesian for an equilateral triangle with
    vertices Vector=(0,0), Graph=(1,0), Ontology=(0.5, sqrt(3)/2)."""
    x = beta + 0.5 * gamma
    y = (np.sqrt(3) / 2) * gamma
    return x, y


def parse_weight_str(s: str) -> tuple[float, float, float]:
    # "(α=0.2, β=0.6, γ=0.2)" → (0.2, 0.6, 0.2)
    inner = s.strip("()")
    parts = [p.strip() for p in inner.split(",")]
    out: list[float] = []
    for p in parts:
        v = p.split("=")[-1]
        out.append(float(v))
    return (out[0], out[1], out[2])


def extract_per_qid_weights(path: Path) -> list[tuple[float, float, float]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    # sample-level weights (first 50 saved)
    triples: list[tuple[float, float, float]] = []
    for s in data.get("samples", []):
        w = s.get("weights")
        if w and len(w) == 3:
            # weights stored as ints 0..10
            triples.append((w[0] / 10, w[1] / 10, w[2] / 10))
    # also use top_weights aggregate
    for tw in data["aggregate"].get("top_weights", []):
        try:
            a, b, g = parse_weight_str(tw["weight"])
            # replicate by count for density visualization, capped
            n = min(tw["count"], 500)
            triples.extend([(a, b, g)] * n)
        except Exception:
            continue
    return triples


def main() -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 5), constrained_layout=True
    )

    for ax, (name, fname, color) in zip(axes, POLICIES):
        triples = extract_per_qid_weights(RESULTS / fname)
        if not triples:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        xs, ys = zip(*[simplex_to_xy(*t) for t in triples])

        # Triangle frame
        vx = [0, 1, 0.5, 0]
        vy = [0, 0, np.sqrt(3) / 2, 0]
        ax.plot(vx, vy, color="black", linewidth=1.2)

        # vertex labels
        ax.text(-0.04, -0.04, "Vector (α=1)", ha="right", va="top", fontsize=9)
        ax.text(1.04, -0.04, "Graph (β=1)", ha="left", va="top", fontsize=9)
        ax.text(0.5, np.sqrt(3) / 2 + 0.03, "Ontology (γ=1)",
                ha="center", va="bottom", fontsize=9)

        # Scatter with jitter
        rng = np.random.default_rng(42)
        xs_j = np.asarray(xs) + rng.normal(0, 0.01, size=len(xs))
        ys_j = np.asarray(ys) + rng.normal(0, 0.01, size=len(ys))
        ax.scatter(xs_j, ys_j, color=color, alpha=0.22, s=18,
                   edgecolor="black", linewidths=0.3)

        # Mean point (bigger, solid)
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        ax.scatter([mean_x], [mean_y], color=color, s=250,
                   edgecolor="black", linewidths=1.5, marker="*",
                   label="mean", zorder=10)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{name}\n(n = {len(triples):,} weight selections)",
                     fontsize=11)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Figure 6-3. DWA weight distribution on Δ³ simplex (5,000 QA, list prompt)",
        fontsize=12,
    )
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
