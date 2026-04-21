"""
Figure 5-1: Actor-Critic network architecture.

Matplotlib box-and-arrow schematic showing:
  s (18-dim) → Linear 18→64 → Tanh → Linear 64→64 → Tanh → features (64)
                                                              ├─ Actor: Linear 64→3 → Softplus+ε → Dirichlet c
                                                              └─ Critic: Linear 64→1 → V(s)

Thesis reference: Ch.5 §2.1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_OUT = ROOT / "docs" / "figures" / "fig5_1_actor_critic.png"

_FONT_PATH = ROOT / "cache" / "malgun.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def box(ax, xy, w, h, text, facecolor="#DDECF5", fontsize=10, bold=False):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, facecolor=facecolor, edgecolor="black",
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight)


def arrow(ax, start, end, color="#333", linewidth=1.5):
    a = FancyArrowPatch(
        start, end,
        arrowstyle="->", mutation_scale=14, linewidth=linewidth, color=color,
    )
    ax.add_patch(a)


def main() -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Input state (with 18-dim breakdown)
    box(ax, (0.3, 5.5), 2.6, 1.5,
        "State s\n(18-dim)\n\ndensity·intent·\nsrc_stats·meta",
        facecolor="#F9E5D0", bold=True, fontsize=9)

    # Shared backbone
    box(ax, (3.3, 6.0), 1.6, 0.8, "Linear\n18 → 64", facecolor="#DDECF5", fontsize=9)
    box(ax, (5.1, 6.0), 1.0, 0.8, "Tanh", facecolor="#E8EFF5", fontsize=9)
    box(ax, (6.3, 6.0), 1.6, 0.8, "Linear\n64 → 64", facecolor="#DDECF5", fontsize=9)
    box(ax, (8.1, 6.0), 1.0, 0.8, "Tanh", facecolor="#E8EFF5", fontsize=9)
    box(ax, (9.3, 6.0), 1.4, 0.8, "features\n(64)", facecolor="#FFE9A8",
        fontsize=9, bold=True)

    # Arrows along backbone
    arrow(ax, (2.9, 6.4), (3.3, 6.4))
    arrow(ax, (4.9, 6.4), (5.1, 6.4))
    arrow(ax, (6.1, 6.4), (6.3, 6.4))
    arrow(ax, (7.9, 6.4), (8.1, 6.4))
    arrow(ax, (9.1, 6.4), (9.3, 6.4))

    # Actor head
    box(ax, (10.9, 6.9), 2.5, 0.8, "Actor head\nLinear 64 → 3",
        facecolor="#FFD3CF", fontsize=9)
    box(ax, (10.9, 6.0), 2.5, 0.8, "Softplus + ε",
        facecolor="#FFE6E3", fontsize=9)
    arrow(ax, (10.7, 7.0), (10.9, 7.3))
    arrow(ax, (12.15, 6.9), (12.15, 6.8))

    # Critic head
    box(ax, (10.9, 4.7), 2.5, 0.8, "Critic head\nLinear 64 → 1",
        facecolor="#D4E7D6", fontsize=9)
    arrow(ax, (10.7, 6.0), (10.9, 5.1))

    # Outputs
    box(ax, (3.3, 2.8), 3.2, 1.2,
        "Dirichlet concentrations\nc = (c_α, c_β, c_γ),  c > 0",
        facecolor="#FFD3CF", fontsize=10, bold=True)
    arrow(ax, (12.15, 6.0), (6.5, 3.4), color="#C04040", linewidth=1.2)

    box(ax, (9.0, 2.8), 3.0, 1.2,
        "State value V(s)\n(scalar)",
        facecolor="#D4E7D6", fontsize=10, bold=True)
    arrow(ax, (12.15, 4.7), (10.5, 4.0), color="#3E8B4A", linewidth=1.2)

    # Dirichlet sampling → simplex
    box(ax, (3.3, 0.5), 3.2, 1.5,
        "Dirichlet(c)\n↓\na = (α, β, γ) ∈ Δ³\n(sample or mean)",
        facecolor="#F9E5D0", fontsize=9, bold=True)
    arrow(ax, (4.9, 2.8), (4.9, 2.0))

    # Ortho init annotation
    box(ax, (3.3, 5.0), 7.4, 0.5,
        "Shared backbone (orthogonal init, gain=1.0)",
        facecolor="#F5F5F5", fontsize=8)
    arrow(ax, (4.9, 5.5), (4.9, 5.45), color="#999")

    # Parameter count annotation
    ax.text(13.7, 3.4, "~ 6,081 params",
            ha="right", fontsize=10, color="#444", style="italic")

    ax.set_title(
        "Figure 5-1. Actor-Critic network architecture "
        "(shared backbone + actor head + critic head)",
        fontsize=11, pad=14,
    )

    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
