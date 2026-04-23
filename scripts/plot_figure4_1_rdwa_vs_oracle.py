"""
Figure 4-1: R-DWA default weights vs Oracle actual weights on Δ³.

Ch.4 에서 R-DWA 의 한계 (기본 가중치 테이블이 실제 보상 지형과 어긋남)
를 시각화. 세 개의 기본 가중치 포인트 (simple→α, multi_hop→β,
conditional→γ) 가 Oracle 의 평균 가중치 (γ-지배적) 와 떨어져 있음을
삼각형 simplex 위에서 보여준다.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
FIG_OUT = ROOT / "docs" / "figures" / "fig4_1_rdwa_vs_oracle.png"

_FONT_PATH = ROOT / "cache" / "malgun.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def bary_to_xy(alpha: float, beta: float, gamma: float) -> tuple[float, float]:
    """Barycentric to Cartesian (equilateral triangle)."""
    x = beta + 0.5 * gamma
    y = (np.sqrt(3) / 2) * gamma
    return x, y


def draw_triangle(ax):
    vx = [0, 1, 0.5, 0]
    vy = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(vx, vy, color="black", linewidth=1.5)
    ax.text(-0.04, -0.05, "Vector (α=1)", ha="right", va="top", fontsize=11)
    ax.text(1.04, -0.05, "Graph (β=1)", ha="left", va="top", fontsize=11)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "Ontology (γ=1)", ha="center", va="bottom", fontsize=11)
    # light grid
    for t in [0.25, 0.5, 0.75]:
        # iso-α lines (parallel to β-γ edge)
        p1 = bary_to_xy(t, 1 - t, 0)
        p2 = bary_to_xy(t, 0, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ddd", linewidth=0.6, zorder=0)


def main() -> int:
    fig, ax = plt.subplots(figsize=(11, 9))
    draw_triangle(ax)

    # R-DWA base points (Table 4-1)
    rdwa_points = [
        ("simple\n(0.6, 0.2, 0.2)", 0.6, 0.2, 0.2, "#4F86C6"),
        ("multi_hop\n(0.2, 0.6, 0.2)", 0.2, 0.6, 0.2, "#5B9B5E"),
        ("conditional\n(0.2, 0.2, 0.6)", 0.2, 0.2, 0.6, "#C07020"),
    ]
    # Oracle mean (Ch.6 §3 analysis)
    oracle_alpha, oracle_beta, oracle_gamma = 0.11, 0.16, 0.73
    # R-DWA actual mean selected (Ch.6 §5.3)
    rdwa_mean_alpha, rdwa_mean_beta, rdwa_mean_gamma = 0.25, 0.45, 0.30
    # L-DWA mean (for reference, seed 42)
    ldwa_alpha, ldwa_beta, ldwa_gamma = 0.25, 0.30, 0.45

    for label, a, b, g, color in rdwa_points:
        x, y = bary_to_xy(a, b, g)
        ax.scatter([x], [y], s=280, color=color, edgecolor="black", linewidths=1.5,
                   marker="o", zorder=5, label=f"R-DWA {label.split(chr(10))[0]}")
        ax.annotate(label, xy=(x, y), xytext=(10, 10), textcoords="offset points",
                    fontsize=9.5)

    # Oracle mean
    x, y = bary_to_xy(oracle_alpha, oracle_beta, oracle_gamma)
    ax.scatter([x], [y], s=380, color="#C00000", edgecolor="black",
               linewidths=1.8, marker="*", zorder=6,
               label=f"Oracle 평균\n(α={oracle_alpha}, β={oracle_beta}, γ={oracle_gamma})")
    ax.annotate(f"Oracle 평균\n(0.11, 0.16, 0.73)",
                xy=(x, y), xytext=(15, -25), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#C00000")

    # R-DWA actual weighted mean
    x, y = bary_to_xy(rdwa_mean_alpha, rdwa_mean_beta, rdwa_mean_gamma)
    ax.scatter([x], [y], s=280, color="#444", edgecolor="black", linewidths=1.5,
               marker="s", zorder=5)
    ax.annotate(f"R-DWA 실제 평균\n(0.25, 0.45, 0.30)",
                xy=(x, y), xytext=(-90, 20), textcoords="offset points",
                fontsize=10, color="#333")

    # L-DWA mean for reference
    x, y = bary_to_xy(ldwa_alpha, ldwa_beta, ldwa_gamma)
    ax.scatter([x], [y], s=320, color="#5B9B5E", edgecolor="black", linewidths=1.5,
               marker="D", zorder=6)
    ax.annotate(f"L-DWA 평균\n(0.25, 0.30, 0.45)",
                xy=(x, y), xytext=(15, 15), textcoords="offset points",
                fontsize=10, color="#3A6B3D", fontweight="bold")

    # Mark the gap as a dashed arrow from R-DWA centroid to Oracle
    rdwa_cx, rdwa_cy = bary_to_xy(rdwa_mean_alpha, rdwa_mean_beta, rdwa_mean_gamma)
    ora_x, ora_y = bary_to_xy(oracle_alpha, oracle_beta, oracle_gamma)
    ax.annotate("", xy=(ora_x, ora_y), xytext=(rdwa_cx, rdwa_cy),
                arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.8,
                                connectionstyle="arc3,rad=0.2", alpha=0.6))
    ax.text(0.3, 0.48, "실제 보상 지형과의 괴리",
            fontsize=10, color="#C00000", style="italic")

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Figure 4-1. R-DWA 규칙 기반 점 vs Oracle 평균 (Δ³ simplex)\n"
        "— R-DWA의 대각선 배치가 실제 보상 지형(Ontology 지배적) 과 정렬되지 않음",
        fontsize=12, pad=15,
    )

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
