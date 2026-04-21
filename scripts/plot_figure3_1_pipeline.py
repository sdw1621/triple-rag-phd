"""
Figure 3-1: Triple-Hybrid RAG pipeline diagram.

Matplotlib box-and-arrow diagram of the 7-stage pipeline described in
Ch.3 §1. Labels match the prose: QueryAnalyzer, DWA (R-DWA | L-DWA),
VectorStore + GraphStore + OntologyStore, merge_contexts, LLM.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_OUT = ROOT / "docs" / "figures" / "fig3_1_pipeline.png"

# Load Korean font (Windows Malgun Gothic copied into cache/ for container use).
_FONT_PATH = ROOT / "cache" / "malgun.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def box(ax, xy, w, h, text, facecolor="#DDECF5", edgecolor="black",
        fontsize=10, bold=False):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, facecolor=facecolor, edgecolor=edgecolor,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True)


def arrow(ax, start, end, color="#333", style="->", linewidth=1.5):
    a = FancyArrowPatch(
        start, end,
        arrowstyle=style, mutation_scale=14, linewidth=linewidth,
        color=color,
    )
    ax.add_patch(a)


def main() -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Row 1 — input + intent
    box(ax, (0.3, 7.3), 2.2, 1.0, "Query q\n(자연어 질의)", facecolor="#F9E5D0", bold=True)
    box(ax, (3.2, 7.3), 3.0, 1.0, "QueryAnalyzer\n(intent + density c_e,c_r,c_c)",
        facecolor="#DDECF5")
    arrow(ax, (2.5, 7.8), (3.2, 7.8))

    # Row 2 — DWA (two variants highlighted)
    box(ax, (6.9, 7.3), 3.0, 1.0, "DWA → (α, β, γ) ∈ Δ³",
        facecolor="#FFE9A8", bold=True)
    arrow(ax, (6.2, 7.8), (6.9, 7.8))
    box(ax, (6.9, 5.9), 1.45, 0.8, "R-DWA\n(규칙 기반, Ch.Ⅳ)", facecolor="#F3F3F3", fontsize=9)
    box(ax, (8.45, 5.9), 1.45, 0.8, "L-DWA\n(PPO 학습, Ch.Ⅴ)", facecolor="#FFD3CF", fontsize=9, bold=True)
    arrow(ax, (8.4, 6.7), (8.4, 7.3), color="#999")
    arrow(ax, (7.6, 6.7), (7.6, 7.3), color="#999")

    # Row 3 — three sources
    src_y = 4.1
    box(ax, (0.3, src_y), 3.0, 1.0, "VectorStore\nFAISS · text-embedding-3-small\nweight: α",
        facecolor="#E0F0E8", fontsize=9)
    box(ax, (3.7, src_y), 3.0, 1.0, "GraphStore\nNetworkX BFS 3-hop\nweight: β",
        facecolor="#E0F0E8", fontsize=9)
    box(ax, (7.1, src_y), 3.0, 1.0, "OntologyStore\nOwlready2 · HermiT\nweight: γ",
        facecolor="#E0F0E8", fontsize=9)
    # arrows from DWA to sources
    arrow(ax, (5.5, 7.3), (1.8, src_y + 1.0), color="#888", linewidth=1.0)
    arrow(ax, (7.5, 7.3), (5.2, src_y + 1.0), color="#888", linewidth=1.0)
    arrow(ax, (9.5, 7.3), (8.6, src_y + 1.0), color="#888", linewidth=1.0)
    # arrows from query to sources
    arrow(ax, (1.4, 7.3), (1.8, src_y + 1.0), color="#555", linewidth=1.0)
    arrow(ax, (1.4, 7.3), (5.2, src_y + 1.0), color="#555", linewidth=1.0)
    arrow(ax, (1.4, 7.3), (8.6, src_y + 1.0), color="#555", linewidth=1.0)

    # Row 4 — merge_contexts
    box(ax, (3.7, 2.4), 3.0, 1.0,
        "merge_contexts\n(slot allocation by α, β, γ)",
        facecolor="#FFE9A8", bold=True)
    arrow(ax, (1.8, src_y), (4.2, 3.4))
    arrow(ax, (5.2, src_y), (5.2, 3.4))
    arrow(ax, (8.6, src_y), (6.2, 3.4))

    # Row 5 — LLM + answer
    box(ax, (7.1, 2.4), 3.0, 1.0,
        "LLM (gpt-4o-mini)\ntemp=0, max=500", facecolor="#DDECF5")
    arrow(ax, (6.7, 2.9), (7.1, 2.9))
    box(ax, (10.5, 2.4), 3.0, 1.0, "Answer a\n(자연어/리스트)",
        facecolor="#F9E5D0", bold=True)
    arrow(ax, (10.1, 2.9), (10.5, 2.9))

    # Prompt style annotation
    box(ax, (7.1, 0.8), 3.0, 1.0,
        "PROMPT_TEMPLATE (sentence)\nor\nPROMPT_TEMPLATE_LIST (§Ⅵ.3.4)",
        facecolor="#F3F3F3", fontsize=8)
    arrow(ax, (8.6, 1.8), (8.6, 2.4), color="#999")

    # Stage labels (1..7)
    ax.text(1.4, 8.5, "(1) 입력", ha="center", fontsize=9, color="#666")
    ax.text(4.7, 8.5, "(2) 의도 분석", ha="center", fontsize=9, color="#666")
    ax.text(8.4, 8.5, "(3) 가중치 결정", ha="center", fontsize=9, color="#666")
    ax.text(5.2, src_y + 1.3, "(4) 3-source 검색", ha="center", fontsize=9, color="#666")
    ax.text(5.2, 3.6, "(5) 컨텍스트 병합", ha="center", fontsize=9, color="#666")
    ax.text(8.6, 3.6, "(6) 생성", ha="center", fontsize=9, color="#666")
    ax.text(12.0, 3.6, "(7) 출력", ha="center", fontsize=9, color="#666")

    ax.set_title(
        "Figure 3-1. Triple-Hybrid RAG pipeline (query → answer, 7 stages). "
        "α, β, γ 은 DWA (R-DWA 또는 L-DWA) 가 질의마다 결정한다.",
        fontsize=11, pad=10,
    )

    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
