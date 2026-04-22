"""
Figure 6-4: λ sensitivity analysis.

Varies the reward function weights and recomputes the Oracle policy
from the 330K university.sqlite cache — for each (λ_F1, λ_EM, λ_Faith)
triple, argmax R over the 66 weight grid per query, then average the
resulting F1, EM, Faithfulness over 5,000 queries.

Shows that the chosen default (0.5, 0.3, 0.2) sits in a stable region
— moderate variation in λ doesn't swing per-query weight choice
dramatically, supporting the robustness claim in Ch.6 §9.

Thesis reference: Ch.6 §9.3 λ 민감도 분석.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = ROOT / "cache" / "university.sqlite"
FIG_OUT = ROOT / "docs" / "figures" / "fig6_4_lambda_sensitivity.png"

_FONT_PATH = ROOT / "cache" / "malgun.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def oracle_stats(conn: sqlite3.Connection, lam_f1: float, lam_em: float,
                 lam_faith: float, lam_lat: float = 0.1) -> dict:
    """Compute per-query argmax reward, return averaged metrics."""
    cur = conn.execute(
        "SELECT query_id, f1, em, faithfulness, latency FROM rewards"
    )
    best: dict[str, tuple[float, float, float, float, float]] = {}
    for qid, f1, em, fa, lat in cur:
        r = (
            lam_f1 * f1 + lam_em * em + lam_faith * fa
            - lam_lat * max(0.0, lat - 5.0)
        )
        prev = best.get(qid)
        if prev is None or r > prev[0]:
            best[qid] = (r, f1, em, fa, lat)
    arr = np.array(list(best.values()))
    return {
        "n": len(best),
        "R": float(arr[:, 0].mean()),
        "F1": float(arr[:, 1].mean()),
        "EM": float(arr[:, 2].mean()),
        "Faith": float(arr[:, 3].mean()),
        "Latency": float(arr[:, 4].mean()),
    }


def main() -> int:
    if not CACHE_DB.exists():
        raise SystemExit(f"Missing cache DB: {CACHE_DB}")

    conn = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)

    # Sweep 1: vary λ_F1 with λ_EM + λ_Faith held proportional
    f1_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    sweep1: list[dict] = []
    for lf1 in f1_values:
        remaining = 1.0 - lf1
        # Keep EM:Faith = 3:2 ratio (as in thesis default 0.3 / 0.2)
        lem = remaining * 0.6
        lfa = remaining * 0.4
        s = oracle_stats(conn, lf1, lem, lfa)
        s["lambda_f1"] = float(lf1)
        sweep1.append(s)

    # Sweep 2: vary λ_Faith with λ_F1 fixed at 0.5
    faith_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    sweep2: list[dict] = []
    for lfa in faith_values:
        lf1 = 0.5
        remaining = 1.0 - lf1 - lfa
        lem = max(remaining, 0.0)
        if lem < 0:
            continue
        s = oracle_stats(conn, lf1, lem, lfa)
        s["lambda_faith"] = float(lfa)
        sweep2.append(s)

    conn.close()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left panel — λ_F1 sweep
    lf1 = [s["lambda_f1"] for s in sweep1]
    ax1.plot(lf1, [s["F1"] for s in sweep1], marker="o", color="#C00000",
             label="F1", linewidth=2)
    ax1.plot(lf1, [s["EM"] for s in sweep1], marker="s", color="#4F86C6",
             label="EM", linewidth=2)
    ax1.plot(lf1, [s["Faith"] for s in sweep1], marker="^", color="#5B9B5E",
             label="Faithfulness", linewidth=2)
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax1.text(0.51, 0.02, "default", rotation=90, fontsize=9, color="gray",
             va="bottom", ha="left")
    ax1.set_xlabel("λ_F1 (reward weight on F1)")
    ax1.set_ylabel("Oracle-policy metric (mean, 5,000 QA)")
    ax1.set_title("Oracle metrics vs λ_F1 (λ_EM:λ_Faith = 3:2 fixed)")
    ax1.set_xticks(f1_values)
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=9)

    # Right panel — λ_Faith sweep
    lfa = [s["lambda_faith"] for s in sweep2]
    ax2.plot(lfa, [s["F1"] for s in sweep2], marker="o", color="#C00000",
             label="F1", linewidth=2)
    ax2.plot(lfa, [s["EM"] for s in sweep2], marker="s", color="#4F86C6",
             label="EM", linewidth=2)
    ax2.plot(lfa, [s["Faith"] for s in sweep2], marker="^", color="#5B9B5E",
             label="Faithfulness", linewidth=2)
    ax2.axvline(0.2, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax2.text(0.21, 0.02, "default", rotation=90, fontsize=9, color="gray",
             va="bottom", ha="left")
    ax2.set_xlabel("λ_Faith (λ_F1=0.5 fixed)")
    ax2.set_ylabel("Oracle-policy metric (mean, 5,000 QA)")
    ax2.set_title("Oracle metrics vs λ_Faith (λ_F1=0.5 fixed)")
    ax2.set_xticks(faith_values)
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "Figure 6-4. Reward-weight sensitivity analysis (Oracle policy, 5,000 QA)",
        fontsize=12,
    )

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")

    # Summary print
    default = next(s for s in sweep1 if abs(s["lambda_f1"] - 0.5) < 1e-9)
    print("\nAt default λ_F1=0.5: F1={:.3f}  EM={:.3f}  Faith={:.3f}  R={:.3f}".format(
        default["F1"], default["EM"], default["Faith"], default["R"]
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
