"""
Figure 5-2: 3-seed PPO convergence overlay.

Reads cache/ppo_checkpoints/seed_{42,123,999}/history.json produced by
scripts/train_ppo.py and plots mean_reward vs update step for all three
seeds on one panel, with light moving-average smoothing for readability.

Thesis reference: Ch.5 §3.5 학습 곡선 관찰.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "cache" / "ppo_checkpoints"
FIG_OUT = ROOT / "docs" / "figures" / "fig5_2_ppo_convergence.png"

SEEDS = [42, 123, 999]
COLORS = {42: "#C00000", 123: "#4F86C6", 999: "#5B9B5E"}


def smooth(arr: list[float], window: int = 50) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if len(a) < window:
        return a
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="valid")


def main() -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), sharex=False, constrained_layout=True
    )

    # Left: mean reward
    for seed in SEEDS:
        hist_path = CKPT_DIR / f"seed_{seed}" / "history.json"
        if not hist_path.exists():
            print(f"WARN: {hist_path} missing, skipping")
            continue
        with hist_path.open(encoding="utf-8") as f:
            hist = json.load(f)
        r = [h["mean_reward"] for h in hist]
        x = np.arange(len(r))
        ax1.plot(x, r, color=COLORS[seed], alpha=0.18, linewidth=0.7)
        s = smooth(r, window=50)
        ax1.plot(x[len(r) - len(s) :], s, color=COLORS[seed],
                 label=f"seed {seed}", linewidth=1.8)

    ax1.set_xlabel("PPO update step")
    ax1.set_ylabel("Mean reward (rollout)")
    ax1.set_title("Mean reward over training (3 seeds)")
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=9)

    # Right: entropy (shows exploration decay)
    for seed in SEEDS:
        hist_path = CKPT_DIR / f"seed_{seed}" / "history.json"
        if not hist_path.exists():
            continue
        with hist_path.open(encoding="utf-8") as f:
            hist = json.load(f)
        ent = [h["entropy"] for h in hist]
        x = np.arange(len(ent))
        ax2.plot(x, ent, color=COLORS[seed], alpha=0.18, linewidth=0.7)
        s = smooth(ent, window=50)
        ax2.plot(x[len(ent) - len(s) :], s, color=COLORS[seed],
                 label=f"seed {seed}", linewidth=1.8)

    ax2.set_xlabel("PPO update step")
    ax2.set_ylabel("Policy entropy")
    ax2.set_title("Policy entropy over training (3 seeds)")
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Figure 5-2. PPO L-DWA training curves — 3 seeds (42, 123, 999)",
        fontsize=12,
    )
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
