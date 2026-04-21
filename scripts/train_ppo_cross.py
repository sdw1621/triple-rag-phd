"""
Train PPO L-DWA on a cross-domain benchmark cache (HotpotQA / MuSiQue / PubMedQA).

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 6 Section 7 (domain-specific fine-tuning)

Unlike scripts/train_ppo.py which assumes the shared university corpus,
this trainer uses per-query contexts from the benchmark itself. State is
built from per-query FAISS search over the supplied paragraphs. Graph
and Ontology are treated as empty (zero source_stats for g and o).

Usage:
    python scripts/train_ppo_cross.py \\
        --benchmark hotpotqa \\
        --input data/hotpotqa/hard_300.json \\
        --cache cache/hotpotqa.sqlite \\
        --output cache/ppo_checkpoints/hotpotqa_seed_42 \\
        --seed 42 --episodes 10000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dwa.base import DWAWeights  # noqa: E402
from src.intent.rule_based import QueryIntent, RuleBasedIntent  # noqa: E402
from src.ppo.actor_critic import ActorCritic  # noqa: E402
from src.ppo.mdp import (  # noqa: E402
    STATE_DIM,
    State,
    extract_query_meta,
    extract_source_stats,
)
from src.ppo.trainer import PPOConfig, PPOTrainer  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402
from src.utils.offline_cache import DEFAULT_GRID, OfflineCache  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("train_ppo_cross")

TOP_K = 3


# ---------- benchmark loader (same as cross_build_cache) ----------

def load_hotpotqa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "qid": str(r["id"]),
            "question": r["question"],
            "answer": r["answer"],
            "contexts": [
                f"{r['context']['title'][i]}: " + " ".join(r['context']['sentences'][i])
                for i in range(min(len(r['context']['title']), len(r['context']['sentences'])))
            ],
        }
        for r in raw
    ]


def load_musique(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "qid": str(r["id"]),
            "question": r["question"],
            "answer": r["answer"],
            "contexts": [
                f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
                for p in r.get("paragraphs", [])
            ],
        }
        for r in raw
    ]


def load_pubmedqa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for r in raw:
        ctx = r.get("context", {})
        out.append({
            "qid": str(r["pubid"]),
            "question": r["question"],
            "answer": r.get("long_answer") or r.get("final_decision", ""),
            "contexts": ctx.get("contexts", []),
        })
    return out


LOADERS: dict[str, Callable[[Path], list[dict]]] = {
    "hotpotqa": load_hotpotqa, "musique": load_musique, "pubmedqa": load_pubmedqa,
}


# ---------- state pre-compute ----------

def precompute_states(items: list[dict], analyzer: RuleBasedIntent) -> list[State]:
    """Build 18-dim State per query using per-query VectorStore."""
    states: list[State] = []
    t0 = time.time()
    for i, item in enumerate(items):
        intent = analyzer.analyze(item["question"])
        if item["contexts"]:
            vs = VectorStore()
            vs.add_documents(item["contexts"])
            v_hits = vs.search(item["question"], top_k=TOP_K)
            v_scores = [s for _, s in v_hits]
        else:
            v_scores = []
        state = State(
            density=intent.density,
            intent_logits=(0.0, 0.0, 0.0),
            source_stats=extract_source_stats(v_scores, [], []),  # graph/onto empty
            query_meta=extract_query_meta(item["question"], intent),
        )
        states.append(state)
        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            logger.info("  states %d/%d (%.1fs)", i + 1, len(items), time.time() - t0)
    return states


def snap_to_simplex_int(weights: DWAWeights, grid: int = DEFAULT_GRID) -> tuple[int, int, int]:
    a = round(weights.alpha * grid)
    b = round(weights.beta * grid)
    g = round(weights.gamma * grid)
    residual = grid - (a + b + g)
    if residual != 0:
        if a >= b and a >= g:
            a += residual
        elif b >= g:
            b += residual
        else:
            g += residual
    return (max(0, min(grid, a)), max(0, min(grid, b)), max(0, min(grid, g)))


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=list(LOADERS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--state-cache", type=Path, default=None)
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    items = LOADERS[args.benchmark](args.input)
    qids = [it["qid"] for it in items]
    logger.info("Benchmark %s: %d queries, seed %d, episodes %d",
                args.benchmark, len(items), args.seed, args.episodes)

    analyzer = RuleBasedIntent()

    if args.state_cache and args.state_cache.exists():
        logger.info("Loading states from %s", args.state_cache)
        with args.state_cache.open("rb") as f:
            states = pickle.load(f)
    else:
        logger.info("Pre-computing %d states...", len(items))
        states = precompute_states(items, analyzer)
        if args.state_cache:
            args.state_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.state_cache.open("wb") as f:
                pickle.dump(states, f)
            logger.info("States saved to %s", args.state_cache)

    def state_provider(idx: int) -> State:
        return states[idx]

    cache = OfflineCache(args.cache)
    logger.info("Cache: %s", cache.stats())

    def reward_fn(query_index: int, weights: DWAWeights) -> float:
        triple = snap_to_simplex_int(weights)
        step = 1.0 / DEFAULT_GRID
        w = DWAWeights(triple[0] * step, triple[1] * step, triple[2] * step)
        rc = cache.get(qids[query_index], w)
        if rc is None:
            return 0.0
        return rc.total_reward()

    config = PPOConfig()
    logger.info("PPOConfig: %s", config.__dict__)

    rng = np.random.default_rng(args.seed)
    ac = ActorCritic().to(args.device)
    trainer = PPOTrainer(
        actor_critic=ac, state_provider=state_provider,
        reward_fn=reward_fn, n_queries=len(items),
        config=config, device=args.device, rng=rng,
    )

    logger.info("Starting training for %d episodes...", args.episodes)
    history: list[dict[str, float]] = []
    t0 = time.time()
    for episode in range(args.episodes):
        metrics = trainer.train_step()
        history.append(metrics)
        if (episode + 1) % args.report_every == 0 or episode == 0:
            recent = history[-args.report_every:]
            mean_R = float(np.mean([m["mean_reward"] for m in recent]))
            elapsed = time.time() - t0
            eta = (args.episodes - episode - 1) / ((episode + 1) / elapsed)
            logger.info(
                "ep %5d/%d  R=%.4f  rate=%.1f ep/s  eta=%.0fs",
                episode + 1, args.episodes, mean_R,
                (episode + 1) / elapsed, eta,
            )
        if (episode + 1) % args.checkpoint_every == 0:
            ck = args.output / f"ep{episode + 1:06d}.pt"
            trainer.save_checkpoint(ck)
            logger.info("  checkpoint: %s", ck)

    final = args.output / "final.pt"
    trainer.save_checkpoint(final)
    elapsed = time.time() - t0
    early = float(np.mean([m["mean_reward"] for m in history[:100]]))
    late = float(np.mean([m["mean_reward"] for m in history[-100:]]))
    logger.info(
        "Training done in %.1fs. Trajectory: early=%.4f late=%.4f Δ=%+.4f (%s)",
        elapsed, early, late, late - early,
        "learning" if late - early > 0.01 else "flat",
    )
    logger.info("Final checkpoint: %s", final)

    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
