"""
Train PPO L-DWA on the English synthetic university benchmark.

Adapted from scripts/train_ppo.py. Key differences:
  - Uses data/university_en/ (English corpus + 403 QA)
  - Graph / Ontology empty (English graph/ontology not built)
  - Reads English cache built by build_cache_en.py (~26.6K entries)

Usage (in-container):
    python scripts/train_ppo_en.py \\
        --cache cache/university_en.sqlite \\
        --output cache/ppo_checkpoints/en_seed_42 \\
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
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dwa.base import DWAWeights  # noqa: E402
from src.intent.rule_based import RuleBasedIntent  # noqa: E402
from src.ppo.actor_critic import ActorCritic  # noqa: E402
from src.ppo.mdp import (  # noqa: E402
    STATE_DIM,
    State,
    extract_query_meta,
    extract_source_stats,
)
from src.ppo.trainer import PPOConfig, PPOTrainer  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402
from src.utils.offline_cache import OfflineCache, discretize_weights  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("train_ppo_en")

CORPUS_PATH = ROOT / "data" / "university_en" / "corpus_en.json"
QA_PATH = ROOT / "data" / "university_en" / "gold_qa_500_en.json"
TOP_K = 3


def make_writer(logdir: Path | None):
    if logdir is None:
        return None
    from torch.utils.tensorboard import SummaryWriter
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(logdir))


def snap_to_simplex_int(weights: DWAWeights, grid: int = 10) -> tuple[int, int, int]:
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


def cache_get_by_int(cache: OfflineCache, query_id: str,
                     triple: tuple[int, int, int]) -> float | None:
    """Look up total_reward by integer triple."""
    a, b, g = triple
    w = DWAWeights(a / 10, b / 10, g / 10)
    r = cache.get(query_id, w)
    return r.total_reward() if r else None


def precompute_states_en(
    qa_data: list[dict], vector: VectorStore, analyzer: RuleBasedIntent
) -> list[State]:
    """English-variant: Graph/Ontology are empty, so source_stats has zeros
    for those two sources. Density comes from the bilingual intent analyzer."""
    states: list[State] = []
    for item in qa_data:
        q = item["query"]
        intent = analyzer.analyze(q)
        v_hits = vector.search(q, top_k=TOP_K)
        v_scores = [s for _, s in v_hits]
        # Graph / Ontology empty
        state = State(
            density=intent.density,
            intent_logits=(0.0, 0.0, 0.0),
            source_stats=extract_source_stats(v_scores, [], []),
            query_meta=extract_query_meta(q, intent),
        )
        states.append(state)
    return states


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--qa-limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--state-cache", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s  Seed: %d  Episodes: %d",
                args.device, args.seed, args.episodes)

    with QA_PATH.open(encoding="utf-8") as f:
        qa_data = json.load(f)
    if args.qa_limit:
        qa_data = qa_data[: args.qa_limit]
    logger.info("Loaded %d training queries (EN)", len(qa_data))

    qids = [str(item["id"]) for item in qa_data]

    # Build Vector from English corpus
    with CORPUS_PATH.open(encoding="utf-8") as f:
        corpus = json.load(f)
    vector = VectorStore()
    t0 = time.time()
    vector.add_documents(corpus["documents"])
    logger.info("VectorStore: %d docs in %.1fs", len(corpus["documents"]), time.time() - t0)

    analyzer = RuleBasedIntent()

    # Pre-compute states
    if args.state_cache and args.state_cache.exists():
        with args.state_cache.open("rb") as f:
            states = pickle.load(f)
        logger.info("Loaded %d states from cache", len(states))
    else:
        logger.info("Pre-computing per-query states...")
        states = precompute_states_en(qa_data, vector, analyzer)
        if args.state_cache:
            args.state_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.state_cache.open("wb") as f:
                pickle.dump(states, f)

    logger.info("States: %d × %d-dim", len(states), STATE_DIM)

    def state_provider(idx: int) -> State:
        return states[idx]

    cache = OfflineCache(args.cache)
    logger.info("Cache: %s", cache.stats())

    cache_hits = {"n_hits": 0, "n_miss": 0}

    def reward_fn(query_index: int, weights: DWAWeights) -> float:
        triple = snap_to_simplex_int(weights)
        r = cache_get_by_int(cache, qids[query_index], triple)
        if r is None:
            cache_hits["n_miss"] += 1
            return 0.0
        cache_hits["n_hits"] += 1
        return r

    config = PPOConfig(
        learning_rate=args.lr,
        rollout_size=args.rollout_size,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        entropy_coef=args.entropy_coef,
    )

    rng = np.random.default_rng(args.seed)
    writer = make_writer(args.output / "tb" if args.tensorboard else None)
    ac = ActorCritic().to(args.device)
    trainer = PPOTrainer(
        actor_critic=ac,
        state_provider=state_provider,
        reward_fn=reward_fn,
        n_queries=len(qa_data),
        config=config,
        device=args.device,
        writer=writer,
        rng=rng,
    )

    logger.info("Training...")
    t0 = time.time()
    history = trainer.train(total_episodes=args.episodes)
    elapsed = time.time() - t0
    logger.info("Training done in %.1fs", elapsed)
    logger.info("Cache hits=%d misses=%d",
                cache_hits["n_hits"], cache_hits["n_miss"])

    # Save final checkpoint
    torch.save(
        {"actor_critic": ac.state_dict(), "config": config.__dict__},
        args.output / "final.pt",
    )
    # Save history for plotting
    with (args.output / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
