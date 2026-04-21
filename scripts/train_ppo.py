"""
Train PPO L-DWA policy using the offline reward cache.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 5 Section 3 (training), Table 5-4 (config)

The trainer wires:
  - StateProvider: maps a training query index to an 18-dim State.
    Pre-computes intent (density) + retrievals (source_stats) once per query
    at startup. intent_logits left as zeros until BERT classifier lands (M6.5).
  - RewardFn: looks up ``cache.get(qid, weights) -> RewardComponents`` and
    returns ``total_reward()``. Continuous Dirichlet actions are snapped to
    the nearest integer simplex point (0.1 grid) before cache lookup.

Usage:
    # Dry-run (short) to verify convergence on seed 42:
    python scripts/train_ppo.py \\
        --cache cache/university.sqlite \\
        --qa data/university/gold_qa_5000.json \\
        --output cache/ppo_checkpoints/seed_42 \\
        --seed 42 --episodes 500 --device cpu

    # Full thesis run (per seed):
    python scripts/train_ppo.py \\
        --cache cache/university.sqlite \\
        --qa data/university/gold_qa_5000.json \\
        --output cache/ppo_checkpoints/seed_42 \\
        --seed 42 --episodes 10000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
from src.rag.university_loader import (  # noqa: E402
    build_documents,
    build_graph,
    build_ontology,
)
from src.rag.vector_store import VectorStore  # noqa: E402
from src.utils.offline_cache import (  # noqa: E402
    DEFAULT_GRID,
    OfflineCache,
)
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("train_ppo")

TOP_K = 3


# ---------- helpers ----------

def snap_to_simplex_int(weights: DWAWeights, grid: int = DEFAULT_GRID) -> tuple[int, int, int]:
    """Snap continuous weights to the nearest integer simplex point.

    The cache only stores entries where a+b+g=grid; independent rounding of
    each component can produce invalid triples (sum=9 or 11). We round each
    component, then redistribute the residual to the largest component so the
    sum equals ``grid`` exactly.
    """
    a = round(weights.alpha * grid)
    b = round(weights.beta * grid)
    g = round(weights.gamma * grid)
    residual = grid - (a + b + g)
    if residual != 0:
        # Attribute the error to the largest component.
        if a >= b and a >= g:
            a += residual
        elif b >= g:
            b += residual
        else:
            g += residual
    # Clip to [0, grid] in case of extreme rounding.
    a = max(0, min(grid, a))
    b = max(0, min(grid, b))
    g = max(0, min(grid, g))
    # Final correction if clipping broke the sum.
    residual = grid - (a + b + g)
    if residual != 0:
        for ref in (0, 1, 2):
            vals = [a, b, g]
            if 0 <= vals[ref] + residual <= grid:
                vals[ref] += residual
                a, b, g = vals
                break
    return (a, b, g)


def cache_get_by_int(
    cache: OfflineCache, query_id: str, triple: tuple[int, int, int], grid: int = DEFAULT_GRID
) -> float | None:
    """Look up a reward by pre-discretized integer triple. Returns total_reward()."""
    step = 1.0 / grid
    weights = DWAWeights(triple[0] * step, triple[1] * step, triple[2] * step)
    rc = cache.get(query_id, weights)
    if rc is None:
        return None
    return rc.total_reward()


# ---------- state pre-compute ----------

def precompute_states(
    queries: list[dict],
    vector: VectorStore,
    graph: Any,
    ontology: Any,
    analyzer: RuleBasedIntent,
) -> list[State]:
    """For each training query, build its 18-dim State once."""
    states: list[State] = []
    t0 = time.time()
    for i, item in enumerate(queries):
        query = item["query"]
        intent = analyzer.analyze(query)
        v_hits = vector.search(query, top_k=TOP_K)
        v_scores = [s for _, s in v_hits]
        # Graph/ontology return raw strings without scores — treat as 1.0 each.
        g_paths = graph.search(query, top_k=TOP_K)
        o_facts = ontology.search(query, top_k=TOP_K)
        g_scores = [1.0] * len(g_paths)
        o_scores = [1.0] * len(o_facts)

        state = State(
            density=intent.density,
            intent_logits=(0.0, 0.0, 0.0),  # TODO M6.5: replace with BERT logits
            source_stats=extract_source_stats(v_scores, g_scores, o_scores),
            query_meta=extract_query_meta(query, intent),
        )
        states.append(state)
        if (i + 1) % 500 == 0 or (i + 1) == len(queries):
            logger.info("  states %d/%d (%.1fs)", i + 1, len(queries), time.time() - t0)
    return states


# ---------- optional TensorBoard writer ----------

def make_writer(logdir: Path | None):
    if logdir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard not available — skipping writer")
        return None
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(logdir))


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Checkpoint directory (created if missing).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Total PPO episodes (thesis Table 5-4: 10000).")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--tensorboard", action="store_true",
                        help="Write TensorBoard logs under <output>/tb.")
    parser.add_argument("--qa-limit", type=int, default=None,
                        help="Use only first N queries (dry-run).")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--state-cache", type=Path, default=None,
                        help="Pickle path to cache/load pre-computed states "
                             "(speeds up multi-seed runs).")
    # Optional hyperparameter overrides (thesis defaults otherwise)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s  Seed: %d  Episodes: %d", args.device, args.seed, args.episodes)

    # ---------- data ----------
    with args.qa.open(encoding="utf-8") as f:
        qa_data = json.load(f)
    if args.qa_limit:
        qa_data = qa_data[: args.qa_limit]
    logger.info("Loaded %d training queries from %s", len(qa_data), args.qa)

    # Map training index → query_id.
    qids: list[str] = [str(item["id"]) for item in qa_data]

    # ---------- pipeline components ----------
    from langchain_openai import OpenAIEmbeddings  # lazy

    logger.info("Loading pipeline components (vector/graph/ontology)...")
    docs = build_documents()
    graph = build_graph()
    ontology = build_ontology()
    analyzer = RuleBasedIntent()

    vector = VectorStore()  # uses OpenAI embedder (same as build_cache.py)
    t0 = time.time()
    vector.add_documents(docs)
    logger.info("Vector store built in %.1fs (%d docs)", time.time() - t0, vector.n_documents)

    # ---------- pre-compute states (with optional disk cache) ----------
    import pickle
    state_cache = args.state_cache
    if state_cache and state_cache.exists():
        logger.info("Loading pre-computed states from %s", state_cache)
        with state_cache.open("rb") as f:
            states = pickle.load(f)
        assert len(states) == len(qa_data), (
            f"state cache size {len(states)} != qa set {len(qa_data)} — "
            "regenerate with --qa-limit matching or delete the cache"
        )
    else:
        logger.info("Pre-computing per-query states (density + source_stats + meta)...")
        states = precompute_states(qa_data, vector, graph, ontology, analyzer)
        if state_cache:
            state_cache.parent.mkdir(parents=True, exist_ok=True)
            with state_cache.open("wb") as f:
                pickle.dump(states, f)
            logger.info("States saved to %s", state_cache)
    logger.info("States ready: %d × %d-dim", len(states), STATE_DIM)

    def state_provider(idx: int) -> State:
        return states[idx]

    # ---------- cache + reward fn ----------
    cache = OfflineCache(args.cache)
    stats = cache.stats()
    logger.info("Cache: %s", stats)

    cache_hits = {"n_hits": 0, "n_miss": 0}

    def reward_fn(query_index: int, weights: DWAWeights) -> float:
        triple = snap_to_simplex_int(weights)
        r = cache_get_by_int(cache, qids[query_index], triple)
        if r is None:
            cache_hits["n_miss"] += 1
            return 0.0
        cache_hits["n_hits"] += 1
        return r

    # ---------- trainer ----------
    config = PPOConfig(
        learning_rate=args.lr,
        rollout_size=args.rollout_size,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        entropy_coef=args.entropy_coef,
    )
    logger.info("PPOConfig: %s", config.__dict__)

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

    # ---------- training loop ----------
    logger.info("Starting training for %d episodes...", args.episodes)
    history: list[dict[str, float]] = []
    t0 = time.time()
    for episode in range(args.episodes):
        metrics = trainer.train_step()
        history.append(metrics)
        if (episode + 1) % args.report_every == 0 or episode == 0:
            recent = history[-args.report_every :]
            mean_R = float(np.mean([m["mean_reward"] for m in recent]))
            mean_pl = float(np.mean([m["policy_loss"] for m in recent]))
            mean_vl = float(np.mean([m["value_loss"] for m in recent]))
            mean_ent = float(np.mean([m["entropy"] for m in recent]))
            elapsed = time.time() - t0
            eps_per_s = (episode + 1) / elapsed
            eta = (args.episodes - episode - 1) / eps_per_s
            logger.info(
                "ep %5d/%d  R=%.4f  π_loss=%.4f  V_loss=%.4f  H=%.3f  "
                "rate=%.1f ep/s  eta=%.0fs",
                episode + 1, args.episodes, mean_R, mean_pl, mean_vl, mean_ent,
                eps_per_s, eta,
            )
        if (episode + 1) % args.checkpoint_every == 0:
            ckpt = args.output / f"ep{episode + 1:06d}.pt"
            trainer.save_checkpoint(ckpt)
            logger.info("  checkpoint: %s", ckpt)

    # ---------- finalize ----------
    final_ckpt = args.output / "final.pt"
    trainer.save_checkpoint(final_ckpt)
    elapsed = time.time() - t0
    logger.info("Training done in %.1fs (%.2f ep/s)", elapsed, args.episodes / elapsed)
    logger.info("Final checkpoint: %s", final_ckpt)
    logger.info("Cache hits: %d / misses: %d", cache_hits["n_hits"], cache_hits["n_miss"])

    # Save history JSON
    history_path = args.output / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("History: %s", history_path)

    if writer is not None:
        writer.close()

    # Quick summary
    early_R = float(np.mean([m["mean_reward"] for m in history[:100]]))
    late_R = float(np.mean([m["mean_reward"] for m in history[-100:]]))
    delta = late_R - early_R
    logger.info(
        "mean_reward trajectory: early=%.4f  late=%.4f  Δ=%+.4f  (%s)",
        early_R, late_R, delta,
        "learning" if delta > 0.01 else "flat" if abs(delta) <= 0.01 else "degrading",
    )

    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
