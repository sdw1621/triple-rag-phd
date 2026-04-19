"""Tests for src.dwa.ldwa.LearnedDWA."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.dwa.base import BaseDWA, DWAWeights
from src.dwa.ldwa import LearnedDWA
from src.intent.rule_based import QueryIntent, RuleBasedIntent
from src.ppo.actor_critic import ActorCritic
from src.ppo.mdp import State
from src.ppo.trainer import PPOConfig, PPOTrainer
from src.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _seed() -> None:
    set_seed(42)


def _zero_state(_query: str, _intent: QueryIntent) -> State:
    return State(
        density=(0.0, 0.0, 0.0),
        intent_logits=(0.0, 0.0, 0.0),
        source_stats=tuple([0.0] * 9),
        query_meta=(0.0, 0.0, 0.0),
    )


def _intent(qtype="simple") -> QueryIntent:
    return QueryIntent(
        query_type=qtype,
        entities=[],
        relations=[],
        constraints=[],
        complexity_score=0.0,
        density=(0.1, 0.2, 0.3),
    )


# ---------- interface ----------

def test_ldwa_implements_base_dwa() -> None:
    ldwa = LearnedDWA(ActorCritic(), _zero_state)
    assert isinstance(ldwa, BaseDWA)


def test_compute_returns_simplex_weights() -> None:
    ldwa = LearnedDWA(ActorCritic(), _zero_state)
    w = ldwa.compute("any query", _intent())
    assert isinstance(w, DWAWeights)
    assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6
    assert all(0.0 <= x <= 1.0 for x in (w.alpha, w.beta, w.gamma))


def test_compute_is_deterministic_given_fixed_state() -> None:
    """act_mean is deterministic — same state should yield same weights."""
    ldwa = LearnedDWA(ActorCritic(), _zero_state)
    w1 = ldwa.compute("q1", _intent())
    w2 = ldwa.compute("q2", _intent())
    # Same state_builder output → same Dirichlet mean.
    assert w1 == w2


def test_state_builder_is_called_with_query_and_intent() -> None:
    seen: list[tuple[str, QueryIntent]] = []

    def builder(query: str, intent: QueryIntent) -> State:
        seen.append((query, intent))
        return _zero_state(query, intent)

    ldwa = LearnedDWA(ActorCritic(), builder)
    intent = _intent("conditional")
    ldwa.compute("test query", intent)
    assert seen == [("test query", intent)]


# ---------- end-to-end with PPOTrainer ----------

def test_ldwa_loads_trained_policy_from_ppo_trainer(tmp_path: Path) -> None:
    """Train a tiny PPO policy on a synthetic α-dominant problem, save the
    checkpoint, then verify LearnedDWA.from_checkpoint reproduces the same
    α-dominant weights at inference time."""

    def state_for_idx(_idx: int) -> State:
        return _zero_state("", _intent())

    def reward_fn(_idx: int, weights: DWAWeights) -> float:
        return float(weights.alpha)

    trainer = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=state_for_idx,
        reward_fn=reward_fn,
        n_queries=32,
        config=PPOConfig(
            rollout_size=32,
            minibatch_size=8,
            update_epochs=4,
            learning_rate=1e-2,
            entropy_coef=0.0,
        ),
    )
    trainer.train(total_episodes=40)
    ckpt = tmp_path / "policy.pt"
    trainer.save_checkpoint(ckpt)

    ldwa = LearnedDWA.from_checkpoint(ckpt, _zero_state)
    w = ldwa.compute("any", _intent())
    # After training on α-dominant reward, learned policy should put
    # noticeably more mass on α than the uniform 1/3 baseline.
    assert w.alpha > 1.0 / 3, f"L-DWA failed to learn α dominance: {w}"


def test_ldwa_works_with_real_intent_analyzer() -> None:
    """Smoke test: glue path from RuleBasedIntent → State → LearnedDWA."""
    analyzer = RuleBasedIntent()
    intent = analyzer.analyze("40세 이하 컴퓨터공학과 교수는?")

    def builder(query: str, intent_in: QueryIntent) -> State:
        return State(
            density=intent_in.density,
            intent_logits=(0.0, 0.0, 0.0),
            source_stats=tuple([0.0] * 9),
            query_meta=(0.0, 0.0, 0.0),
        )

    ldwa = LearnedDWA(ActorCritic(), builder)
    w = ldwa.compute("40세 이하 컴퓨터공학과 교수는?", intent)
    assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6


# ---------- pluggable into TripleHybridRAG ----------

def test_ldwa_pluggable_into_triple_hybrid_rag() -> None:
    """L-DWA should drop into TripleHybridRAG without changes (BaseDWA contract)."""
    import hashlib

    import numpy as np

    from src.rag.graph_store import GraphStore
    from src.rag.ontology_store import OntologyStore
    from src.rag.triple_hybrid_rag import TripleHybridRAG
    from src.rag.vector_store import VectorStore

    class HashEmbedder:
        DIM = 32

        def _embed(self, text: str) -> list[float]:
            vec = np.zeros(self.DIM, dtype="float32")
            for tok in text.split():
                h = hashlib.md5(tok.encode("utf-8")).digest()
                for i in range(0, len(h), 4):
                    vec[int.from_bytes(h[i : i + 4], "big") % self.DIM] += 1.0
            if not vec.any():
                vec[0] = 1.0
            return vec.tolist()

        def embed_documents(self, documents: list[str]) -> list[list[float]]:
            return [self._embed(d) for d in documents]

        def embed_query(self, query: str) -> list[float]:
            return self._embed(query)

    class EchoLLM:
        def generate(self, prompt: str) -> str:
            return "ok"

    vec = VectorStore(embedder=HashEmbedder())
    vec.add_documents(["doc1"])

    def builder(query: str, intent: QueryIntent) -> State:
        return _zero_state(query, intent)

    ldwa = LearnedDWA(ActorCritic(), builder)
    pipeline = TripleHybridRAG(
        vector_store=vec,
        graph_store=GraphStore(),
        ontology_store=OntologyStore(try_owlready=False),
        dwa=ldwa,
        llm=EchoLLM(),
    )
    result = pipeline.query("test")
    assert isinstance(result.weights, DWAWeights)
