"""Tests for src.ppo.trainer.PPOTrainer.

Uses a synthetic problem: there's a known-best weight (e.g., always pick
α-dominant) and the trainer should converge toward it. Reward is
closed-form so tests are fully offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dwa.base import DWAWeights
from src.ppo.actor_critic import ActorCritic
from src.ppo.mdp import State
from src.ppo.trainer import (
    PPOConfig,
    PPOTrainer,
    Rollout,
    _action_to_weights,
    compute_gae,
)
from src.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _seed() -> None:
    set_seed(42)


# ---------- helpers ----------

def _zero_state(_idx: int) -> State:
    return State(
        density=(0.0, 0.0, 0.0),
        intent_logits=(0.0, 0.0, 0.0),
        source_stats=tuple([0.0] * 9),
        query_meta=(0.0, 0.0, 0.0),
    )


def _alpha_dominant_reward(_idx: int, weights: DWAWeights) -> float:
    """Reward that strictly increases with α — optimal action is (1, 0, 0)."""
    return float(weights.alpha)


# ---------- _action_to_weights ----------

def test_action_to_weights_handles_simplex_vector() -> None:
    a = torch.tensor([0.5, 0.3, 0.2])
    w = _action_to_weights(a)
    assert w.alpha + w.beta + w.gamma == pytest.approx(1.0, abs=1e-9)


def test_action_to_weights_renormalizes_drifted_vector() -> None:
    a = torch.tensor([0.5001, 0.3001, 0.2001])  # sums to 1.0003
    w = _action_to_weights(a)
    assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-9


def test_action_to_weights_handles_near_zero_components() -> None:
    a = torch.tensor([0.999, 1e-10, 1e-10])
    w = _action_to_weights(a)
    assert w.alpha > 0.99
    # No exception even with extreme values.


# ---------- compute_gae ----------

def test_compute_gae_single_step_collapses_to_r_minus_v() -> None:
    rewards = torch.tensor([1.0, 0.5, 0.0])
    values = torch.tensor([0.4, 0.4, 0.4])
    advantages, returns = compute_gae(rewards, values, gamma=0.99, lambda_=0.95)
    assert torch.allclose(advantages, rewards - values)
    assert torch.allclose(returns, rewards)


# ---------- collect_rollout ----------

def test_collect_rollout_shapes() -> None:
    config = PPOConfig(rollout_size=8)
    trainer = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=100,
        config=config,
    )
    rollout = trainer.collect_rollout()
    assert isinstance(rollout, Rollout)
    assert rollout.states.shape == (8, 18)
    assert rollout.actions.shape == (8, 3)
    assert rollout.log_probs.shape == (8,)
    assert rollout.rewards.shape == (8,)
    assert rollout.values.shape == (8,)
    assert len(rollout.query_indices) == 8


def test_collect_rollout_actions_on_simplex() -> None:
    trainer = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=10,
    )
    rollout = trainer.collect_rollout()
    sums = rollout.actions.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_collect_rollout_uses_provided_rng() -> None:
    """Two trainers with same RNG seed should produce identical query indices."""
    cfg = PPOConfig(rollout_size=4)
    t1 = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=100,
        config=cfg,
        rng=np.random.default_rng(7),
    )
    t2 = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=100,
        config=cfg,
        rng=np.random.default_rng(7),
    )
    r1 = t1.collect_rollout()
    r2 = t2.collect_rollout()
    assert r1.query_indices == r2.query_indices


# ---------- train_step ----------

def test_train_step_returns_metric_dict() -> None:
    trainer = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=100,
        config=PPOConfig(rollout_size=16, minibatch_size=8, update_epochs=2),
    )
    metrics = trainer.train_step()
    assert set(metrics.keys()) >= {
        "policy_loss", "value_loss", "entropy", "approx_kl", "mean_reward", "mean_value"
    }
    for v in metrics.values():
        assert np.isfinite(v)


def test_train_increases_mean_reward_on_alpha_dominant_problem() -> None:
    """Trainer should learn to push α toward 1 → mean_reward should rise."""
    trainer = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=64,
        config=PPOConfig(
            rollout_size=64,
            minibatch_size=16,
            update_epochs=4,
            learning_rate=1e-2,  # higher LR for fast convergence in tests
            entropy_coef=0.0,    # disable entropy bonus to push toward greedy α
        ),
        rng=np.random.default_rng(0),
    )
    history = trainer.train(total_episodes=30)
    early = float(np.mean([h["mean_reward"] for h in history[:5]]))
    late = float(np.mean([h["mean_reward"] for h in history[-5:]]))
    assert late > early, f"reward did not improve: early={early:.3f}, late={late:.3f}"


# ---------- TensorBoard writer ----------

class _CapturingWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.calls.append((tag, float(value), int(step)))


def test_writer_called_per_train_step() -> None:
    writer = _CapturingWriter()
    trainer = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=10,
        config=PPOConfig(rollout_size=8, minibatch_size=4, update_epochs=1),
        writer=writer,
    )
    trainer.train(total_episodes=2)
    # Each train_step writes ≥ 4 metrics; over 2 episodes expect ≥ 8 calls.
    assert len(writer.calls) >= 8
    tags = {tag for tag, _, _ in writer.calls}
    assert "train/mean_reward" in tags


# ---------- checkpoint ----------

def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    trainer = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=10,
        config=PPOConfig(rollout_size=4, minibatch_size=4, update_epochs=1),
    )
    trainer.train(total_episodes=1)
    saved_step = trainer._step
    saved_state = {k: v.clone() for k, v in trainer.ac.state_dict().items()}

    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt)

    # Build a fresh trainer (different init) and load.
    set_seed(999)
    fresh = PPOTrainer(
        actor_critic=ActorCritic(),
        state_provider=_zero_state,
        reward_fn=_alpha_dominant_reward,
        n_queries=10,
    )
    fresh.load_checkpoint(ckpt)
    assert fresh._step == saved_step
    for key, val in saved_state.items():
        assert torch.allclose(fresh.ac.state_dict()[key], val)


# ---------- config defaults ----------

def test_ppo_config_defaults_match_thesis_table_5_4() -> None:
    cfg = PPOConfig()
    assert cfg.learning_rate == 3e-4
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_ratio == 0.2
    assert cfg.value_coef == 0.5
    assert cfg.entropy_coef == 0.01
    assert cfg.max_grad_norm == 0.5
    assert cfg.gamma == 0.99
    assert cfg.rollout_size == 32
    assert cfg.update_epochs == 4
    assert cfg.minibatch_size == 8
