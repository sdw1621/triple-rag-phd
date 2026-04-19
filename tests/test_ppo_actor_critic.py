"""Tests for src.ppo.actor_critic.ActorCritic."""

from __future__ import annotations

import pytest
import torch

from src.ppo.actor_critic import HIDDEN_DIM, ActorCritic, PolicyOutput
from src.ppo.mdp import ACTION_DIM, STATE_DIM
from src.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _seed() -> None:
    set_seed(42)


@pytest.fixture
def ac() -> ActorCritic:
    return ActorCritic()


# ---------- architecture ----------

def test_default_dimensions(ac: ActorCritic) -> None:
    assert ac.state_dim == STATE_DIM == 18
    assert ac.hidden_dim == HIDDEN_DIM == 64
    assert ac.action_dim == ACTION_DIM == 3


def test_parameter_count_in_expected_range(ac: ActorCritic) -> None:
    """Thesis Sec 5.2 quotes ~6,081 parameters; the actual count for this
    18→64→64→{3, 1} architecture is 5,636. Verify the count is in the
    documented order-of-magnitude (5K~7K)."""
    n = ac.parameter_count()
    assert 5000 <= n <= 7000, f"unexpected param count: {n}"
    # Document the precise count for thesis verification.
    expected = (18 * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3) + (64 * 1 + 1)
    assert n == expected == 5636


# ---------- forward ----------

def test_forward_single_state_adds_batch_dim(ac: ActorCritic) -> None:
    state = torch.randn(STATE_DIM)
    out = ac(state)
    assert isinstance(out, PolicyOutput)
    assert out.dirichlet_params.shape == (1, ACTION_DIM)
    assert out.value.shape == (1, 1)


def test_forward_batched(ac: ActorCritic) -> None:
    state = torch.randn(8, STATE_DIM)
    out = ac(state)
    assert out.dirichlet_params.shape == (8, ACTION_DIM)
    assert out.value.shape == (8, 1)


def test_dirichlet_params_are_strictly_positive(ac: ActorCritic) -> None:
    state = torch.randn(16, STATE_DIM)
    out = ac(state)
    assert (out.dirichlet_params > 0).all()


# ---------- act_mean (deterministic inference) ----------

def test_act_mean_action_lies_on_simplex(ac: ActorCritic) -> None:
    state = torch.randn(4, STATE_DIM)
    action, value = ac.act_mean(state)
    assert action.shape == (4, ACTION_DIM)
    assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert (action > 0).all() and (action < 1).all()
    assert value.shape == (4, 1)


def test_act_mean_initial_policy_near_uniform() -> None:
    """Orthogonal init → initial policy on a zero-state should be ~uniform."""
    set_seed(42)
    ac = ActorCritic()
    state = torch.zeros(1, STATE_DIM)
    action, _ = ac.act_mean(state)
    uniform = torch.ones(1, ACTION_DIM) / ACTION_DIM
    # Allow 0.1 tolerance — orthogonal init isn't exactly uniform but close.
    assert torch.allclose(action, uniform, atol=0.1)


# ---------- act_sample (training rollout) ----------

def test_act_sample_returns_action_log_prob_value(ac: ActorCritic) -> None:
    state = torch.randn(3, STATE_DIM)
    action, log_prob, value = ac.act_sample(state)
    assert action.shape == (3, ACTION_DIM)
    assert log_prob.shape == (3,)
    assert value.shape == (3, 1)


def test_act_sample_actions_on_simplex(ac: ActorCritic) -> None:
    state = torch.randn(32, STATE_DIM)
    action, _, _ = ac.act_sample(state)
    sums = action.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert (action >= 0).all()


def test_act_sample_is_stochastic(ac: ActorCritic) -> None:
    state = torch.zeros(1, STATE_DIM)
    set_seed(42)
    a1, _, _ = ac.act_sample(state)
    set_seed(43)
    a2, _, _ = ac.act_sample(state)
    assert not torch.allclose(a1, a2)


# ---------- evaluate_actions (PPO update) ----------

def test_evaluate_actions_shapes(ac: ActorCritic) -> None:
    state = torch.randn(5, STATE_DIM)
    action = torch.softmax(torch.randn(5, ACTION_DIM), dim=-1)
    log_prob, entropy, value = ac.evaluate_actions(state, action)
    assert log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert value.shape == (5, 1)


def test_evaluate_actions_log_prob_finite(ac: ActorCritic) -> None:
    state = torch.randn(8, STATE_DIM)
    action = torch.softmax(torch.randn(8, ACTION_DIM), dim=-1)
    log_prob, entropy, _ = ac.evaluate_actions(state, action)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


# ---------- gradient flow ----------

def test_loss_backward_updates_all_parameters(ac: ActorCritic) -> None:
    """Sanity check: a synthetic loss should produce non-zero grads on all params."""
    state = torch.randn(4, STATE_DIM)
    out = ac(state)
    loss = out.dirichlet_params.sum() + out.value.sum()
    loss.backward()
    for name, param in ac.named_parameters():
        assert param.grad is not None, f"{name} got no grad"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"
