"""
PPO trainer for L-DWA.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 5 Section 3 (training algorithm), Table 5-4 (config)

The MDP is essentially single-step (one query → one weight choice → one
scalar reward), so GAE reduces to ``A = R - V`` and ``returns = R``. We keep
the GAE machinery in place so multi-step extensions (e.g. dialog turns) drop
in without rewriting the trainer.

Reward source is a ``reward_fn`` callable injected by the caller; in
production this wraps :class:`OfflineCache` so every PPO step is an O(1)
dict lookup instead of a $0.0003 LLM call. Tests pass closed-form synthetic
rewards.

TensorBoard logging is optional — passing ``writer=None`` disables it
(useful for unit tests and short dry-runs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch

from src.dwa.base import DWAWeights
from src.ppo.actor_critic import ActorCritic
from src.ppo.mdp import State

logger = logging.getLogger(__name__)


# ---------- config ----------

@dataclass
class PPOConfig:
    """Hyperparameters from thesis Table 5-4."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    rollout_size: int = 32
    update_epochs: int = 4
    minibatch_size: int = 8

    advantage_normalize: bool = True


# ---------- rollout ----------

@dataclass
class Rollout:
    """One batch of (state, action, log_prob, reward, value) tuples."""

    states: torch.Tensor       # (N, state_dim)
    actions: torch.Tensor      # (N, 3) — simplex
    log_probs: torch.Tensor    # (N,)
    rewards: torch.Tensor      # (N,)
    values: torch.Tensor       # (N,)
    query_indices: list[int]   # indices into the training query set

    def __len__(self) -> int:
        return self.states.shape[0]


# ---------- providers (injection points) ----------

class StateProvider(Protocol):
    """``provider(query_index) -> State``: build the 18-dim state for a query."""

    def __call__(self, query_index: int) -> State: ...


class RewardFn(Protocol):
    """``reward_fn(query_index, weights) -> reward``."""

    def __call__(self, query_index: int, weights: DWAWeights) -> float: ...


# ---------- helpers ----------

def _action_to_weights(action_row: torch.Tensor) -> DWAWeights:
    """Convert one ``(3,)`` simplex action to :class:`DWAWeights`.

    Renormalizes to compensate for float drift (Dirichlet samples are
    theoretically on the simplex but sum to 1 only up to float32 precision).
    """
    a = action_row.detach().cpu().double().numpy()
    a = np.clip(a, 1e-8, None)
    a = a / a.sum()
    return DWAWeights(float(a[0]), float(a[1]), float(a[2]))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al. 2016).

    For our 1-step setting (no terminal/next-state), this reduces to
    ``A = R - V`` and ``returns = R``. The general form is kept for
    forward-compatibility with multi-step extensions.

    Args:
        rewards: ``(N,)`` per-step rewards (assumed terminal at each step).
        values: ``(N,)`` value estimates.
        gamma: Discount factor.
        lambda_: GAE lambda.

    Returns:
        (advantages, returns), each ``(N,)``.
    """
    # Treat each rollout step as terminal → next_value = 0.
    deltas = rewards + gamma * 0.0 - values  # = rewards - values
    advantages = deltas  # 1-step, lambda has no effect
    returns = advantages + values
    # Reference uses of gamma/lambda_ — kept for the multi-step extension.
    _ = (gamma, lambda_)
    return advantages, returns


# ---------- writer protocol ----------

class _Writer(Protocol):
    """Subset of :class:`torch.utils.tensorboard.SummaryWriter` used here."""

    def add_scalar(self, tag: str, value: float, step: int) -> None: ...


# ---------- trainer ----------

class PPOTrainer:
    """PPO trainer with single-step MDP reduction and offline-cache rewards.

    Args:
        actor_critic: The policy network.
        state_provider: Callable returning a :class:`State` for a query index.
        reward_fn: Callable returning a scalar reward for (query_index, weights).
        n_queries: Total number of training queries; rollout draws indices
            uniformly in ``[0, n_queries)``.
        config: :class:`PPOConfig`. Defaults to thesis Table 5-4.
        device: PyTorch device string.
        writer: Optional TensorBoard ``SummaryWriter``. None disables logging.
        rng: Optional :class:`numpy.random.Generator` for reproducible
            rollout sampling.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        state_provider: StateProvider,
        reward_fn: RewardFn,
        n_queries: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
        writer: _Writer | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_queries <= 0:
            raise ValueError("n_queries must be > 0")
        self.ac = actor_critic.to(device)
        self.state_provider = state_provider
        self.reward_fn = reward_fn
        self.n_queries = n_queries
        self.config = config or PPOConfig()
        self.device = device
        self.writer = writer
        self.rng = rng or np.random.default_rng()
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(), lr=self.config.learning_rate
        )
        self._step: int = 0  # global PPO update step counter

    # ---------- rollout ----------

    def collect_rollout(self) -> Rollout:
        """Sample :attr:`PPOConfig.rollout_size` (state, action, reward) tuples."""
        idxs = self.rng.integers(0, self.n_queries, size=self.config.rollout_size).tolist()
        states = torch.stack(
            [self.state_provider(int(i)).to_tensor(self.device) for i in idxs]
        )
        self.ac.eval()
        with torch.no_grad():
            actions, log_probs, values = self.ac.act_sample(states)
        rewards: list[float] = []
        for i, action_row in zip(idxs, actions):
            weights = _action_to_weights(action_row)
            rewards.append(float(self.reward_fn(int(i), weights)))
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        return Rollout(
            states=states,
            actions=actions.detach(),
            log_probs=log_probs.detach(),
            rewards=rewards_t,
            values=values.detach().squeeze(-1),
            query_indices=idxs,
        )

    # ---------- update ----------

    def ppo_update(
        self,
        rollout: Rollout,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """Run :attr:`PPOConfig.update_epochs` epochs of PPO update.

        Returns mean per-epoch metrics: ``policy_loss``, ``value_loss``,
        ``entropy``, ``approx_kl``.
        """
        if self.config.advantage_normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(rollout)
        indices = np.arange(n)
        all_metrics: list[dict[str, float]] = []

        self.ac.train()
        for _ in range(self.config.update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, n, self.config.minibatch_size):
                mb = indices[start : start + self.config.minibatch_size]
                if len(mb) < 2:
                    continue  # advantage normalization needs ≥2 samples
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                mb_states = rollout.states[mb_t]
                mb_actions = rollout.actions[mb_t]
                mb_old_logp = rollout.log_probs[mb_t]
                mb_advantages = advantages[mb_t]
                mb_returns = returns[mb_t]

                new_logp, entropy, value = self.ac.evaluate_actions(mb_states, mb_actions)
                ratio = torch.exp(new_logp - mb_old_logp)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (mb_returns - value.squeeze(-1)).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                all_metrics.append(
                    {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_bonus.item()),
                        "approx_kl": float(approx_kl),
                    }
                )

        if not all_metrics:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        return {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}

    # ---------- top-level loops ----------

    def train_step(self) -> dict[str, float]:
        """One full PPO step: rollout → GAE → update. Returns metrics."""
        rollout = self.collect_rollout()
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            self.config.gamma,
            self.config.gae_lambda,
        )
        update_metrics = self.ppo_update(rollout, advantages, returns)
        update_metrics["mean_reward"] = float(rollout.rewards.mean().item())
        update_metrics["mean_value"] = float(rollout.values.mean().item())

        if self.writer is not None:
            for tag, value in update_metrics.items():
                self.writer.add_scalar(f"train/{tag}", value, self._step)
        self._step += 1
        return update_metrics

    def train(self, total_episodes: int) -> list[dict[str, float]]:
        """Run :paramref:`total_episodes` PPO steps. Returns per-step metrics."""
        history: list[dict[str, float]] = []
        for episode in range(total_episodes):
            metrics = self.train_step()
            history.append(metrics)
            if (episode + 1) % 100 == 0:
                logger.info(
                    "episode %d/%d  mean_R=%.3f  policy_loss=%.4f  value_loss=%.4f",
                    episode + 1, total_episodes,
                    metrics["mean_reward"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                )
        return history

    # ---------- checkpoint ----------

    def save_checkpoint(self, path: Path) -> None:
        """Save policy weights, optimizer state, and step counter."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_critic": self.ac.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self._step,
                "config": self.config.__dict__,
            },
            str(path),
        )

    def load_checkpoint(self, path: Path) -> None:
        """Restore a saved checkpoint into the current trainer."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.ac.load_state_dict(ckpt["actor_critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = int(ckpt.get("step", 0))
