"""
Learned Dynamic Weighting Algorithm (L-DWA) — PPO-trained policy.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 5 (CORE contribution — replaces R-DWA)

Inference path:
    query + intent  →  StateProvider  →  18-dim State
    State           →  ActorCritic.act_mean  →  Dirichlet mean (α, β, γ)
    (α, β, γ)       →  DWAWeights (simplex-validated)

The state provider is injected so callers control how source statistics are
gathered (typically: run a quick retrieval probe before deciding weights, or
read cached statistics from a previous turn).

For training-time use, see :mod:`src.ppo.trainer.PPOTrainer`. This module is
strictly the inference-time `BaseDWA` implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch

from src.dwa.base import BaseDWA, DWAWeights
from src.intent.rule_based import QueryIntent
from src.ppo.actor_critic import ActorCritic
from src.ppo.mdp import State

logger = logging.getLogger(__name__)


# (query, intent) → 18-dim State.  This is the same callable shape used by
# PPOTrainer's StateProvider but specialised for inference (takes the raw
# query string + intent rather than a query index).
StateBuilder = Callable[[str, QueryIntent], State]


class LearnedDWA(BaseDWA):
    """L-DWA: PPO-trained Actor-Critic projected to deterministic Dirichlet mean.

    Args:
        actor_critic: Trained :class:`ActorCritic`. Will be set to ``eval()``.
        state_builder: Callable producing the 18-dim :class:`State` for a
            (query, intent) pair. Typically wraps a fast retrieval probe.
        device: Device to move the policy to. Default "cpu".

    Example:
        >>> ac = ActorCritic()  # trained
        >>> def builder(query, intent):
        ...     return State(
        ...         density=intent.density,
        ...         intent_logits=(0.0, 0.0, 0.0),
        ...         source_stats=tuple([0.0] * 9),
        ...         query_meta=(0.0, 0.0, 0.0),
        ...     )
        >>> ldwa = LearnedDWA(ac, builder)
        >>> w = ldwa.compute("test", QueryIntent(  # doctest: +SKIP
        ...     query_type="simple", entities=[], relations=[], constraints=[],
        ...     complexity_score=0.0, density=(0.0, 0.0, 0.0)))
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        state_builder: StateBuilder,
        device: str = "cpu",
    ) -> None:
        self.ac = actor_critic.to(device)
        self.ac.eval()
        self.state_builder = state_builder
        self.device = device

    # ---------- BaseDWA ----------

    def compute(self, query: str, intent: QueryIntent) -> DWAWeights:
        """Run the trained policy and return the deterministic Dirichlet-mean weight."""
        state = self.state_builder(query, intent)
        state_tensor = state.to_tensor(self.device)
        action, _ = self.ac.act_mean(state_tensor)  # (1, 3)
        # Renormalize to compensate for float drift (Dirichlet mean is exact
        # in theory but the divide can introduce float32 noise).
        a = action[0].detach().cpu().double()
        a = (a / a.sum()).tolist()
        return DWAWeights(float(a[0]), float(a[1]), float(a[2]))

    # ---------- factories ----------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        state_builder: StateBuilder,
        device: str = "cpu",
    ) -> "LearnedDWA":
        """Load a PPOTrainer checkpoint into an inference-only LearnedDWA.

        The checkpoint format is what :meth:`PPOTrainer.save_checkpoint`
        writes — only the ``actor_critic`` state dict is consumed here.
        """
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        ac = ActorCritic()
        ac.load_state_dict(ckpt["actor_critic"])
        return cls(actor_critic=ac, state_builder=state_builder, device=device)
