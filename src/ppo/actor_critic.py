"""
Actor-Critic policy network for PPO L-DWA.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 5 Section 2 (Figure 5-3, ~6K param network)

Architecture (Figure 5-3):
    state(18) → Linear(64) → Tanh → Linear(64) → Tanh → backbone features
        ↓ Actor head:  Linear(64 → 3) → Softplus → Dirichlet concentrations
        ↓ Critic head: Linear(64 → 1)        → state value

Key design choices:
- Shared backbone (parameter-efficient, ~5.6K params total).
- Softplus + 1e-6 ensures Dirichlet concentrations are strictly positive.
- Orthogonal init with gain=1.0 — initial policy is approximately uniform
  (1/3, 1/3, 1/3) on the simplex, matching the prior R-DWA equal-weight
  starting point so PPO begins from a sensible baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from src.ppo.mdp import ACTION_DIM, STATE_DIM

logger = logging.getLogger(__name__)

HIDDEN_DIM: int = 64
DIRICHLET_EPSILON: float = 1e-6  # added to Softplus output for numerical safety


@dataclass(frozen=True)
class PolicyOutput:
    """Per-state Actor-Critic output."""

    dirichlet_params: torch.Tensor  # (B, 3) strictly positive
    value: torch.Tensor             # (B, 1) state-value estimate


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic for the simplex action space.

    Args:
        state_dim: Input dim. Default 18 (thesis).
        hidden_dim: Backbone hidden dim. Default 64.
        action_dim: Output dim of the Dirichlet concentration head. Default 3.

    Example:
        >>> import torch
        >>> ac = ActorCritic()
        >>> state = torch.randn(4, 18)  # batch of 4
        >>> out = ac(state)
        >>> out.dirichlet_params.shape
        torch.Size([4, 3])
        >>> out.value.shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init (gain=1.0) on Linear layers; biases zero."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    # ---------- forward ----------

    def forward(self, state: torch.Tensor) -> PolicyOutput:
        """Single-state or batched forward pass.

        Args:
            state: ``(state_dim,)`` or ``(B, state_dim)`` float32 tensor.

        Returns:
            :class:`PolicyOutput` with batch dim added if input was 1-D.
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)
        features = self.backbone(state)
        # Softplus → strictly positive concentrations; +eps for safety.
        dirichlet_params = F.softplus(self.actor_head(features)) + DIRICHLET_EPSILON
        value = self.critic_head(features)
        return PolicyOutput(dirichlet_params=dirichlet_params, value=value)

    # ---------- inference helpers ----------

    @torch.no_grad()
    def act_mean(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-time deterministic action: Dirichlet mean.

        Returns:
            (action, value) where action is ``(B, 3)`` on the simplex.
        """
        out = self.forward(state)
        # Dirichlet mean: α_i / sum(α)
        action = out.dirichlet_params / out.dirichlet_params.sum(dim=-1, keepdim=True)
        return action, out.value

    def act_sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-time stochastic action: sample from Dirichlet.

        Returns:
            (action, log_prob, value).
        """
        out = self.forward(state)
        dist = Dirichlet(out.dirichlet_params)
        action = dist.rsample()  # reparameterized sample (differentiable)
        log_prob = dist.log_prob(action)
        return action, log_prob, out.value

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO update: log-prob, entropy, value of given (state, action).

        Args:
            state: ``(B, state_dim)``.
            action: ``(B, action_dim)`` on the simplex.

        Returns:
            (log_prob, entropy, value) each shape ``(B,)`` (or ``(B, 1)`` for value).
        """
        out = self.forward(state)
        dist = Dirichlet(out.dirichlet_params)
        # Numerical safety: clamp action away from simplex boundary.
        clamped = action.clamp(min=DIRICHLET_EPSILON)
        clamped = clamped / clamped.sum(dim=-1, keepdim=True)
        log_prob = dist.log_prob(clamped)
        entropy = dist.entropy()
        return log_prob, entropy, out.value

    # ---------- introspection ----------

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
