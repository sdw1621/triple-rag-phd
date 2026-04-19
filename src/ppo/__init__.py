"""PPO infrastructure for L-DWA training."""

from src.ppo.actor_critic import ActorCritic, PolicyOutput
from src.ppo.mdp import (
    ACTION_DIM,
    LATENCY_BUDGET_SEC,
    STATE_DIM,
    Action,
    State,
    build_state,
    compute_reward,
    extract_query_meta,
    extract_source_stats,
)
from src.ppo.trainer import PPOConfig, PPOTrainer, Rollout, compute_gae

__all__ = [
    "STATE_DIM",
    "ACTION_DIM",
    "LATENCY_BUDGET_SEC",
    "State",
    "Action",
    "compute_reward",
    "extract_source_stats",
    "extract_query_meta",
    "build_state",
    "ActorCritic",
    "PolicyOutput",
    "PPOConfig",
    "PPOTrainer",
    "Rollout",
    "compute_gae",
]
