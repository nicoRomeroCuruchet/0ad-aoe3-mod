"""Algorithm-independent training experiment orchestration."""

from __future__ import annotations

from typing import Any, Callable

from rl.agents.base import AgentSpec, Policy, TrainRequest, Trainer
from rl.agents.registry import build_trainer

from .config import ExperimentConfig


TrainerBuilder = Callable[[AgentSpec], Trainer]


def train_policy(
    config: ExperimentConfig,
    env: Any,
    *,
    trainer_builder: TrainerBuilder = build_trainer,
) -> Policy:
    """Train the configured agent against an already-created environment."""

    trainer = trainer_builder(config.agent)
    request = TrainRequest(
        env=env,
        agent=config.agent,
        total_steps=config.training.total_steps,
        seed=config.training.seed,
    )
    return trainer.fit(request)
