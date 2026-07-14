"""Algorithm-independent training experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rl.agents.base import AgentSpec, Policy, TrainRequest, Trainer
from rl.agents.registry import build_trainer

from .config import EnvironmentConfig, ExperimentConfig
from .environments import build_environment
from .evaluation import EvaluationReport, evaluate


EnvironmentBuilder = Callable[[EnvironmentConfig], Any]
TrainerBuilder = Callable[[AgentSpec], Trainer]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The reusable outputs of one completed train-and-evaluate run."""

    policy: Policy
    environment: Any
    evaluation: EvaluationReport


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


def run_training(
    config: ExperimentConfig,
    *,
    env: Any | None = None,
    environment_builder: EnvironmentBuilder = build_environment,
    trainer_builder: TrainerBuilder = build_trainer,
) -> TrainingResult:
    """Construct, train, and evaluate while depending only on shared contracts."""

    selected_env = env if env is not None else environment_builder(config.environment)
    policy = train_policy(
        config,
        selected_env,
        trainer_builder=trainer_builder,
    )
    report = evaluate(
        selected_env,
        policy,
        episodes=config.evaluation.episodes,
        deterministic=config.evaluation.deterministic,
        seed=config.evaluation.seed,
    )
    return TrainingResult(
        policy=policy,
        environment=selected_env,
        evaluation=report,
    )
