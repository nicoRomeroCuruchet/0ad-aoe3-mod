"""Algorithm-independent training experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from rl.agents.base import AgentSpec, Policy, TrainRequest, Trainer
from rl.agents.registry import build_trainer

from .config import ExperimentConfig
from .evaluation import DecisionObserver, DecisionRecord, evaluate


TrainerBuilder = Callable[[AgentSpec], Trainer]


class DecisionObserverEnv(gym.Wrapper):
    """Report the observation and action immediately before each training step."""

    def __init__(self, env: gym.Env, observer: DecisionObserver):
        super().__init__(env)
        self._observer = observer
        self._episode = -1
        self._step = 0
        self._observation: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._episode += 1
        self._step = 0
        self._observation = np.asarray(observation).copy()
        return observation, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("environment must be reset before the first step")
        self._observer(
            DecisionRecord(
                episode=self._episode,
                step=self._step,
                observation=self._observation,
                action=action,
            )
        )
        transition = super().step(action)
        self._observation = np.asarray(transition[0]).copy()
        self._step += 1
        return transition


def train_policy(
    config: ExperimentConfig,
    env: Any,
    *,
    trainer_builder: TrainerBuilder = build_trainer,
    decision_observer: DecisionObserver | None = None,
    log_dir: Path | None = None,
    best_model_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
) -> Policy:
    """Train the configured agent against an already-created environment."""

    trainer = trainer_builder(config.agent)
    training_env = (
        env if decision_observer is None else DecisionObserverEnv(env, decision_observer)
    )
    solve_evaluator = None
    if config.training.solved_window_episodes is not None:

        def solve_evaluator(policy: Policy) -> float:
            report = evaluate(
                env,
                policy,
                episodes=config.training.solved_window_episodes,
                deterministic=True,
                seed=config.evaluation.seed,
                decision_observer=decision_observer,
            )
            return report.success_rate

    request = TrainRequest(
        env=training_env,
        agent=config.agent,
        total_steps=config.training.total_steps,
        seed=config.training.seed,
        log_dir=log_dir,
        log_interval=config.training.log_interval,
        best_model_path=best_model_path,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
        solved_window_episodes=config.training.solved_window_episodes,
        solved_success_rate=config.training.solved_success_rate,
        solved_min_steps=config.training.solved_min_steps,
        solved_check_interval_steps=config.training.solved_check_interval_steps,
        solve_evaluator=solve_evaluator,
    )
    return trainer.fit(request)
