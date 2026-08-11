"""Algorithm-independent training experiment orchestration."""

from __future__ import annotations

import time

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
    check_delay: float = 0.0,
    demo_env: Any = None,
) -> Policy:
    """Train the configured agent against an already-created environment."""

    trainer = trainer_builder(config.agent)
    training_env = (
        env if decision_observer is None else DecisionObserverEnv(env, decision_observer)
    )
    solve_evaluator = None
    if config.training.solved_window_episodes is not None:

        def solve_evaluator(policy: Policy) -> float:
            watcher = None

            report = evaluate(
                env,
                policy,
                episodes=config.training.solved_window_episodes,
                deterministic=True,
                seed=config.evaluation.seed,
                decision_observer=_combine(decision_observer, watcher),
            )
            if demo_env is not None:
                # One paced episode on the visual engine, so the run can be
                # watched without a renderer attached to headless training.
                # Pacing every simulation turn rather than every decision keeps the
                # motion smooth: a decision advances many turns at once, so
                # pausing between decisions looks like burst, freeze, burst.
                if check_delay > 0.0 and hasattr(demo_env, "sim_frame_observer"):
                    demo_env.sim_frame_observer = lambda: time.sleep(check_delay)

                def paced(record: DecisionRecord) -> None:
                    del record

                try:
                    demo = evaluate(
                        demo_env,
                        policy,
                        episodes=1,
                        deterministic=True,
                        seed=config.evaluation.seed,
                        decision_observer=paced,
                    )
                except Exception as error:  # noqa: BLE001 - watching must not kill training
                    # The demo server is a viewing convenience. Losing it must
                    # never end a training run that is otherwise healthy.
                    print(f"demo: unavailable ({type(error).__name__})", flush=True)
                else:
                    print(
                        "demo: "
                        f"reward={demo.mean_total_reward:.1f} "
                        f"steps={demo.mean_steps:.0f} "
                        f"success={demo.success_rate:.0%}",
                        flush=True,
                    )
                finally:
                    if hasattr(demo_env, "sim_frame_observer"):
                        demo_env.sim_frame_observer = None
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


def _combine(
    first: DecisionObserver | None,
    second: DecisionObserver | None,
) -> DecisionObserver | None:
    if first is None:
        return second
    if second is None:
        return first

    def combined(record: DecisionRecord) -> None:
        first(record)
        second(record)

    return combined
