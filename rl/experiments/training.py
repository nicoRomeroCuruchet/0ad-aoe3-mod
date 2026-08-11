"""Algorithm-independent training experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from rl.agents.base import AgentSpec, Policy, TrainRequest, Trainer
from rl.agents.registry import build_trainer

from rl.gather.live_view import LiveEpisodeView
from rl.gather.team_render import TeamRenderState

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
    live_view: LiveEpisodeView | None = None,
) -> Policy:
    """Train the configured agent against an already-created environment."""

    trainer = trainer_builder(config.agent)
    training_env = (
        env if decision_observer is None else DecisionObserverEnv(env, decision_observer)
    )
    solve_evaluator = None
    if config.training.solved_window_episodes is not None:

        def solve_evaluator(policy: Policy) -> float:
            # The live view watches only the first episode of the check, which
            # is enough to see what the policy is doing without slowing the
            # rest of the window down.
            watcher = None
            if live_view is not None:

                def watcher(record: DecisionRecord) -> None:
                    state = _render_state(env)
                    if state is not None:
                        live_view.observe(state)

            report = evaluate(
                env,
                policy,
                episodes=config.training.solved_window_episodes,
                deterministic=True,
                seed=config.evaluation.seed,
                decision_observer=_combine(decision_observer, watcher),
            )
            if live_view is not None:
                page = live_view.publish(
                    steps=_model_steps(policy),
                    summary=(
                        f"success {report.success_rate:.0%} "
                        f"| mean reward {report.mean_total_reward:.1f} "
                        f"| mean steps {report.mean_steps:.1f}"
                    ),
                )
                print(f"live_view: {page}", flush=True)
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


def _model_steps(policy: Policy) -> int:
    model = getattr(policy, "model", None)
    return int(getattr(model, "num_timesteps", 0))


def _render_state(env: Any) -> TeamRenderState | None:
    """Read the current scene from a team environment, if it is one."""

    roster = getattr(env, "_roster", None)
    if roster is None or roster.dropsite is None:
        return None
    targets = ()
    target_index = getattr(env, "_target_index", ())
    if target_index:
        targets = tuple(
            tuple(roster.resources[index].position()) for index in target_index
        )
    return TeamRenderState(
        villager_xz=tuple(tuple(unit.position()) for unit in roster.villagers),
        resource_xz=tuple(tuple(unit.position()) for unit in roster.resources),
        resource_remaining=(),
        carried=tuple(getattr(env, "_previous_carried", ())),
        dropsite_xz=tuple(roster.dropsite.position()),
        targets_xz=targets,
    )
