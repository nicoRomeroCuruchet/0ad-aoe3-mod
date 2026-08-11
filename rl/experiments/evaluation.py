"""Algorithm-independent policy evaluation on Gymnasium environments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from rl.agents.base import EpisodeResult, Policy, _freeze_mapping


class EvaluationEnv(Protocol):
    """The subset of the Gymnasium environment contract used here."""

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, Mapping[str, Any]]: ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]: ...


def _frozen_array(value: object) -> np.ndarray:
    frozen = np.asarray(value).copy()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """The exact policy input and output before the environment advances."""

    episode: int
    step: int
    observation: np.ndarray
    action: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", _frozen_array(self.observation))
        object.__setattr__(self, "action", _frozen_array(self.action))


DecisionObserver = Callable[[DecisionRecord], None]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One immutable transition emitted for visualization or diagnostics."""

    episode: int
    step: int
    observation: np.ndarray
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", _frozen_array(self.observation))
        object.__setattr__(self, "action", _frozen_array(self.action))
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "info", _freeze_mapping(self.info))


StepObserver = Callable[[StepRecord], None]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Immutable per-episode results with common aggregate metrics."""

    episodes: tuple[EpisodeResult, ...]

    def __post_init__(self) -> None:
        frozen_episodes = tuple(self.episodes)
        if not frozen_episodes:
            raise ValueError("an evaluation report needs at least one episode")
        if not all(isinstance(result, EpisodeResult) for result in frozen_episodes):
            raise ValueError("episodes must contain EpisodeResult values")
        object.__setattr__(self, "episodes", frozen_episodes)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def mean_total_reward(self) -> float:
        return fmean(result.total_reward for result in self.episodes)

    @property
    def mean_steps(self) -> float:
        return fmean(result.steps for result in self.episodes)

    @property
    def success_rate(self) -> float:
        """Fraction terminated successfully under the environment contract."""

        return fmean(result.terminated for result in self.episodes)


def run_episode(
    env: EvaluationEnv,
    policy: Policy,
    *,
    episode: int,
    deterministic: bool,
    seed: int | None = None,
    decision_observer: DecisionObserver | None = None,
    observer: StepObserver | None = None,
) -> EpisodeResult:
    """Run one episode and return its complete terminal summary."""

    if seed is None:
        observation, _ = env.reset()
    else:
        observation, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    final_info: Mapping[str, Any] = {}

    while not (terminated or truncated):
        policy_observation = np.asarray(observation).copy()
        action = policy.act(
            observation,
            deterministic=deterministic,
        )
        if decision_observer is not None:
            decision_observer(
                DecisionRecord(
                    episode=episode,
                    step=steps,
                    observation=policy_observation,
                    action=action,
                )
            )
        observation, reward, terminated, truncated, final_info = env.step(action)
        if observer is not None:
            observer(
                StepRecord(
                    episode=episode,
                    step=steps,
                    observation=policy_observation,
                    action=action,
                    reward=reward,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=final_info,
                )
            )
        total_reward += float(reward)
        steps += 1

    return EpisodeResult(
        episode=episode,
        total_reward=total_reward,
        steps=steps,
        terminated=bool(terminated),
        truncated=bool(truncated),
        final_info=final_info,
    )


def evaluate(
    env: EvaluationEnv,
    policy: Policy,
    *,
    episodes: int,
    deterministic: bool,
    seed: int | None = None,
    decision_observer: DecisionObserver | None = None,
    observer: StepObserver | None = None,
) -> EvaluationReport:
    """Evaluate one policy with consecutive, reproducible episode seeds."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    results = tuple(
        run_episode(
            env,
            policy,
            episode=episode,
            deterministic=deterministic,
            seed=None if seed is None else seed + episode,
            decision_observer=decision_observer,
            observer=observer,
        )
        for episode in range(episodes)
    )
    return EvaluationReport(episodes=results)
