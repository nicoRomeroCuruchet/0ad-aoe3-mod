"""Shared contracts for policies, trainers, and their run data."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


Observation = NDArray[np.float32]
Action = NDArray[Any]


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        frozen = value.copy()
        frozen.setflags(write=False)
        return frozen
    return value


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_value(value) for key, value in values.items()}
    )


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@runtime_checkable
class Policy(Protocol):
    """A policy that can choose one action from one observation."""

    def act(
        self,
        observation: Observation,
        *,
        deterministic: bool,
    ) -> Action:
        """Choose an action without changing the environment."""


SolveEvaluator = Callable[[Policy], float]


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """The registered agent name and its implementation-specific parameters."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")

        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Why a training invocation stopped and how much work it completed."""

    stop_reason: str
    start_num_timesteps: int
    end_num_timesteps: int
    steps_this_run: int
    solve_checks: int = 0
    last_success_rate: float | None = None

    def __post_init__(self) -> None:
        if self.stop_reason not in {"solved", "safety_cap", "steps_complete"}:
            raise ValueError("stop_reason must describe a supported training outcome")
        for field_name in (
            "start_num_timesteps",
            "end_num_timesteps",
            "steps_this_run",
            "solve_checks",
        ):
            value = getattr(self, field_name)
            if not _is_integer(value) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.end_num_timesteps < self.start_num_timesteps:
            raise ValueError("end_num_timesteps cannot precede start_num_timesteps")
        if self.steps_this_run != self.end_num_timesteps - self.start_num_timesteps:
            raise ValueError("steps_this_run must equal the timestep difference")
        if self.last_success_rate is not None:
            if (
                isinstance(self.last_success_rate, bool)
                or not isinstance(self.last_success_rate, Real)
                or not math.isfinite(float(self.last_success_rate))
                or not 0.0 <= float(self.last_success_rate) <= 1.0
            ):
                raise ValueError("last_success_rate must be finite and in [0, 1]")
            object.__setattr__(
                self,
                "last_success_rate",
                float(self.last_success_rate),
            )


@dataclass(frozen=True, slots=True)
class TrainRequest:
    """Everything a trainer needs for one bounded training run."""

    env: Any
    agent: AgentSpec
    total_steps: int
    seed: int
    log_dir: Path | None = None
    log_interval: int = 1
    best_model_path: Path | None = None
    checkpoint_path: Path | None = None
    resume_from: Path | None = None
    solved_window_episodes: int | None = None
    solved_success_rate: float | None = None
    solved_min_steps: int = 0
    solved_check_interval_steps: int | None = None
    solve_evaluator: SolveEvaluator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent, AgentSpec):
            raise ValueError("agent must be an AgentSpec")
        if not _is_integer(self.total_steps) or self.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if not _is_integer(self.seed) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.log_dir is not None and not isinstance(self.log_dir, Path):
            raise ValueError("log_dir must be a pathlib.Path or None")
        if not _is_integer(self.log_interval) or self.log_interval <= 0:
            raise ValueError("log_interval must be a positive integer")
        if self.best_model_path is not None and not isinstance(
            self.best_model_path,
            Path,
        ):
            raise ValueError("best_model_path must be a pathlib.Path or None")
        if self.checkpoint_path is not None and not isinstance(
            self.checkpoint_path,
            Path,
        ):
            raise ValueError("checkpoint_path must be a pathlib.Path or None")
        if self.resume_from is not None and not isinstance(self.resume_from, Path):
            raise ValueError("resume_from must be a pathlib.Path or None")
        has_window = self.solved_window_episodes is not None
        has_success_rate = self.solved_success_rate is not None
        if has_window != has_success_rate:
            raise ValueError(
                "solved window and success rate must be configured together"
            )
        if has_window and (
            not _is_integer(self.solved_window_episodes)
            or self.solved_window_episodes <= 0
        ):
            raise ValueError("solved window must be a positive integer")
        if has_success_rate and (
            isinstance(self.solved_success_rate, bool)
            or not isinstance(self.solved_success_rate, Real)
            or not math.isfinite(float(self.solved_success_rate))
            or not 0.0 < float(self.solved_success_rate) <= 1.0
        ):
            raise ValueError("solved success rate must be finite and in (0, 1]")
        if not _is_integer(self.solved_min_steps) or self.solved_min_steps < 0:
            raise ValueError("solved minimum steps must be a non-negative integer")
        if not has_window and self.solved_min_steps:
            raise ValueError("solved minimum steps requires solved stopping")
        if has_window:
            if (
                not _is_integer(self.solved_check_interval_steps)
                or self.solved_check_interval_steps <= 0
            ):
                raise ValueError("solved check interval must be a positive integer")
            if not callable(self.solve_evaluator):
                raise ValueError("solved stopping requires a callable evaluator")
        elif self.solved_check_interval_steps is not None:
            raise ValueError("solved check interval requires solved stopping")
        elif self.solve_evaluator is not None:
            raise ValueError("solve evaluator requires solved stopping")
        if has_success_rate:
            object.__setattr__(
                self,
                "solved_success_rate",
                float(self.solved_success_rate),
            )


@runtime_checkable
class Trainer(Protocol):
    """An implementation that learns a policy from a training request."""

    def fit(self, request: TrainRequest) -> Policy:
        """Train for the requested budget and return an evaluation policy."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Algorithm-independent measurements from one completed episode."""

    episode: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    final_info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_integer(self.episode) or self.episode < 0:
            raise ValueError("episode must be a non-negative integer")
        if isinstance(self.total_reward, bool) or not isinstance(
            self.total_reward, Real
        ):
            raise ValueError("total_reward must be a real number")
        if not _is_integer(self.steps) or self.steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if not isinstance(self.terminated, bool):
            raise ValueError("terminated must be a boolean")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a boolean")
        if not isinstance(self.final_info, Mapping):
            raise ValueError("final_info must be a mapping")

        object.__setattr__(self, "total_reward", float(self.total_reward))
        object.__setattr__(self, "final_info", _freeze_mapping(self.final_info))
