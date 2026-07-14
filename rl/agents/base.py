"""Shared contracts for policies, trainers, and their run data."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


Observation = NDArray[np.float32]
Action = NDArray[np.float32]


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
class TrainRequest:
    """Everything a trainer needs for one bounded training run."""

    env: Any
    agent: AgentSpec
    total_steps: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.agent, AgentSpec):
            raise ValueError("agent must be an AgentSpec")
        if not _is_integer(self.total_steps) or self.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if not _is_integer(self.seed) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


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
