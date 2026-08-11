"""Typed loading and validation for tracked experiment TOML files."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping
import tomllib

from rl.agents.base import AgentSpec, _freeze_mapping


class ConfigError(ValueError):
    """Raised when an experiment file does not satisfy the config contract."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_non_empty_string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _require_positive_integer(value: object, key: str) -> int:
    if not _is_integer(value) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")
    return value


def _require_non_negative_integer(value: object, key: str) -> int:
    if not _is_integer(value) or value < 0:
        raise ConfigError(f"{key} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Environment registry key and constructor parameters."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "environment.name")
        if not isinstance(self.parameters, Mapping):
            raise ConfigError("environment parameters must be a mapping")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Common training budget shared by all trainer implementations."""

    total_steps: int
    seed: int
    log_interval: int = 1
    solved_window_episodes: int | None = None
    solved_success_rate: float | None = None
    solved_min_steps: int = 0
    solved_check_interval_steps: int | None = None

    def __post_init__(self) -> None:
        _require_positive_integer(self.total_steps, "training.total_steps")
        _require_non_negative_integer(self.seed, "training.seed")
        _require_positive_integer(self.log_interval, "training.log_interval")
        has_window = self.solved_window_episodes is not None
        has_success_rate = self.solved_success_rate is not None
        if has_window != has_success_rate:
            raise ConfigError(
                "training.solved_window_episodes and "
                "training.solved_success_rate must be configured together"
            )
        if has_window:
            _require_positive_integer(
                self.solved_window_episodes,
                "training.solved_window_episodes",
            )
        if has_success_rate and (
            isinstance(self.solved_success_rate, bool)
            or not isinstance(self.solved_success_rate, Real)
            or not math.isfinite(float(self.solved_success_rate))
            or not 0.0 < float(self.solved_success_rate) <= 1.0
        ):
            raise ConfigError(
                "training.solved_success_rate must be finite and in (0, 1]"
            )
        _require_non_negative_integer(
            self.solved_min_steps,
            "training.solved_min_steps",
        )
        if not has_window and self.solved_min_steps:
            raise ConfigError(
                "training.solved_min_steps requires solved stopping"
            )
        if has_window:
            if self.solved_check_interval_steps is None:
                raise ConfigError(
                    "training solved stopping requires a check interval"
                )
            _require_positive_integer(
                self.solved_check_interval_steps,
                "training.solved_check_interval_steps",
            )
        elif self.solved_check_interval_steps is not None:
            raise ConfigError(
                "training.solved_check_interval_steps requires solved stopping"
            )
        if has_success_rate:
            object.__setattr__(
                self,
                "solved_success_rate",
                float(self.solved_success_rate),
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Common policy-evaluation settings."""

    episodes: int
    deterministic: bool
    seed: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.episodes, "evaluation.episodes")
        if not isinstance(self.deterministic, bool):
            raise ConfigError("evaluation.deterministic must be a boolean")
        _require_non_negative_integer(self.seed, "evaluation.seed")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The complete, algorithm-independent experiment definition."""

    environment: EnvironmentConfig
    agent: AgentSpec
    training: TrainingConfig
    evaluation: EvaluationConfig


def _require_section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in data:
        raise ConfigError(f"missing required section '{name}'")

    section = data[name]
    if not isinstance(section, Mapping):
        raise ConfigError(f"section '{name}' must be a table")
    return section


def _require_key(section: Mapping[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ConfigError(f"missing required key '{section_name}.{key}'")
    return section[key]


def _reject_unknown_keys(
    section: Mapping[str, Any],
    section_name: str,
    expected: frozenset[str],
) -> None:
    unknown = section.keys() - expected
    if unknown:
        key = sorted(unknown)[0]
        raise ConfigError(f"unknown key '{section_name}.{key}'")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one TOML experiment file and validate its shared contract."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{config_path}: invalid TOML: {error}") from error

    environment_data = _require_section(data, "environment")
    agent_data = _require_section(data, "agent")
    training_data = _require_section(data, "training")
    evaluation_data = _require_section(data, "evaluation")

    environment_name = _require_non_empty_string(
        _require_key(environment_data, "environment", "name"),
        "environment.name",
    )
    agent_name = _require_non_empty_string(
        _require_key(agent_data, "agent", "name"),
        "agent.name",
    )

    _reject_unknown_keys(
        training_data,
        "training",
        frozenset(
            {
                "total_steps",
                "seed",
                "log_interval",
                "solved_window_episodes",
                "solved_success_rate",
                "solved_min_steps",
                "solved_check_interval_steps",
            }
        ),
    )
    _reject_unknown_keys(
        evaluation_data,
        "evaluation",
        frozenset({"episodes", "deterministic", "seed"}),
    )

    environment_parameters = {
        key: value for key, value in environment_data.items() if key != "name"
    }
    agent_parameters = {
        key: value for key, value in agent_data.items() if key != "name"
    }

    return ExperimentConfig(
        environment=EnvironmentConfig(
            name=environment_name,
            parameters=environment_parameters,
        ),
        agent=AgentSpec(name=agent_name, parameters=agent_parameters),
        training=TrainingConfig(
            total_steps=_require_key(training_data, "training", "total_steps"),
            seed=_require_key(training_data, "training", "seed"),
            log_interval=training_data.get("log_interval", 1),
            solved_window_episodes=training_data.get("solved_window_episodes"),
            solved_success_rate=training_data.get("solved_success_rate"),
            solved_min_steps=training_data.get("solved_min_steps", 0),
            solved_check_interval_steps=training_data.get(
                "solved_check_interval_steps"
            ),
        ),
        evaluation=EvaluationConfig(
            episodes=_require_key(evaluation_data, "evaluation", "episodes"),
            deterministic=_require_key(
                evaluation_data,
                "evaluation",
                "deterministic",
            ),
            seed=_require_key(evaluation_data, "evaluation", "seed"),
        ),
    )
