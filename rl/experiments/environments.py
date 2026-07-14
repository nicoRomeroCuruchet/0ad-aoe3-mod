"""Explicit construction of environments named by experiment configs."""

from __future__ import annotations

from typing import Any, Mapping

from rl.gather.factory import make_gather_env

from .config import EnvironmentConfig


class UnknownEnvironmentError(ValueError):
    """Raised when a config names an environment absent from this repo."""


class EnvironmentConfigError(ValueError):
    """Raised when environment-specific parameters are invalid."""


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def build_environment(config: EnvironmentConfig) -> Any:
    """Construct the environment described by a validated config."""

    if config.name != "zero_ad_gather":
        raise UnknownEnvironmentError(
            f"unknown environment '{config.name}'; available environments: "
            "zero_ad_gather",
        )

    parameters = _thaw(config.parameters)
    scenario = parameters.pop("scenario", None)
    if not isinstance(scenario, str) or not scenario.strip():
        raise EnvironmentConfigError(
            "zero_ad_gather requires 'scenario' as a non-empty path",
        )
    return make_gather_env(scenario, **parameters)
