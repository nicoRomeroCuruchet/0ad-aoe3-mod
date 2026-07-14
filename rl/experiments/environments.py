"""Explicit construction of environments named by experiment configs."""

from __future__ import annotations

import ipaddress
import math
from numbers import Real
from typing import Any, Mapping
from urllib.parse import urlsplit

from rl.gather.factory import make_gather_env

from .config import EnvironmentConfig


class UnknownEnvironmentError(ValueError):
    """Raised when a config names an environment absent from this repo."""


class EnvironmentConfigError(ValueError):
    """Raised when environment-specific parameters are invalid."""


_GATHER_PARAMETERS = frozenset(
    {
        "scenario",
        "uri",
        "map_size_m",
        "horizon",
        "reach_threshold",
        "sim_steps_per_action",
        "save_replay",
    }
)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _validate_positive_integer(name: str, value: Any, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > maximum
    ):
        raise EnvironmentConfigError(
            f"{name} must be a positive integer no greater than {maximum}",
        )


def _validate_positive_number(name: str, value: Any, maximum: float) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
        or value > maximum
    ):
        raise EnvironmentConfigError(
            f"{name} must be a finite positive number no greater than {maximum:g}",
        )


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_uri(value: Any, *, allow_remote: bool) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EnvironmentConfigError("uri must be a non-empty HTTP URL")

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise EnvironmentConfigError("uri must be an absolute HTTP URL")
    if parsed.username is not None or parsed.password is not None:
        raise EnvironmentConfigError("uri must not contain credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise EnvironmentConfigError("uri contains an invalid port") from error
    if port is None:
        raise EnvironmentConfigError("uri must include an explicit port")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise EnvironmentConfigError("uri must not contain a path, query, or fragment")
    if not allow_remote and not _is_loopback_host(parsed.hostname):
        raise EnvironmentConfigError(
            "remote server URI rejected; use the explicit remote-server opt-in",
        )


def _validate_gather_parameters(
    parameters: Mapping[str, Any],
    *,
    allow_remote: bool,
) -> None:
    unknown = parameters.keys() - _GATHER_PARAMETERS
    if unknown:
        key = sorted(unknown)[0]
        raise EnvironmentConfigError(f"unknown zero_ad_gather parameter '{key}'")

    if "uri" in parameters:
        _validate_uri(parameters["uri"], allow_remote=allow_remote)
    if "horizon" in parameters:
        _validate_positive_integer("horizon", parameters["horizon"], 100_000)
    if "sim_steps_per_action" in parameters:
        _validate_positive_integer(
            "sim_steps_per_action",
            parameters["sim_steps_per_action"],
            10_000,
        )
    if "map_size_m" in parameters:
        _validate_positive_number("map_size_m", parameters["map_size_m"], 1_000_000)
    if "reach_threshold" in parameters:
        _validate_positive_number(
            "reach_threshold",
            parameters["reach_threshold"],
            1_000_000,
        )
    if "save_replay" in parameters and not isinstance(parameters["save_replay"], bool):
        raise EnvironmentConfigError("save_replay must be a boolean")

    horizon = parameters.get("horizon", 50)
    simulation_steps = parameters.get("sim_steps_per_action", 10)
    if horizon * simulation_steps > 1_000_000:
        raise EnvironmentConfigError(
            "horizon and sim_steps_per_action exceed the combined workload limit",
        )


def build_environment(
    config: EnvironmentConfig,
    *,
    allow_remote: bool = False,
) -> Any:
    """Construct the environment described by a validated config."""

    if config.name != "zero_ad_gather":
        raise UnknownEnvironmentError(
            f"unknown environment '{config.name}'; available environments: "
            "zero_ad_gather",
        )

    parameters = _thaw(config.parameters)
    _validate_gather_parameters(parameters, allow_remote=allow_remote)
    scenario = parameters.pop("scenario", None)
    if not isinstance(scenario, str) or not scenario.strip():
        raise EnvironmentConfigError(
            "zero_ad_gather requires 'scenario' as a non-empty path",
        )
    return make_gather_env(scenario, **parameters)
