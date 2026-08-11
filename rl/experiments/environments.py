"""Explicit construction of environments named by experiment configs."""

from __future__ import annotations

import ipaddress
import math
from numbers import Real
from typing import Any, Mapping
from urllib.parse import urlsplit

from rl.gather.factory import make_gather_env, make_team_gather_env

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
        "reward_mode",
        "stock_resource",
        "stock_player",
        "stock_success_threshold",
        "gather_command_distance",
        "agent_controls_gather",
        "gather_action_threshold",
        "agent_controls_click",
        "click_action_threshold",
        "resource_state_observation",
        "lifecycle_state_observation",
        "carried_resource_observation_scale",
        "stock_observation_scale",
        "distance_shaping_scale",
        "gather_ready_reward",
        "carried_resource_delta_reward_scale",
        "gather_cycle_no_click_reward",
        "carrying_no_click_reward",
        "click_gather_cycle_penalty",
        "backend_retries",
        "backend_retry_delay",
        "server_command",
        "server_startup_delay",
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


def _validate_non_negative_number(name: str, value: Any, maximum: float) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
        or value > maximum
    ):
        raise EnvironmentConfigError(
            f"{name} must be a finite non-negative number no greater than {maximum:g}",
        )


def _validate_bounded_number(name: str, value: Any, minimum: float, maximum: float) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < minimum
        or value > maximum
    ):
        raise EnvironmentConfigError(
            f"{name} must be a finite number between {minimum:g} and {maximum:g}",
        )


def _validate_non_negative_integer(name: str, value: Any, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > maximum
    ):
        raise EnvironmentConfigError(
            f"{name} must be a non-negative integer no greater than {maximum}",
        )


def _validate_non_empty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EnvironmentConfigError(f"{name} must be a non-empty string")


def _validate_server_command(value: Any) -> None:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(part, str) and part for part in value)
    ):
        raise EnvironmentConfigError(
            "server_command must be a non-empty list of strings",
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


_TEAM_GATHER_PARAMETERS = frozenset(
    {
        "scenario",
        "uri",
        "villager_count",
        "resource_count",
        "map_size_m",
        "horizon",
        "sim_steps_per_action",
        "gather_command_distance",
        "click_action_threshold",
        "stock_resource",
        "stock_player",
        "stock_success_threshold",
        "min_delivery_per_villager",
        "carried_resource_observation_scale",
        "stock_observation_scale",
        "resource_amount_scale",
        "distance_shaping_scale",
        "carried_resource_delta_reward_scale",
        "click_gather_cycle_penalty",
        "backend_retries",
        "backend_retry_delay",
        "save_replay",
    }
)


def _validate_team_gather_parameters(
    parameters: Mapping[str, Any],
    *,
    allow_remote: bool,
) -> None:
    unknown = parameters.keys() - _TEAM_GATHER_PARAMETERS
    if unknown:
        key = sorted(unknown)[0]
        raise EnvironmentConfigError(f"unknown zero_ad_team_gather parameter '{key}'")

    if "uri" in parameters:
        _validate_uri(parameters["uri"], allow_remote=allow_remote)
    for name, maximum in (("villager_count", 64), ("resource_count", 64)):
        if name in parameters:
            _validate_positive_integer(name, parameters[name], maximum)
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
    if "save_replay" in parameters and not isinstance(parameters["save_replay"], bool):
        raise EnvironmentConfigError("save_replay must be a boolean")


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
    if "reward_mode" in parameters and parameters["reward_mode"] not in {
        "distance_delta",
        "stock_delta",
    }:
        raise EnvironmentConfigError(
            "reward_mode must be 'distance_delta' or 'stock_delta'",
        )
    if "stock_resource" in parameters:
        _validate_non_empty_string("stock_resource", parameters["stock_resource"])
    if "stock_player" in parameters:
        _validate_positive_integer("stock_player", parameters["stock_player"], 16)
    if "stock_success_threshold" in parameters:
        _validate_positive_number(
            "stock_success_threshold",
            parameters["stock_success_threshold"],
            1_000_000,
        )
    if "gather_command_distance" in parameters:
        _validate_positive_number(
            "gather_command_distance",
            parameters["gather_command_distance"],
            1_000_000,
        )
    if "agent_controls_gather" in parameters and not isinstance(
        parameters["agent_controls_gather"],
        bool,
    ):
        raise EnvironmentConfigError("agent_controls_gather must be a boolean")
    if "gather_action_threshold" in parameters:
        _validate_bounded_number(
            "gather_action_threshold",
            parameters["gather_action_threshold"],
            -1.0,
            1.0,
        )
    if "agent_controls_click" in parameters and not isinstance(
        parameters["agent_controls_click"],
        bool,
    ):
        raise EnvironmentConfigError("agent_controls_click must be a boolean")
    if "click_action_threshold" in parameters:
        _validate_bounded_number(
            "click_action_threshold",
            parameters["click_action_threshold"],
            -1.0,
            1.0,
        )
    if "resource_state_observation" in parameters and not isinstance(
        parameters["resource_state_observation"],
        bool,
    ):
        raise EnvironmentConfigError("resource_state_observation must be a boolean")
    if "lifecycle_state_observation" in parameters and not isinstance(
        parameters["lifecycle_state_observation"],
        bool,
    ):
        raise EnvironmentConfigError("lifecycle_state_observation must be a boolean")
    if parameters.get("lifecycle_state_observation", False):
        if not parameters.get("resource_state_observation", False):
            raise EnvironmentConfigError(
                "lifecycle_state_observation requires resource_state_observation"
            )
        if parameters.get("reward_mode", "distance_delta") != "stock_delta":
            raise EnvironmentConfigError(
                "lifecycle_state_observation requires stock_delta reward mode"
            )
    if "carried_resource_observation_scale" in parameters:
        _validate_positive_number(
            "carried_resource_observation_scale",
            parameters["carried_resource_observation_scale"],
            1_000_000,
        )
    if "stock_observation_scale" in parameters:
        _validate_positive_number(
            "stock_observation_scale",
            parameters["stock_observation_scale"],
            1_000_000,
        )
    if "distance_shaping_scale" in parameters:
        _validate_non_negative_number(
            "distance_shaping_scale",
            parameters["distance_shaping_scale"],
            1_000_000,
        )
    if "gather_ready_reward" in parameters:
        _validate_non_negative_number(
            "gather_ready_reward",
            parameters["gather_ready_reward"],
            1_000_000,
        )
    if "carried_resource_delta_reward_scale" in parameters:
        _validate_non_negative_number(
            "carried_resource_delta_reward_scale",
            parameters["carried_resource_delta_reward_scale"],
            1_000_000,
        )
    if "gather_cycle_no_click_reward" in parameters:
        _validate_non_negative_number(
            "gather_cycle_no_click_reward",
            parameters["gather_cycle_no_click_reward"],
            1_000_000,
        )
    if "carrying_no_click_reward" in parameters:
        _validate_non_negative_number(
            "carrying_no_click_reward",
            parameters["carrying_no_click_reward"],
            1_000_000,
        )
    if "click_gather_cycle_penalty" in parameters:
        _validate_non_negative_number(
            "click_gather_cycle_penalty",
            parameters["click_gather_cycle_penalty"],
            1_000_000,
        )
    if "backend_retries" in parameters:
        _validate_non_negative_integer(
            "backend_retries",
            parameters["backend_retries"],
            100,
        )
    if "backend_retry_delay" in parameters:
        _validate_non_negative_number(
            "backend_retry_delay",
            parameters["backend_retry_delay"],
            3600,
        )
    if "server_command" in parameters:
        _validate_server_command(parameters["server_command"])
    if "server_startup_delay" in parameters:
        _validate_non_negative_number(
            "server_startup_delay",
            parameters["server_startup_delay"],
            3600,
        )

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

    if config.name not in {"zero_ad_gather", "zero_ad_team_gather"}:
        raise UnknownEnvironmentError(
            f"unknown environment '{config.name}'; available environments: "
            "zero_ad_gather, zero_ad_team_gather",
        )

    parameters = _thaw(config.parameters)
    if config.name == "zero_ad_team_gather":
        _validate_team_gather_parameters(parameters, allow_remote=allow_remote)
    else:
        _validate_gather_parameters(parameters, allow_remote=allow_remote)
    scenario = parameters.pop("scenario", None)
    if not isinstance(scenario, str) or not scenario.strip():
        raise EnvironmentConfigError(
            f"{config.name} requires 'scenario' as a non-empty path",
        )
    if config.name == "zero_ad_team_gather":
        return make_team_gather_env(scenario, **parameters)
    return make_gather_env(scenario, **parameters)
