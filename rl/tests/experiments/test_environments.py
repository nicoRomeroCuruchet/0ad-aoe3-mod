from types import MappingProxyType

import pytest

from rl.experiments.config import EnvironmentConfig
from rl.experiments.environments import (
    EnvironmentConfigError,
    UnknownEnvironmentError,
    build_environment,
)


def test_build_environment_forwards_scenario_and_parameters(monkeypatch):
    sentinel = object()
    calls = []

    def fake_make_gather_env(scenario, **parameters):
        calls.append((scenario, parameters))
        return sentinel

    monkeypatch.setattr(
        "rl.experiments.environments.make_gather_env",
        fake_make_gather_env,
    )
    config = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={
            "scenario": "rl/reset_config.json",
            "uri": "http://localhost:6000",
            "horizon": 25,
        },
    )

    environment = build_environment(config)

    assert environment is sentinel
    assert calls == [
        (
            "rl/reset_config.json",
            {"horizon": 25, "uri": "http://localhost:6000"},
        )
    ]
    assert isinstance(config.parameters, MappingProxyType)
    assert config.parameters["scenario"] == "rl/reset_config.json"


def test_build_environment_forwards_m1_stock_parameters(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rl.experiments.environments.make_gather_env",
        lambda scenario, **parameters: calls.append((scenario, parameters)) or object(),
    )
    config = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={
            "scenario": "rl/reset_config.json",
            "reward_mode": "stock_delta",
            "stock_resource": "wood",
            "stock_player": 1,
            "stock_success_threshold": 1.0,
            "gather_command_distance": 12.0,
            "agent_controls_click": True,
            "click_action_threshold": 0.0,
            "resource_state_observation": True,
            "carried_resource_observation_scale": 20.0,
            "stock_observation_scale": 1000.0,
            "distance_shaping_scale": 0.02,
            "gather_ready_reward": 0.25,
            "carried_resource_delta_reward_scale": 0.2,
            "gather_cycle_no_click_reward": 0.02,
            "carrying_no_click_reward": 0.05,
            "click_gather_cycle_penalty": 1.0,
            "backend_retries": 2,
            "backend_retry_delay": 0.25,
            "server_command": ["./run_game.sh", "--rl-interface=127.0.0.1:6000"],
            "server_startup_delay": 0.0,
        },
    )

    build_environment(config)

    assert calls == [
        (
            "rl/reset_config.json",
            {
                "agent_controls_click": True,
                "backend_retries": 2,
                "backend_retry_delay": 0.25,
                "carried_resource_observation_scale": 20.0,
                "click_action_threshold": 0.0,
                "click_gather_cycle_penalty": 1.0,
                "carried_resource_delta_reward_scale": 0.2,
                "distance_shaping_scale": 0.02,
                "gather_command_distance": 12.0,
                "gather_ready_reward": 0.25,
                "gather_cycle_no_click_reward": 0.02,
                "carrying_no_click_reward": 0.05,
                "resource_state_observation": True,
                "reward_mode": "stock_delta",
                "server_command": ["./run_game.sh", "--rl-interface=127.0.0.1:6000"],
                "server_startup_delay": 0.0,
                "stock_observation_scale": 1000.0,
                "stock_player": 1,
                "stock_resource": "wood",
                "stock_success_threshold": 1.0,
            },
        )
    ]


def test_build_environment_rejects_unknown_environment_names():
    config = EnvironmentConfig(name="typo")

    with pytest.raises(UnknownEnvironmentError, match="zero_ad_gather"):
        build_environment(config)


def test_gather_environment_requires_a_scenario_path():
    config = EnvironmentConfig(name="zero_ad_gather", parameters={"horizon": 5})

    with pytest.raises(EnvironmentConfigError, match="scenario"):
        build_environment(config)


def test_gather_environment_rejects_non_string_scenario_paths():
    config = EnvironmentConfig(name="zero_ad_gather", parameters={"scenario": 42})

    with pytest.raises(EnvironmentConfigError, match="non-empty path"):
        build_environment(config)


def test_gather_environment_rejects_remote_or_credentialed_uris_by_default():
    remote = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={
            "scenario": "rl/reset_config.json",
            "uri": "http://example.test:6000",
        },
    )
    credentialed = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={
            "scenario": "rl/reset_config.json",
            "uri": "http://user:password@localhost:6000",
        },
    )

    with pytest.raises(EnvironmentConfigError, match="remote server"):
        build_environment(remote)
    with pytest.raises(EnvironmentConfigError, match="credentials"):
        build_environment(credentialed, allow_remote=True)


def test_gather_environment_allows_an_explicit_remote_server_opt_in(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rl.experiments.environments.make_gather_env",
        lambda scenario, **parameters: calls.append((scenario, parameters)) or object(),
    )
    config = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={
            "scenario": "rl/reset_config.json",
            "uri": "http://192.0.2.10:6000",
        },
    )

    build_environment(config, allow_remote=True)

    assert calls == [("rl/reset_config.json", {"uri": "http://192.0.2.10:6000"})]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("horizon", 0),
        ("horizon", 1.5),
        ("sim_steps_per_action", -1),
        ("map_size_m", float("inf")),
        ("map_size_m", 0.0),
        ("reach_threshold", "near"),
        ("save_replay", 1),
        ("reward_mode", "closer"),
        ("stock_resource", ""),
        ("stock_player", 0),
        ("stock_success_threshold", 0.0),
        ("gather_command_distance", 0.0),
        ("agent_controls_gather", 1),
        ("gather_action_threshold", 1.5),
        ("agent_controls_click", 1),
        ("click_action_threshold", 1.5),
        ("resource_state_observation", 1),
        ("carried_resource_observation_scale", 0.0),
        ("stock_observation_scale", 0.0),
        ("distance_shaping_scale", -0.1),
        ("gather_ready_reward", -0.1),
        ("carried_resource_delta_reward_scale", -0.1),
        ("gather_cycle_no_click_reward", -0.1),
        ("carrying_no_click_reward", -0.1),
        ("click_gather_cycle_penalty", -0.1),
        ("backend_retries", -1),
        ("backend_retry_delay", -0.1),
        ("server_command", "./run_game.sh"),
        ("server_command", []),
        ("server_startup_delay", -0.1),
    ],
)
def test_gather_environment_validates_environment_specific_parameters(
    parameter,
    value,
):
    config = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={"scenario": "rl/reset_config.json", parameter: value},
    )

    with pytest.raises(EnvironmentConfigError, match=parameter):
        build_environment(config)


def test_gather_environment_rejects_unknown_parameters():
    config = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={"scenario": "rl/reset_config.json", "horzion": 50},
    )

    with pytest.raises(EnvironmentConfigError, match="horzion"):
        build_environment(config)


def test_gather_environment_bounds_the_combined_episode_workload():
    config = EnvironmentConfig(
        name="zero_ad_gather",
        parameters={
            "scenario": "rl/reset_config.json",
            "horizon": 100_000,
            "sim_steps_per_action": 10_000,
        },
    )

    with pytest.raises(EnvironmentConfigError, match="combined workload"):
        build_environment(config)
