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
