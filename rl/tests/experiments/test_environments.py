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
