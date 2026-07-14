from pathlib import Path

import pytest

from rl.gather.env import ZeroADGatherEnv
from rl.gather.factory import make_gather_env


def test_factory_reads_scenario_and_forwards_environment_parameters(
    tmp_path, fake_backend
):
    game, actions = fake_backend
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text('{"settings": {"mapName": "M0"}}', encoding="utf-8")

    env = make_gather_env(
        scenario_path,
        uri="http://example.invalid:1234",
        map_size_m=256.0,
        horizon=17,
        save_replay=True,
        game=game,
        actions=actions,
    )

    assert isinstance(env, ZeroADGatherEnv)
    assert env.scenario_config == '{"settings": {"mapName": "M0"}}'
    assert env.map_size_m == 256.0
    assert env.horizon == 17
    assert env.save_replay is True
    assert env.game is game
    assert env.actions is actions


def test_factory_accepts_a_string_path(tmp_path, fake_backend):
    game, actions = fake_backend
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text("{}", encoding="utf-8")

    env = make_gather_env(str(scenario_path), game=game, actions=actions)

    assert env.scenario_config == "{}"


def test_factory_propagates_a_missing_scenario_error(tmp_path):
    missing_path = Path(tmp_path) / "missing.json"

    with pytest.raises(FileNotFoundError, match="missing.json"):
        make_gather_env(missing_path)


def test_factory_is_available_from_the_package_without_eager_backend_loading():
    import rl.gather as gather

    assert gather.make_gather_env is make_gather_env
    assert "make_gather_env" in dir(gather)


def test_package_rejects_unknown_lazy_exports():
    import rl.gather as gather

    with pytest.raises(AttributeError, match="unknown_export"):
        gather.unknown_export
