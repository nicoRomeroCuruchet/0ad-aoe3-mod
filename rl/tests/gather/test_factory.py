import os
from pathlib import Path

import pytest

import rl.gather.factory as factory_module
from rl.gather.env import ZeroADGatherEnv
from rl.gather.factory import make_gather_env


@pytest.fixture(autouse=True)
def use_temporary_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(factory_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(factory_module, "SCENARIO_ROOT", tmp_path)


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


def test_factory_rejects_scenarios_outside_the_repo(tmp_path, monkeypatch):
    allowed_root = tmp_path / "repo"
    allowed_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(factory_module, "SCENARIO_ROOT", allowed_root)

    with pytest.raises(ValueError, match="inside the repository"):
        make_gather_env(outside)


def test_factory_rejects_non_json_and_non_regular_files(tmp_path):
    text_file = tmp_path / "scenario.txt"
    text_file.write_text("{}", encoding="utf-8")
    fifo = tmp_path / "scenario.json"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="JSON"):
        make_gather_env(text_file)
    with pytest.raises(ValueError, match="regular file"):
        make_gather_env(fifo)


def test_factory_rejects_scenarios_over_the_size_limit(tmp_path, monkeypatch):
    scenario_path = tmp_path / "large.json"
    scenario_path.write_text("{" + " " * 32 + "}", encoding="utf-8")
    monkeypatch.setattr(factory_module, "MAX_SCENARIO_BYTES", 16)

    with pytest.raises(ValueError, match="too large"):
        make_gather_env(scenario_path)


def test_factory_rejects_malformed_json(tmp_path):
    scenario_path = tmp_path / "malformed.json"
    scenario_path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        make_gather_env(scenario_path)


def test_factory_is_available_from_the_package_without_eager_backend_loading():
    import rl.gather as gather

    assert gather.make_gather_env is make_gather_env
    assert "make_gather_env" in dir(gather)


def test_package_rejects_unknown_lazy_exports():
    import rl.gather as gather

    with pytest.raises(AttributeError, match="unknown_export"):
        gather.unknown_export
