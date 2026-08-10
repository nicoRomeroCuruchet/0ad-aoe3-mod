from dataclasses import FrozenInstanceError

import pytest

from rl.experiments.config import ConfigError, load_experiment_config


VALID_CONFIG = """
[environment]
name = "zero_ad_gather"
scenario = "rl/reset_config.json"
horizon = 50
sim_steps_per_action = 10

[agent]
name = "custom_sac"
hidden_sizes = [256, 256]
learning_rate = 0.0003

[agent.optimizer]
name = "adam"
weight_decay = 0.0

[training]
total_steps = 10000
seed = 1
log_interval = 2

[evaluation]
episodes = 20
deterministic = true
seed = 1001
"""


def write_config(tmp_path, text=VALID_CONFIG):
    path = tmp_path / "experiment.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_experiment_config_builds_frozen_typed_sections(tmp_path):
    config = load_experiment_config(write_config(tmp_path))

    assert config.environment.name == "zero_ad_gather"
    assert config.environment.parameters["scenario"] == "rl/reset_config.json"
    assert config.environment.parameters["horizon"] == 50
    assert config.agent.name == "custom_sac"
    assert config.agent.parameters["hidden_sizes"] == (256, 256)
    assert config.agent.parameters["optimizer"]["name"] == "adam"
    assert config.training.total_steps == 10_000
    assert config.training.seed == 1
    assert config.training.log_interval == 2
    assert config.evaluation.episodes == 20
    assert config.evaluation.deterministic is True
    assert config.evaluation.seed == 1001

    with pytest.raises(FrozenInstanceError):
        config.training.seed = 2
    with pytest.raises(TypeError):
        config.environment.parameters["horizon"] = 60
    with pytest.raises(TypeError):
        config.agent.parameters["optimizer"]["name"] = "sgd"


def test_load_experiment_config_reports_invalid_toml(tmp_path):
    path = write_config(tmp_path, "[environment\nname = 'broken'")

    with pytest.raises(ConfigError, match="invalid TOML"):
        load_experiment_config(path)


def test_load_experiment_config_requires_every_section(tmp_path):
    text = VALID_CONFIG.replace("[evaluation]", "[not_evaluation]")

    with pytest.raises(ConfigError, match="missing required section 'evaluation'"):
        load_experiment_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("name = \"zero_ad_gather\"", "name = \"   \"", "environment.name"),
        ("name = \"custom_sac\"", "name = 7", "agent.name"),
        ("total_steps = 10000", "total_steps = true", "training.total_steps"),
        ("total_steps = 10000", "total_steps = 0", "training.total_steps"),
        ("seed = 1", "seed = -1", "training.seed"),
        ("log_interval = 2", "log_interval = 0", "training.log_interval"),
        ("episodes = 20", "episodes = 0", "evaluation.episodes"),
        ("deterministic = true", "deterministic = \"yes\"", "evaluation.deterministic"),
        ("seed = 1001", "seed = -1", "evaluation.seed"),
    ],
)
def test_load_experiment_config_rejects_invalid_values(
    tmp_path,
    old,
    new,
    message,
):
    text = VALID_CONFIG.replace(old, new, 1)

    with pytest.raises(ConfigError, match=message):
        load_experiment_config(write_config(tmp_path, text))


def test_load_experiment_config_rejects_unknown_training_and_evaluation_keys(tmp_path):
    text = VALID_CONFIG.replace(
        "total_steps = 10000",
        "total_steps = 10000\nstepz = 5",
    )

    with pytest.raises(ConfigError, match="unknown key 'training.stepz'"):
        load_experiment_config(write_config(tmp_path, text))


def test_load_experiment_config_defaults_training_log_interval(tmp_path):
    text = VALID_CONFIG.replace("log_interval = 2\n", "")

    config = load_experiment_config(write_config(tmp_path, text))

    assert config.training.log_interval == 1
