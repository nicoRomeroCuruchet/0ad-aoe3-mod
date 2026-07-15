from pathlib import Path

from rl.experiments.config import load_experiment_config


CONFIG_DIRECTORY = Path(__file__).resolve().parents[2] / "configs"


def test_all_tracked_experiment_configs_are_valid_and_comparable():
    paths = sorted(CONFIG_DIRECTORY.glob("*.toml"))

    assert {
        "m0_oracle.toml",
        "m0_random.toml",
        "m0_sb3_sac.toml",
    }.issubset(path.name for path in paths)

    configs_by_name = {path.name: load_experiment_config(path) for path in paths}
    starter_configs = [
        configs_by_name[name]
        for name in ("m0_oracle.toml", "m0_random.toml", "m0_sb3_sac.toml")
    ]
    assert {
        "oracle",
        "random",
        "sb3_sac",
    } == {config.agent.name for config in starter_configs}
    assert {config.environment.name for config in starter_configs} == {"zero_ad_gather"}
    assert {
        config.environment.parameters["scenario"] for config in starter_configs
    } == {"rl/reset_config.json"}
    assert {config.evaluation.seed for config in starter_configs} == {1000}
