from pathlib import Path

from rl.experiments.config import load_experiment_config


CONFIG_DIRECTORY = Path(__file__).resolve().parents[2] / "configs"


def test_all_tracked_experiment_configs_are_valid_and_comparable():
    paths = sorted(CONFIG_DIRECTORY.glob("*.toml"))

    assert {
        "m0_oracle.toml",
        "m0_random.toml",
        "m0_sb3_sac.toml",
        "m1_oracle.toml",
        "m1_random.toml",
        "m1_sb3_sac.toml",
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

    m1_configs = [
        configs_by_name[name]
        for name in ("m1_oracle.toml", "m1_random.toml", "m1_sb3_sac.toml")
    ]
    assert {
        "oracle",
        "random",
        "sb3_sac",
    } == {config.agent.name for config in m1_configs}
    assert {
        config.environment.parameters["reward_mode"] for config in m1_configs
    } == {"stock_delta"}
    assert {
        config.environment.parameters["stock_resource"] for config in m1_configs
    } == {"wood"}
    assert {
        config.environment.parameters["agent_controls_click"] for config in m1_configs
    } == {True}
    assert {
        config.environment.parameters["click_action_threshold"] for config in m1_configs
    } == {0.0}
    assert {
        config.environment.parameters["resource_state_observation"]
        for config in m1_configs
    } == {True}
    assert {
        config.environment.parameters["carried_resource_observation_scale"]
        for config in m1_configs
    } == {20.0}
    assert {
        config.environment.parameters["stock_observation_scale"] for config in m1_configs
    } == {1000.0}
    assert {
        config.environment.parameters["distance_shaping_scale"]
        for config in m1_configs
    } == {0.02}
    assert {
        config.environment.parameters["gather_ready_reward"] for config in m1_configs
    } == {0.25}
    assert {
        config.environment.parameters["carried_resource_delta_reward_scale"]
        for config in m1_configs
    } == {0.2}
    assert {
        config.environment.parameters["gather_cycle_no_click_reward"]
        for config in m1_configs
    } == {0.02}
    assert {
        config.environment.parameters["carrying_no_click_reward"]
        for config in m1_configs
    } == {0.05}
    assert {
        config.environment.parameters["click_gather_cycle_penalty"]
        for config in m1_configs
    } == {1.0}
