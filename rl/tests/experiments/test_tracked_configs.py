from pathlib import Path

from rl.experiments.config import load_experiment_config


CONFIG_DIRECTORY = Path(__file__).resolve().parents[2] / "configs"


def test_all_tracked_experiment_configs_are_valid_and_comparable():
    paths = sorted(CONFIG_DIRECTORY.glob("*.toml"))

    assert {
        "m0_oracle.toml",
        "m0_random.toml",
        "m0_sb3_ppo.toml",
        "m0_sb3_sac.toml",
        "m1_oracle.toml",
        "m1_random.toml",
        "m1_sb3_ppo.toml",
        "m1_sb3_sac.toml",
    }.issubset(path.name for path in paths)

    configs_by_name = {path.name: load_experiment_config(path) for path in paths}
    assert {
        configs_by_name[name].evaluation.episodes
        for name in ("m0_oracle.toml", "m1_oracle.toml")
    } == {1}
    assert {
        configs_by_name[name].evaluation.episodes
        for name in (
            "m0_sb3_sac.toml",
            "m1_sb3_sac.toml",
            "m0_sb3_ppo.toml",
            "m1_sb3_ppo.toml",
        )
    } == {20}
    assert {
        configs_by_name[name].evaluation.episodes
        for name in ("m0_random.toml", "m1_random.toml")
    } == {10}
    for name in ("m0_sb3_sac.toml", "m1_sb3_sac.toml"):
        parameters = configs_by_name[name].agent.parameters
        assert parameters["batch_size"] == 128
        assert parameters["train_freq"] == 8
        assert parameters["gradient_steps"] == 8
    for name in ("m0_sb3_ppo.toml", "m1_sb3_ppo.toml"):
        parameters = configs_by_name[name].agent.parameters
        assert parameters["batch_size"] == 64
        assert parameters["n_epochs"] == 10

    # SAC and PPO must stop on the same task-completion criterion and share the
    # same safety cap so their learning curves stay directly comparable.
    for milestone, total_steps, min_steps, interval in (
        ("m0", 50_000, 1_000, 1_250),
        ("m1", 500_000, 2_500, 2_500),
    ):
        for algorithm in ("sac", "ppo"):
            training = configs_by_name[f"{milestone}_sb3_{algorithm}.toml"].training
            assert training.solved_window_episodes == 20
            assert training.solved_success_rate == 0.8
            assert training.total_steps == total_steps
            assert training.solved_min_steps == min_steps
            assert training.solved_check_interval_steps == interval

    starter_configs = [
        configs_by_name[name]
        for name in (
            "m0_oracle.toml",
            "m0_random.toml",
            "m0_sb3_sac.toml",
            "m0_sb3_ppo.toml",
        )
    ]
    assert {
        "oracle",
        "random",
        "sb3_sac",
        "sb3_ppo",
    } == {config.agent.name for config in starter_configs}
    assert {config.environment.name for config in starter_configs} == {"zero_ad_gather"}
    assert {
        config.environment.parameters["scenario"] for config in starter_configs
    } == {"rl/reset_config.json"}
    assert {config.evaluation.seed for config in starter_configs} == {1000}

    m1_configs = [
        configs_by_name[name]
        for name in (
            "m1_oracle.toml",
            "m1_random.toml",
            "m1_sb3_sac.toml",
            "m1_sb3_ppo.toml",
        )
    ]
    assert {
        "oracle",
        "random",
        "sb3_sac",
        "sb3_ppo",
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
        config.environment.parameters["lifecycle_state_observation"]
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
    } == {0.0}
    assert {
        config.environment.parameters["carried_resource_delta_reward_scale"]
        for config in m1_configs
    } == {0.2}
    assert {
        config.environment.parameters["gather_cycle_no_click_reward"]
        for config in m1_configs
    } == {0.0}
    assert {
        config.environment.parameters["carrying_no_click_reward"]
        for config in m1_configs
    } == {0.0}
    assert {
        config.environment.parameters["click_gather_cycle_penalty"]
        for config in m1_configs
    } == {1.0}


def test_m2_configs_share_the_team_scenario_and_threshold():
    paths = sorted(CONFIG_DIRECTORY.glob("m2_*.toml"))
    configs = {path.name: load_experiment_config(path) for path in paths}

    assert {
        "m2_oracle.toml",
        "m2_random.toml",
        "m2_sb3_ppo.toml",
    }.issubset(configs)
    for config in configs.values():
        assert config.environment.name == "zero_ad_team_gather"
        assert config.environment.parameters["villager_count"] == 4
        assert config.environment.parameters["resource_count"] == 4
        assert config.environment.parameters["stock_success_threshold"] == 80.0
        assert config.environment.parameters["click_gather_cycle_penalty"] == 1.0
        assert (
            config.environment.parameters["scenario"]
            == "rl/scenarios/team_reset_config.json"
        )
    training = configs["m2_sb3_ppo.toml"].training
    assert training.solved_success_rate == 0.8
    assert training.solved_check_interval_steps == 3_000


def test_m2_pays_for_finishing_fast_with_a_shared_villager_policy():
    configs = {
        path.name: load_experiment_config(path)
        for path in sorted(CONFIG_DIRECTORY.glob("m2_*.toml"))
    }

    for config in configs.values():
        assert config.environment.parameters["horizon"] == 40

    agent = configs["m2_sb3_ppo.toml"].agent.parameters
    # One network per villager, so an idle villager cannot diverge from a
    # working one, and a discount that makes finishing sooner worth more.
    assert agent["policy"] == "SharedVillagerPolicy"
    assert agent["gamma"] == 0.9
