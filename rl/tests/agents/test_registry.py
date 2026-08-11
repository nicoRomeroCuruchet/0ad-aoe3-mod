from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from rl.agents.base import AgentSpec, Policy
from rl.agents.baselines import GatherOraclePolicy, RandomPolicy
from rl.agents.registry import (
    AgentCapabilityError,
    UnknownAgentError,
    available_agents,
    build_policy,
    build_trainer,
    ensure_can_build,
    ensure_can_save,
    load_policy,
    save_policy,
)
from rl.agents.sb3 import SB3Policy, SB3PPOTrainer, SB3SACTrainer


class DummyEnv:
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


class CommandDummyEnv:
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)


class SavableModel:
    def __init__(self):
        self.saved_to = None
        self.replay_buffer_saved_to = None
        self.num_timesteps = 9

    def save(self, path):
        self.saved_to = path
        destination = Path(path)
        if destination.suffix == "":
            destination = Path(f"{destination}.zip")
        destination.write_bytes(b"model")

    def save_replay_buffer(self, path):
        self.replay_buffer_saved_to = path
        Path(path).write_bytes(b"replay")


def test_registry_lists_explicit_supported_agent_names():
    assert {"oracle", "random", "sb3_sac", "sb3_ppo"}.issubset(available_agents())


def test_registry_builds_seeded_random_and_oracle_policies():
    first_random = build_policy(AgentSpec("random"), DummyEnv(), seed=11)
    second_random = build_policy(AgentSpec("random"), DummyEnv(), seed=11)
    oracle = build_policy(AgentSpec("oracle"), DummyEnv(), seed=11)
    observation = np.array([0.0, 0.0, 0.25, -0.75, 0.5], dtype=np.float32)

    assert isinstance(first_random, RandomPolicy)
    assert isinstance(oracle, GatherOraclePolicy)
    ensure_can_build(AgentSpec("oracle"))
    np.testing.assert_array_equal(
        first_random.act(observation, deterministic=False),
        second_random.act(observation, deterministic=False),
    )
    np.testing.assert_array_equal(
        oracle.act(observation, deterministic=True),
        np.array([0.25, -0.75], dtype=np.float32),
    )


def test_registry_builds_oracle_that_matches_command_action_space():
    oracle = build_policy(AgentSpec("oracle"), CommandDummyEnv(), seed=11)
    observation = np.array([0.0, 0.0, 0.25, -0.75, 0.5], dtype=np.float32)

    np.testing.assert_array_equal(
        oracle.act(observation, deterministic=True),
        np.array([0.25, -0.75, 1.0], dtype=np.float32),
    )


def test_registry_builds_the_sb3_trainer_without_importing_sb3():
    trainer = build_trainer(AgentSpec("sb3_sac"))

    assert isinstance(trainer, SB3SACTrainer)
    ensure_can_save(AgentSpec("sb3_sac"))


def test_registry_builds_the_ppo_trainer_without_importing_sb3():
    trainer = build_trainer(AgentSpec("sb3_ppo"))

    assert isinstance(trainer, SB3PPOTrainer)
    ensure_can_save(AgentSpec("sb3_ppo"))


def test_registry_rejects_unsupported_operations():
    with pytest.raises(AgentCapabilityError, match="cannot be trained"):
        build_trainer(AgentSpec("oracle"))

    with pytest.raises(AgentCapabilityError, match="cannot be built directly"):
        ensure_can_build(AgentSpec("sb3_sac"))

    with pytest.raises(AgentCapabilityError, match="cannot load"):
        load_policy(AgentSpec("random"), "unused")

    with pytest.raises(AgentCapabilityError, match="cannot save"):
        ensure_can_save(AgentSpec("oracle"))


def test_registry_reports_unknown_agents_and_available_choices():
    with pytest.raises(UnknownAgentError, match="available agents") as error:
        build_trainer(AgentSpec("typo"))

    for starter_agent in ("oracle", "random", "sb3_sac", "sb3_ppo"):
        assert starter_agent in str(error.value)


def test_registry_saves_sb3_policy_through_its_registered_serializer(tmp_path: Path):
    model = SavableModel()
    policy: Policy = SB3Policy(model)

    save_policy(AgentSpec("sb3_sac"), policy, tmp_path / "model")

    assert (tmp_path / "model.zip").read_bytes() == b"model"
    assert (tmp_path / "model.replay_buffer.pkl").read_bytes() == b"replay"
    assert (tmp_path / "model.checkpoint.json").is_file()


def test_registry_refuses_to_save_a_policy_with_the_wrong_adapter(tmp_path: Path):
    policy = GatherOraclePolicy()

    with pytest.raises(TypeError, match="SB3Policy"):
        save_policy(AgentSpec("sb3_sac"), policy, tmp_path / "model")
