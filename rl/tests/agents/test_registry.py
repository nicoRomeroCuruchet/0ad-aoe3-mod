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
    load_policy,
    save_policy,
)
from rl.agents.sb3 import SB3Policy, SB3SACTrainer


class DummyEnv:
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


class SavableModel:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path


def test_registry_lists_explicit_supported_agent_names():
    assert available_agents() == ("oracle", "random", "sb3_sac")


def test_registry_builds_seeded_random_and_oracle_policies():
    first_random = build_policy(AgentSpec("random"), DummyEnv(), seed=11)
    second_random = build_policy(AgentSpec("random"), DummyEnv(), seed=11)
    oracle = build_policy(AgentSpec("oracle"), DummyEnv(), seed=11)
    observation = np.array([0.0, 0.0, 0.25, -0.75, 0.5], dtype=np.float32)

    assert isinstance(first_random, RandomPolicy)
    assert isinstance(oracle, GatherOraclePolicy)
    np.testing.assert_array_equal(
        first_random.act(observation, deterministic=False),
        second_random.act(observation, deterministic=False),
    )
    np.testing.assert_array_equal(
        oracle.act(observation, deterministic=True),
        np.array([0.25, -0.75], dtype=np.float32),
    )


def test_registry_builds_the_sb3_trainer_without_importing_sb3():
    trainer = build_trainer(AgentSpec("sb3_sac"))

    assert isinstance(trainer, SB3SACTrainer)


def test_registry_rejects_unsupported_operations():
    with pytest.raises(AgentCapabilityError, match="cannot be trained"):
        build_trainer(AgentSpec("oracle"))

    with pytest.raises(AgentCapabilityError, match="cannot be built directly"):
        build_policy(AgentSpec("sb3_sac"), DummyEnv(), seed=0)

    with pytest.raises(AgentCapabilityError, match="cannot load"):
        load_policy(AgentSpec("random"), "unused")


def test_registry_reports_unknown_agents_and_available_choices():
    with pytest.raises(UnknownAgentError, match="oracle, random, sb3_sac"):
        build_trainer(AgentSpec("typo"))


def test_registry_saves_sb3_policy_through_its_registered_serializer(tmp_path: Path):
    model = SavableModel()
    policy: Policy = SB3Policy(model)

    save_policy(AgentSpec("sb3_sac"), policy, tmp_path / "model")

    assert model.saved_to == str(tmp_path / "model")


def test_registry_refuses_to_save_a_policy_with_the_wrong_adapter(tmp_path: Path):
    policy = GatherOraclePolicy()

    with pytest.raises(TypeError, match="SB3Policy"):
        save_policy(AgentSpec("sb3_sac"), policy, tmp_path / "model")
