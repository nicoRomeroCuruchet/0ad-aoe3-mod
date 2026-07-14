from pathlib import Path

import numpy as np
import pytest

from rl.agents.base import AgentSpec, Policy, TrainRequest
from rl.agents.sb3 import (
    SB3DependencyError,
    SB3Policy,
    SB3SACTrainer,
    load_sb3_sac_policy,
)


class FakeModel:
    def __init__(self):
        self.predict_calls = []
        self.saved_to = None

    def predict(self, observation, *, deterministic):
        self.predict_calls.append((observation, deterministic))
        return np.array([0.25, -0.5], dtype=np.float64), None

    def save(self, path):
        self.saved_to = path


class FakeSAC:
    constructed = []
    loaded_from = []

    def __init__(self, policy, env, **kwargs):
        self.policy_name = policy
        self.env = env
        self.kwargs = kwargs
        self.learn_calls = []
        self.predict_calls = []
        FakeSAC.constructed.append(self)

    def learn(self, *, total_timesteps):
        self.learn_calls.append(total_timesteps)
        return self

    def predict(self, observation, *, deterministic):
        self.predict_calls.append((observation, deterministic))
        return np.zeros(2, dtype=np.float32), None

    def save(self, path):
        self.saved_to = path

    @classmethod
    def load(cls, path):
        cls.loaded_from.append(path)
        return cls("loaded", None)


def test_sb3_policy_adapts_predict_to_the_common_policy_contract(tmp_path: Path):
    model = FakeModel()
    policy = SB3Policy(model)
    observation = np.array([1.0, 2.0], dtype=np.float32)

    action = policy.act(observation, deterministic=True)
    policy.save(tmp_path / "model")

    assert isinstance(policy, Policy)
    np.testing.assert_array_equal(action, np.array([0.25, -0.5], dtype=np.float32))
    assert model.predict_calls == [(observation, True)]
    assert model.saved_to == str(tmp_path / "model")


def test_sb3_sac_trainer_owns_the_library_specific_training_loop(monkeypatch):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    env = object()
    request = TrainRequest(
        env=env,
        agent=AgentSpec(
            name="sb3_sac",
            parameters={
                "policy": "MultiInputPolicy",
                "learning_starts": 200,
                "buffer_size": 50_000,
            },
        ),
        total_steps=1_234,
        seed=7,
    )

    policy = SB3SACTrainer().fit(request)

    model = FakeSAC.constructed[-1]
    assert isinstance(policy, SB3Policy)
    assert model.policy_name == "MultiInputPolicy"
    assert model.env is env
    assert model.kwargs == {
        "buffer_size": 50_000,
        "learning_starts": 200,
        "seed": 7,
    }
    assert model.learn_calls == [1_234]


def test_load_sb3_sac_policy_uses_the_same_adapter(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    model_path = tmp_path / "saved-model.zip"

    policy = load_sb3_sac_policy(model_path)

    assert isinstance(policy, SB3Policy)
    assert FakeSAC.loaded_from[-1] == str(model_path)


def test_missing_stable_baselines_dependency_has_an_actionable_error(monkeypatch):
    def fail_import(_name):
        raise ModuleNotFoundError("stable_baselines3")

    monkeypatch.setattr("rl.agents.sb3.import_module", fail_import)

    with pytest.raises(SB3DependencyError, match="rl/requirements.txt"):
        load_sb3_sac_policy("model")
