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
        self.replay_buffer_saved_to = None

    def predict(self, observation, *, deterministic):
        self.predict_calls.append((observation, deterministic))
        return np.array([0.25, -0.5], dtype=np.float64), None

    def save(self, path):
        self.saved_to = path

    def save_replay_buffer(self, path):
        self.replay_buffer_saved_to = path


class FakeSAC:
    constructed = []
    loaded_from = []
    loaded_kwargs = []

    def __init__(self, policy, env, **kwargs):
        self.policy_name = policy
        self.env = env
        self.kwargs = kwargs
        self.learn_calls = []
        self.predict_calls = []
        self.logger = None
        self.saved_paths = []
        self.replay_buffer_saved_paths = []
        self.replay_buffer_loaded_from = []
        self.random_seed = None
        self.num_timesteps = 250
        self.learning_starts = kwargs.get("learning_starts", 100)
        FakeSAC.constructed.append(self)

    def set_random_seed(self, seed):
        self.random_seed = seed

    def set_logger(self, logger):
        self.logger = logger

    def learn(
        self,
        *,
        total_timesteps,
        log_interval,
        callback=None,
        reset_num_timesteps=True,
    ):
        self.learn_calls.append(
            (total_timesteps, log_interval, callback, reset_num_timesteps)
        )
        return self

    def predict(self, observation, *, deterministic):
        self.predict_calls.append((observation, deterministic))
        return np.zeros(2, dtype=np.float32), None

    def save(self, path):
        self.saved_to = path
        self.saved_paths.append(path)

    def save_replay_buffer(self, path):
        self.replay_buffer_saved_paths.append(path)

    def load_replay_buffer(self, path):
        self.replay_buffer_loaded_from.append(path)

    @classmethod
    def load(cls, path, **kwargs):
        cls.loaded_from.append(path)
        cls.loaded_kwargs.append(kwargs)
        parameters = dict(kwargs)
        env = parameters.pop("env", None)
        return cls("loaded", env, **parameters)


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
    assert model.replay_buffer_saved_to == str(
        tmp_path / "model.replay_buffer.pkl"
    )


def test_sb3_sac_trainer_owns_the_library_specific_training_loop(monkeypatch):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    configured_loggers = []

    def fake_configure(path, formats):
        configured_loggers.append((path, formats))
        return {"path": path, "formats": formats}

    monkeypatch.setattr("rl.agents.sb3._load_logger_configure", lambda: fake_configure)
    env = object()
    log_dir = Path("rl/runs/example/training")
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
        log_dir=log_dir,
        log_interval=3,
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
    assert model.learn_calls == [(1_234, 3, None, True)]
    assert configured_loggers == [(str(log_dir), ["stdout", "csv", "json"])]
    assert model.logger == {
        "path": str(log_dir),
        "formats": ["stdout", "csv", "json"],
    }


def test_sb3_sac_trainer_can_continue_from_a_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    configured_loggers = []

    def fake_configure(path, formats):
        configured_loggers.append((path, formats))
        return {"path": path, "formats": formats}

    monkeypatch.setattr("rl.agents.sb3._load_logger_configure", lambda: fake_configure)
    env = object()
    checkpoint = tmp_path / "best_model"
    replay_buffer = tmp_path / "best_model.replay_buffer.pkl"
    replay_buffer.write_bytes(b"trusted replay")
    request = TrainRequest(
        env=env,
        agent=AgentSpec(
            name="sb3_sac",
            parameters={"policy": "MlpPolicy", "learning_starts": 200},
        ),
        total_steps=500,
        seed=11,
        log_dir=tmp_path / "training",
        log_interval=2,
        resume_from=checkpoint,
    )

    policy = SB3SACTrainer().fit(request)

    model = policy.model
    assert isinstance(policy, SB3Policy)
    assert FakeSAC.loaded_from[-1] == str(checkpoint)
    assert FakeSAC.loaded_kwargs[-1] == {
        "env": env,
        "learning_starts": 200,
        "seed": 11,
    }
    assert model.replay_buffer_loaded_from == [str(replay_buffer)]
    assert model.random_seed == 11
    assert model.learn_calls == [(500, 2, None, False)]
    assert configured_loggers == [
        (str(tmp_path / "training"), ["stdout", "csv", "json"])
    ]


def test_sb3_sac_trainer_rewarms_a_legacy_model_only_checkpoint(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(
            name="sb3_sac",
            parameters={"policy": "MlpPolicy", "learning_starts": 200},
        ),
        total_steps=500,
        seed=11,
        resume_from=tmp_path / "legacy_model",
    )

    policy = SB3SACTrainer().fit(request)

    assert policy.model.replay_buffer_loaded_from == []
    assert policy.model.learning_starts == policy.model.num_timesteps + 200


def test_sb3_sac_trainer_saves_the_best_episode_reward(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)

    class FakeCallback:
        def __init__(self):
            self.locals = {}
            self.model = None

    monkeypatch.setattr(
        "rl.agents.sb3._load_base_callback_class",
        lambda: FakeCallback,
    )
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=123,
        seed=7,
        best_model_path=tmp_path / "best_model",
    )

    SB3SACTrainer().fit(request)

    model = FakeSAC.constructed[-1]
    callback = model.learn_calls[-1][2]
    callback.model = model
    callback.locals = {
        "infos": [
            {"episode": {"r": 0.0}},
            {"episode": {"r": -1.0}},
            {"episode": {"r": 2.5}},
            {"episode": {"r": 2.5}},
        ]
    }

    assert callback._on_step() is True
    assert model.saved_to == str(tmp_path / "best_model")
    assert model.saved_paths == [
        str(tmp_path / "best_model"),
        str(tmp_path / "best_model"),
    ]
    assert model.replay_buffer_saved_paths == [
        str(tmp_path / "best_model.replay_buffer.pkl"),
        str(tmp_path / "best_model.replay_buffer.pkl"),
    ]
    assert callback.best_reward == 2.5


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

    with pytest.raises(SB3DependencyError, match="uv sync --locked"):
        load_sb3_sac_policy("model")
