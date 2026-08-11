import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from rl.agents.base import AgentSpec, Policy, TrainRequest
from rl.agents.sb3 import (
    SB3DependencyError,
    SB3Policy,
    SB3PPOTrainer,
    SB3SACTrainer,
    _learn_until_solved,
    _save_checkpoint,
    _validate_checkpoint_pair,
    load_sb3_ppo_policy,
    load_sb3_sac_policy,
)


class FakeModel:
    def __init__(self):
        self.predict_calls = []
        self.saved_to = None
        self.replay_buffer_saved_to = None
        self.num_timesteps = 7

    def predict(self, observation, *, deterministic):
        self.predict_calls.append((observation, deterministic))
        return np.array([0.25, -0.5], dtype=np.float64), None

    def save(self, path):
        self.saved_to = path
        destination = Path(path)
        if destination.suffix == "":
            destination = Path(f"{destination}.zip")
        destination.write_bytes(f"model:{self.num_timesteps}".encode())

    def save_replay_buffer(self, path):
        self.replay_buffer_saved_to = path
        Path(path).write_bytes(f"replay:{self.num_timesteps}".encode())


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
        destination = Path(path)
        if destination.suffix == "":
            destination = Path(f"{destination}.zip")
        destination.write_bytes(f"model:{self.num_timesteps}".encode())

    def save_replay_buffer(self, path):
        self.replay_buffer_saved_paths.append(path)
        Path(path).write_bytes(f"replay:{self.num_timesteps}".encode())

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
    assert (tmp_path / "model.zip").read_bytes() == b"model:7"
    assert (tmp_path / "model.replay_buffer.pkl").read_bytes() == b"replay:7"
    manifest = json.loads((tmp_path / "model.checkpoint.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["num_timesteps"] == 7
    assert manifest["model_sha256"] == sha256(b"model:7").hexdigest()
    assert manifest["replay_buffer_sha256"] == sha256(b"replay:7").hexdigest()


@pytest.mark.parametrize(
    ("name", "archive_name", "companion_base"),
    [
        ("model", "model.zip", "model"),
        ("model.zip", "model.zip", "model"),
        ("model.ckpt", "model.ckpt", "model.ckpt"),
        ("copy.v1.zip", "copy.v1.zip", "copy.v1"),
        ("copy.v1", "copy.v1", "copy.v1"),
    ],
)
def test_checkpoint_save_preserves_sb3_path_suffix_rules(
    tmp_path: Path,
    name,
    archive_name,
    companion_base,
):
    _save_checkpoint(FakeModel(), tmp_path / name)

    assert (tmp_path / archive_name).read_bytes() == b"model:7"
    assert (tmp_path / f"{companion_base}.replay_buffer.pkl").is_file()
    assert (tmp_path / f"{companion_base}.checkpoint.json").is_file()


def test_failed_replay_serialization_preserves_previous_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "model"
    _save_checkpoint(FakeModel(), checkpoint)
    paths = (
        tmp_path / "model.zip",
        tmp_path / "model.replay_buffer.pkl",
        tmp_path / "model.checkpoint.json",
    )
    previous = {path: path.read_bytes() for path in paths}

    class FailingReplayModel(FakeModel):
        def save_replay_buffer(self, path):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _save_checkpoint(FailingReplayModel(), checkpoint)

    assert {path: path.read_bytes() for path in paths} == previous
    assert not (tmp_path / "model.checkpoint.pending.json").exists()
    assert not tuple(tmp_path.glob(".model.*"))


def test_committed_manifest_wins_over_a_stale_pending_marker(tmp_path: Path):
    checkpoint = tmp_path / "model"
    _save_checkpoint(FakeModel(), checkpoint)
    (tmp_path / "model.checkpoint.pending.json").write_text("{}")

    assert _validate_checkpoint_pair(checkpoint) == 7


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
                "policy_kwargs": {"net_arch": (64, 64)},
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
        "policy_kwargs": {"net_arch": [64, 64]},
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
            parameters={
                "policy": "MlpPolicy",
                "learning_starts": 200,
                "policy_kwargs": {"net_arch": (64, 64)},
            },
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


def test_sb3_sac_trainer_validates_a_managed_pair_before_resuming(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    checkpoint = tmp_path / "model"
    source = FakeSAC("MlpPolicy", object())
    _save_checkpoint(source, checkpoint)
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=10,
        seed=0,
        resume_from=checkpoint,
    )

    policy = SB3SACTrainer().fit(request)

    assert policy.model.replay_buffer_loaded_from == [
        str(tmp_path / "model.replay_buffer.pkl")
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


def test_sb3_sac_trainer_saves_the_best_ten_episode_mean(
    monkeypatch,
    tmp_path: Path,
):
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
    for reward in range(9):
        callback.locals = {"infos": [{"episode": {"r": float(reward)}}]}
        assert callback._on_step() is True
    assert model.saved_paths == []

    callback.locals = {"infos": [{"episode": {"r": 9.0}}]}
    assert callback._on_step() is True
    for reward in [0.0] * 10:
        callback.locals = {"infos": [{"episode": {"r": reward}}]}
        assert callback._on_step() is True
    for reward in [10.0] * 10:
        callback.locals = {"infos": [{"episode": {"r": reward}}]}
        assert callback._on_step() is True

    assert len(model.saved_paths) == 2
    assert len(model.replay_buffer_saved_paths) == 2
    assert callback.best_mean_reward == 10.0
    assert (tmp_path / "best_model.zip").is_file()
    assert (tmp_path / "best_model.replay_buffer.pkl").is_file()
    assert (tmp_path / "best_model.checkpoint.json").is_file()


def test_sb3_sac_trainer_stops_only_after_deterministic_solve_check(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)

    class ChunkedFakeSAC(FakeSAC):
        def __init__(self, policy, env, **kwargs):
            super().__init__(policy, env, **kwargs)
            self.num_timesteps = 0
            self.set_env_calls = []

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
            if reset_num_timesteps:
                self.num_timesteps = 0
            self.num_timesteps += total_timesteps
            return self

        def get_env(self):
            return self.env

        def set_env(self, env, *, force_reset):
            self.env = env
            self.set_env_calls.append((env, force_reset))

    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: ChunkedFakeSAC)
    checked_at = []
    rates = iter((0.0, 0.8))

    def evaluate_solution(policy):
        checked_at.append(policy.model.num_timesteps)
        return next(rates)

    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=100,
        seed=7,
        checkpoint_path=tmp_path / "model",
        solved_window_episodes=20,
        solved_success_rate=0.8,
        solved_min_steps=40,
        solved_check_interval_steps=20,
        solve_evaluator=evaluate_solution,
    )

    policy = SB3SACTrainer().fit(request)

    model = policy.model
    assert [call[0] for call in model.learn_calls] == [20, 20, 20]
    assert [call[3] for call in model.learn_calls] == [True, False, False]
    assert checked_at == [40, 60]
    assert model.set_env_calls == [(request.env, True), (request.env, True)]
    assert policy.training_outcome.stop_reason == "solved"
    assert policy.training_outcome.steps_this_run == 60
    assert policy.training_outcome.last_success_rate == 0.8
    assert len(model.saved_paths) == 3
    assert len(model.replay_buffer_saved_paths) == 3
    assert (tmp_path / "model.zip").is_file()
    assert (tmp_path / "model.replay_buffer.pkl").is_file()
    assert (tmp_path / "model.checkpoint.json").is_file()


def test_sb3_sac_trainer_rejects_a_mismatched_checkpoint_pair(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    checkpoint = tmp_path / "model"
    _save_checkpoint(FakeModel(), checkpoint)
    (tmp_path / "model.replay_buffer.pkl").write_bytes(b"different generation")
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=10,
        seed=0,
        resume_from=checkpoint,
    )

    load_count = len(FakeSAC.loaded_from)
    with pytest.raises(RuntimeError, match="checksum"):
        SB3SACTrainer().fit(request)
    assert len(FakeSAC.loaded_from) == load_count


def test_interrupted_public_replacement_is_rejected_before_deserialization(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    checkpoint = tmp_path / "model"
    _save_checkpoint(FakeModel(), checkpoint)
    original_replace = Path.replace
    final_replay = tmp_path / "model.replay_buffer.pkl"

    def fail_replay_replace(source, target):
        if Path(target) == final_replay:
            raise OSError("replace interrupted")
        return original_replace(source, target)

    replacement = FakeModel()
    replacement.num_timesteps = 8
    monkeypatch.setattr(Path, "replace", fail_replay_replace)
    with pytest.raises(OSError, match="replace interrupted"):
        _save_checkpoint(replacement, checkpoint)

    assert (tmp_path / "model.checkpoint.pending.json").is_file()
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=10,
        seed=0,
        resume_from=checkpoint,
    )
    load_count = len(FakeSAC.loaded_from)
    with pytest.raises(RuntimeError, match="checksum"):
        SB3SACTrainer().fit(request)
    assert len(FakeSAC.loaded_from) == load_count


def test_failed_retry_preserves_an_inherited_pending_marker(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    checkpoint = tmp_path / "model"
    archive = tmp_path / "model.zip"
    replay = tmp_path / "model.replay_buffer.pkl"
    pending = tmp_path / "model.checkpoint.pending.json"
    archive.write_bytes(b"legacy model")
    replay.write_bytes(b"legacy replay")
    original_replace = Path.replace

    def interrupt_first_upgrade(source, target):
        if Path(target) == replay:
            raise OSError("replace interrupted")
        return original_replace(source, target)

    first_replacement = FakeModel()
    first_replacement.num_timesteps = 8
    monkeypatch.setattr(Path, "replace", interrupt_first_upgrade)
    with pytest.raises(OSError, match="replace interrupted"):
        _save_checkpoint(first_replacement, checkpoint)
    monkeypatch.setattr(Path, "replace", original_replace)

    assert archive.read_bytes() == b"model:8"
    assert replay.read_bytes() == b"legacy replay"
    assert pending.is_file()

    class FailingRetryModel(FakeModel):
        def save_replay_buffer(self, path):
            raise OSError("retry disk full")

    with pytest.raises(OSError, match="retry disk full"):
        _save_checkpoint(FailingRetryModel(), checkpoint)

    assert pending.is_file()
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=10,
        seed=0,
        resume_from=checkpoint,
    )
    load_count = len(FakeSAC.loaded_from)
    with pytest.raises(RuntimeError, match="incomplete"):
        SB3SACTrainer().fit(request)
    assert len(FakeSAC.loaded_from) == load_count


def test_pending_checkpoint_without_a_manifest_is_not_treated_as_legacy(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    checkpoint = tmp_path / "model"
    (tmp_path / "model.zip").write_bytes(b"partial")
    (tmp_path / "model.checkpoint.pending.json").write_text("{}")
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=10,
        seed=0,
        resume_from=checkpoint,
    )

    load_count = len(FakeSAC.loaded_from)
    with pytest.raises(RuntimeError, match="incomplete"):
        SB3SACTrainer().fit(request)
    assert len(FakeSAC.loaded_from) == load_count


def test_non_utf8_checkpoint_manifest_has_a_clean_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    checkpoint = tmp_path / "model"
    (tmp_path / "model.checkpoint.json").write_bytes(b"\xff")
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=10,
        seed=0,
        resume_from=checkpoint,
    )

    with pytest.raises(RuntimeError, match="manifest is unreadable"):
        SB3SACTrainer().fit(request)


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_checkpoint_manifest_requires_an_exact_integer_version(
    tmp_path: Path,
    version,
):
    checkpoint = tmp_path / "model"
    _save_checkpoint(FakeModel(), checkpoint)
    manifest_path = tmp_path / "model.checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = version
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="unsupported"):
        _validate_checkpoint_pair(checkpoint)


def test_sb3_sac_trainer_records_safety_cap_when_solve_check_keeps_failing(
    monkeypatch,
):
    class ChunkedFakeSAC(FakeSAC):
        def __init__(self, policy, env, **kwargs):
            super().__init__(policy, env, **kwargs)
            self.num_timesteps = 0

        def learn(self, *, total_timesteps, log_interval, callback=None, reset_num_timesteps=True):
            self.learn_calls.append(
                (total_timesteps, log_interval, callback, reset_num_timesteps)
            )
            if reset_num_timesteps:
                self.num_timesteps = 0
            self.num_timesteps += total_timesteps
            return self

        def get_env(self):
            return self.env

        def set_env(self, env, *, force_reset):
            del force_reset
            self.env = env

    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: ChunkedFakeSAC)
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=55,
        seed=7,
        solved_window_episodes=5,
        solved_success_rate=1.0,
        solved_min_steps=0,
        solved_check_interval_steps=20,
        solve_evaluator=lambda policy: 0.0,
    )

    policy = SB3SACTrainer().fit(request)

    assert [call[0] for call in policy.model.learn_calls] == [20, 20, 15]
    assert policy.training_outcome.stop_reason == "safety_cap"
    assert policy.training_outcome.steps_this_run == 55
    assert policy.training_outcome.solve_checks == 3


def test_resumed_solved_model_is_checked_before_more_training():
    class ResumedFakeModel:
        def __init__(self):
            self.num_timesteps = 250
            self.env = object()
            self.learn_calls = []
            self.set_env_calls = []

        def learn(self, **kwargs):
            self.learn_calls.append(kwargs)

        def get_env(self):
            return self.env

        def set_env(self, env, *, force_reset):
            self.set_env_calls.append((env, force_reset))

    model = ResumedFakeModel()
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_sac"),
        total_steps=100,
        seed=7,
        solved_window_episodes=20,
        solved_success_rate=0.8,
        solved_min_steps=200,
        solved_check_interval_steps=20,
        solve_evaluator=lambda policy: 1.0,
    )

    outcome = _learn_until_solved(
        model,
        request,
        callback=None,
        reset_num_timesteps=False,
    )

    assert model.learn_calls == []
    assert model.set_env_calls == [(model.env, True)]
    assert outcome.stop_reason == "solved"
    assert outcome.steps_this_run == 0
    assert outcome.solve_checks == 1


def test_load_sb3_sac_policy_uses_the_same_adapter(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_sac_class", lambda: FakeSAC)
    model_path = tmp_path / "saved-model.zip"

    policy = load_sb3_sac_policy(model_path)

    assert isinstance(policy, SB3Policy)
    assert FakeSAC.loaded_from[-1] == str(model_path)


class FakePPO(FakeSAC):
    """An on-policy stand-in: no replay buffer to save, load, or warm up."""

    constructed = []
    loaded_from = []
    loaded_kwargs = []

    save_replay_buffer = None
    load_replay_buffer = None

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)
        FakePPO.constructed.append(self)


def test_on_policy_checkpoints_record_no_replay_buffer(tmp_path: Path):
    checkpoint = tmp_path / "model"

    _save_checkpoint(FakePPO("MlpPolicy", object()), checkpoint)

    manifest = json.loads((tmp_path / "model.checkpoint.json").read_text())
    assert manifest["replay_buffer_sha256"] is None
    assert manifest["num_timesteps"] == 250
    assert not (tmp_path / "model.replay_buffer.pkl").exists()
    assert _validate_checkpoint_pair(checkpoint) == 250


def test_sb3_ppo_trainer_trains_until_the_task_is_solved(monkeypatch):
    class ChunkedFakePPO(FakePPO):
        def __init__(self, policy, env, **kwargs):
            super().__init__(policy, env, **kwargs)
            self.num_timesteps = 0

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
            if reset_num_timesteps:
                self.num_timesteps = 0
            self.num_timesteps += total_timesteps
            return self

        def get_env(self):
            return self.env

        def set_env(self, env, *, force_reset):
            del force_reset
            self.env = env

    monkeypatch.setattr("rl.agents.sb3._load_ppo_class", lambda: ChunkedFakePPO)
    success_rates = iter((0.5, 1.0))
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_ppo", parameters={"n_steps": 256}),
        total_steps=5_000,
        seed=7,
        solved_window_episodes=20,
        solved_success_rate=0.8,
        solved_min_steps=0,
        solved_check_interval_steps=1_250,
        solve_evaluator=lambda policy: next(success_rates),
    )

    policy = SB3PPOTrainer().fit(request)

    assert policy.model.kwargs == {"n_steps": 256, "seed": 7}
    assert [call[0] for call in policy.model.learn_calls] == [1_250, 1_250]
    assert policy.training_outcome.stop_reason == "solved"
    assert policy.training_outcome.steps_this_run == 2_500
    assert policy.training_outcome.solve_checks == 2


def test_sb3_ppo_resume_skips_replay_buffer_recovery(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_ppo_class", lambda: FakePPO)
    checkpoint = tmp_path / "model"
    _save_checkpoint(FakePPO("MlpPolicy", object()), checkpoint)
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_ppo"),
        total_steps=10,
        seed=3,
        resume_from=checkpoint,
    )

    policy = SB3PPOTrainer().fit(request)

    assert FakePPO.loaded_from[-1] == str(checkpoint)
    assert policy.model.random_seed == 3
    assert policy.model.learn_calls[-1][3] is False


def test_load_sb3_ppo_policy_uses_the_same_adapter(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_ppo_class", lambda: FakePPO)
    model_path = tmp_path / "saved-model.zip"

    policy = load_sb3_ppo_policy(model_path)

    assert isinstance(policy, SB3Policy)
    assert FakePPO.loaded_from[-1] == str(model_path)


def test_missing_stable_baselines_dependency_has_an_actionable_error(monkeypatch):
    def fail_import(_name):
        raise ModuleNotFoundError("stable_baselines3")

    monkeypatch.setattr("rl.agents.sb3.import_module", fail_import)

    with pytest.raises(SB3DependencyError, match="uv sync --locked"):
        load_sb3_sac_policy("model")


def test_resuming_does_not_reapply_the_m1_warm_start(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("rl.agents.sb3._load_ppo_class", lambda: FakePPO)
    warm_starts = []
    monkeypatch.setattr(
        "rl.agents.shared_policy.initialize_from_m1",
        lambda policy, path: warm_starts.append(path),
    )
    checkpoint = tmp_path / "model"
    _save_checkpoint(FakePPO("MlpPolicy", object()), checkpoint)
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="sb3_ppo", parameters={"m1_checkpoint": "m1/model"}),
        total_steps=10,
        seed=0,
        resume_from=checkpoint,
    )

    SB3PPOTrainer().fit(request)

    assert warm_starts == []
    assert "m1_checkpoint" not in FakePPO.loaded_kwargs[-1]
