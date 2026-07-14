import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import rl.train as train_cli
from rl.agents.base import EpisodeResult
from rl.experiments.evaluation import DecisionRecord, EvaluationReport


EXPERIMENT = """
[environment]
name = "zero_ad_gather"
scenario = "rl/reset_config.json"
uri = "http://localhost:6000"

[agent]
name = "sb3_sac"
policy = "MlpPolicy"

[training]
total_steps = 5000
seed = 3

[evaluation]
episodes = 1
deterministic = true
seed = 4
"""


class ConstantPolicy:
    def act(self, observation, *, deterministic):
        del observation, deterministic
        return np.zeros(2, dtype=np.float32)


class CloseableEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return np.full(5, 0.25, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.full(5, 0.5, dtype=np.float32), 0.0, True, False, {}

    def close(self):
        self.closed = True


class FakeAgentView:
    def __init__(self, events, *, close_error=None):
        self.events = events
        self.close_error = close_error
        self.closed = False

    def update(self, record):
        self.events.append(("view", record))

    def pause(self, delay):
        self.events.append(("pause", delay))

    def close(self):
        self.events.append("close_view")
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def completed_report():
    return EvaluationReport(
        episodes=(
            EpisodeResult(
                episode=0,
                total_reward=2.0,
                steps=1,
                terminated=True,
                truncated=False,
                final_info={"distance": 0.0},
            ),
        )
    )


def write_experiment(tmp_path: Path) -> Path:
    path = tmp_path / "m0_sb3_sac.toml"
    path.write_text(EXPERIMENT, encoding="utf-8")
    return path


def test_apply_overrides_returns_a_new_validated_config(tmp_path: Path):
    original = train_cli.load_experiment_config(write_experiment(tmp_path))

    updated = train_cli.apply_overrides(
        original,
        uri="http://example.test:7000",
        total_steps=12,
    )

    assert updated is not original
    assert updated.environment.parameters["uri"] == "http://example.test:7000"
    assert updated.training.total_steps == 12
    assert original.environment.parameters["uri"] == "http://localhost:6000"
    assert original.training.total_steps == 5000


@pytest.mark.parametrize("delay", ["-0.1", "nan", "inf", "-inf"])
def test_parser_rejects_invalid_agent_view_delays(delay):
    with pytest.raises(SystemExit):
        train_cli.build_parser().parse_args(["--delay", delay])


def test_main_runs_configured_training_records_artifacts_and_closes_env(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    experiment_path = write_experiment(tmp_path)
    env = CloseableEnv()
    calls = {}
    events = []

    def fake_build_environment(config, **kwargs):
        calls["environment"] = (config, kwargs)
        return env

    def fake_train_policy(config, selected_env, *, decision_observer):
        calls["config"] = config
        calls["env"] = selected_env
        calls["training_observer"] = decision_observer
        events.append("train")
        return ConstantPolicy()

    def fake_save(agent, policy, path):
        calls["saved"] = (agent, policy, Path(path))
        events.append("save")

    def fake_evaluate(selected_env, policy, **kwargs):
        calls["evaluation"] = (selected_env, policy, kwargs)
        events.append("evaluate")
        return completed_report()

    monkeypatch.setattr(train_cli, "build_environment", fake_build_environment)
    monkeypatch.setattr(train_cli, "train_policy", fake_train_policy)
    monkeypatch.setattr(train_cli, "save_policy", fake_save)
    monkeypatch.setattr(train_cli, "evaluate", fake_evaluate)

    exit_code = train_cli.main(
        [
            "--experiment",
            str(experiment_path),
            "--run-root",
            str(tmp_path / "runs"),
            "--timesteps",
            "12",
            "--uri",
            "http://example.test:7000",
            "--allow-remote-server",
        ]
    )

    run_directories = list((tmp_path / "runs").iterdir())
    assert exit_code == 0
    assert len(run_directories) == 1
    assert calls["env"] is env
    assert calls["config"].training.total_steps == 12
    assert calls["saved"][2] == run_directories[0] / "model"
    assert calls["environment"][1] == {"allow_remote": True}
    assert calls["training_observer"] is None
    assert calls["evaluation"][2]["decision_observer"] is None
    assert events == ["train", "save", "evaluate"]
    assert env.closed is True

    resolved = json.loads(
        (run_directories[0] / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert resolved["environment"]["uri"] == "http://example.test:7000"
    assert resolved["training"]["total_steps"] == 12
    metadata = json.loads(
        (run_directories[0] / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "complete"
    assert "modelo:" in capsys.readouterr().out


def test_main_saves_the_checkpoint_before_a_failed_post_training_evaluation(
    tmp_path: Path,
    monkeypatch,
):
    env = CloseableEnv()
    events = []
    agent_view = FakeAgentView(events)
    monkeypatch.setattr(
        train_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )
    monkeypatch.setattr(
        train_cli,
        "train_policy",
        lambda config, selected_env, *, decision_observer: (
            events.append("train") or ConstantPolicy()
        ),
    )
    monkeypatch.setattr(
        train_cli,
        "save_policy",
        lambda agent, policy, path: events.append("save"),
    )
    monkeypatch.setattr(
        train_cli,
        "open_agent_view",
        lambda selected_env: events.append("open_view") or agent_view,
    )

    def fail_evaluation(env, policy, **kwargs):
        events.append("evaluate")
        raise RuntimeError("server disconnected")

    monkeypatch.setattr(train_cli, "evaluate", fail_evaluation)

    with pytest.raises(RuntimeError, match="server disconnected"):
        train_cli.main(
            [
                "--experiment",
                str(write_experiment(tmp_path)),
                "--run-root",
                str(tmp_path / "runs"),
                "--agent-view",
            ]
        )

    assert events == ["open_view", "train", "save", "evaluate", "close_view"]
    assert agent_view.closed is True
    assert env.closed is True
    run_directory = next((tmp_path / "runs").iterdir())
    assert (run_directory / "resolved_config.json").is_file()
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "checkpoint_saved"
    assert not (run_directory / "metrics.json").exists()


def test_main_shows_and_paces_training_and_post_training_evaluation(
    tmp_path: Path,
    monkeypatch,
):
    env = CloseableEnv()
    events = []
    agent_view = FakeAgentView(events)
    opened_for = []
    monkeypatch.setattr(
        train_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )

    def fake_train_policy(config, selected_env, *, decision_observer):
        del config, selected_env
        events.append("train")
        decision_observer(
            DecisionRecord(
                episode=0,
                step=0,
                observation=np.full(5, 0.25, dtype=np.float32),
                action=np.ones(2, dtype=np.float32),
            )
        )
        return ConstantPolicy()

    monkeypatch.setattr(train_cli, "train_policy", fake_train_policy)
    monkeypatch.setattr(
        train_cli,
        "save_policy",
        lambda agent, policy, path: events.append("save"),
    )
    monkeypatch.setattr(
        train_cli,
        "open_agent_view",
        lambda selected_env: (
            opened_for.append(selected_env)
            or events.append("open_view")
            or agent_view
        ),
    )

    def fake_evaluate(selected_env, policy, **kwargs):
        events.append("evaluate")
        kwargs["decision_observer"](
            DecisionRecord(
                episode=0,
                step=0,
                observation=np.zeros(5, dtype=np.float32),
                action=np.zeros(2, dtype=np.float32),
            )
        )
        return completed_report()

    monkeypatch.setattr(train_cli, "evaluate", fake_evaluate)

    exit_code = train_cli.main(
        [
            "--experiment",
            str(write_experiment(tmp_path)),
            "--run-root",
            str(tmp_path / "runs"),
            "--agent-view",
            "--delay",
            "0.5",
        ]
    )

    assert exit_code == 0
    assert opened_for == [env]
    assert events[0:2] == ["open_view", "train"]
    assert events[2][0] == "view"
    np.testing.assert_array_equal(
        events[2][1].observation,
        np.full(5, 0.25, dtype=np.float32),
    )
    np.testing.assert_array_equal(events[2][1].action, np.ones(2, dtype=np.float32))
    assert events[3:6] == [("pause", 0.5), "save", "evaluate"]
    assert events[6][0] == "view"
    assert events[7:] == [("pause", 0.5), "close_view"]
    assert agent_view.closed is True
    assert env.closed is True


def test_main_closes_the_environment_when_agent_view_close_fails(
    tmp_path: Path,
    monkeypatch,
):
    env = CloseableEnv()
    agent_view = FakeAgentView([], close_error=RuntimeError("close failed"))
    monkeypatch.setattr(train_cli, "build_environment", lambda config, **kwargs: env)
    monkeypatch.setattr(
        train_cli,
        "train_policy",
        lambda config, selected_env, *, decision_observer: ConstantPolicy(),
    )
    monkeypatch.setattr(train_cli, "save_policy", lambda agent, policy, path: None)
    monkeypatch.setattr(train_cli, "open_agent_view", lambda selected_env: agent_view)
    monkeypatch.setattr(
        train_cli,
        "evaluate",
        lambda env, policy, **kwargs: completed_report(),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        train_cli.main(
            [
                "--experiment",
                str(write_experiment(tmp_path)),
                "--run-root",
                str(tmp_path / "runs"),
                "--agent-view",
            ]
        )

    assert agent_view.closed is True
    assert env.closed is True


def test_main_does_not_train_when_agent_view_cannot_open(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    env = CloseableEnv()
    events = []
    monkeypatch.setattr(train_cli, "build_environment", lambda config, **kwargs: env)
    monkeypatch.setattr(
        train_cli,
        "train_policy",
        lambda config, selected_env, *, decision_observer: (
            events.append("train") or ConstantPolicy()
        ),
    )
    monkeypatch.setattr(
        train_cli,
        "save_policy",
        lambda agent, policy, path: events.append("save"),
    )

    def fail_to_open(selected_env):
        events.append("open_view")
        raise train_cli.AgentViewUnavailable("no graphical display")

    monkeypatch.setattr(train_cli, "open_agent_view", fail_to_open)
    monkeypatch.setattr(
        train_cli,
        "evaluate",
        lambda env, policy, **kwargs: (_ for _ in ()).throw(AssertionError),
    )

    with pytest.raises(SystemExit) as error:
        train_cli.main(
            [
                "--experiment",
                str(write_experiment(tmp_path)),
                "--run-root",
                str(tmp_path / "runs"),
                "--agent-view",
            ]
        )

    assert error.value.code == 2
    assert "no graphical display" in capsys.readouterr().err
    assert events == ["open_view"]
    assert env.closed is True
    run_directory = next((tmp_path / "runs").iterdir())
    assert not (run_directory / "metadata.json").exists()
    assert not (run_directory / "metrics.json").exists()


def test_main_refuses_to_overwrite_an_existing_checkpoint_without_force(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "student-model"
    output.with_suffix(".zip").write_bytes(b"existing")
    monkeypatch.setattr(
        train_cli,
        "build_environment",
        lambda config, **kwargs: (_ for _ in ()).throw(AssertionError),
    )

    with pytest.raises(SystemExit) as error:
        train_cli.main(
            [
                "--experiment",
                str(write_experiment(tmp_path)),
                "--out",
                str(output),
            ]
        )

    assert error.value.code == 2
    assert "--force" in capsys.readouterr().err
