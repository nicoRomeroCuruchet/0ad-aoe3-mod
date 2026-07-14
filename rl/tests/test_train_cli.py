import json
from pathlib import Path

import numpy as np
import pytest

import rl.train as train_cli
from rl.agents.base import EpisodeResult
from rl.experiments.evaluation import EvaluationReport


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


class CloseableEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


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

    def fake_train_policy(config, selected_env):
        calls["config"] = config
        calls["env"] = selected_env
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
    monkeypatch.setattr(
        train_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )
    monkeypatch.setattr(
        train_cli,
        "train_policy",
        lambda config, selected_env: events.append("train") or ConstantPolicy(),
    )
    monkeypatch.setattr(
        train_cli,
        "save_policy",
        lambda agent, policy, path: events.append("save"),
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
            ]
        )

    assert events == ["train", "save", "evaluate"]
    assert env.closed is True
    run_directory = next((tmp_path / "runs").iterdir())
    assert (run_directory / "resolved_config.json").is_file()
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "checkpoint_saved"
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
