from pathlib import Path

import numpy as np
import pytest

import rl.eval as eval_cli
from rl.agents.baselines import GatherOraclePolicy
from rl.experiments.evaluation import DecisionRecord, StepRecord


ORACLE_EXPERIMENT = """
[environment]
name = "zero_ad_gather"
scenario = "rl/reset_config.json"
uri = "http://localhost:6000"

[agent]
name = "oracle"

[training]
total_steps = 5000
seed = 3

[evaluation]
episodes = 1
deterministic = true
seed = 4
"""


SB3_EXPERIMENT = ORACLE_EXPERIMENT.replace('name = "oracle"', 'name = "sb3_sac"')


class OneStepEnv:
    map_size_m = 512.0
    observation_labels = (
        "villager_x_norm",
        "villager_z_norm",
        "resource_x_norm",
        "resource_z_norm",
        "distance_norm",
    )

    def __init__(self):
        self.closed = False
        self.reset_seeds = []

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        return np.array([0.0, 0.0, 0.5, -0.5, 0.25], dtype=np.float32), {}

    def step(self, action):
        np.testing.assert_array_equal(action, np.array([0.5, -0.5], dtype=np.float32))
        return np.zeros(5, dtype=np.float32), 3.0, True, False, {"distance": 0.0}

    def close(self):
        self.closed = True


class FakeAgentView:
    def __init__(self, events=None):
        self.events = events
        self.records = []
        self.close_calls = 0

    def update(self, record):
        self.records.append(record)
        if self.events is not None:
            self.events.append(("view", record))

    def pause(self, delay):
        if self.events is not None:
            self.events.append(("pause", delay))

    def close(self):
        self.close_calls += 1


def write_experiment(tmp_path: Path, contents: str = ORACLE_EXPERIMENT) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_apply_overrides_preserves_config_and_adds_replay(tmp_path: Path):
    original = eval_cli.load_experiment_config(write_experiment(tmp_path))

    updated = eval_cli.apply_overrides(
        original,
        uri="http://example.test:7000",
        episodes=3,
        save_replay=True,
    )

    assert updated.environment.parameters["uri"] == "http://example.test:7000"
    assert updated.environment.parameters["save_replay"] is True
    assert updated.evaluation.episodes == 3
    assert "save_replay" not in original.environment.parameters
    assert original.evaluation.episodes == 1


@pytest.mark.parametrize("delay", ["nan", "inf", "-inf"])
def test_parser_rejects_non_finite_delays(delay):
    with pytest.raises(SystemExit):
        eval_cli.build_parser().parse_args(["--delay", delay])


def test_main_evaluates_registered_baseline_in_both_modes_and_closes_env(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    env = OneStepEnv()
    monkeypatch.setattr(
        eval_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )

    exit_code = eval_cli.main(
        [
            "--experiment",
            str(write_experiment(tmp_path)),
            "--mode",
            "both",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "determinista" in output
    assert "estocástica" in output
    assert env.reset_seeds == [4, 4]
    assert env.closed is True


def test_default_mode_uses_the_determinism_declared_in_the_config(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    env = OneStepEnv()
    monkeypatch.setattr(
        eval_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )

    exit_code = eval_cli.main(["--experiment", str(write_experiment(tmp_path))])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "determinista" in output
    assert "estocástica" not in output
    assert env.reset_seeds == [4]


def test_verbose_observer_reports_denormalized_target_and_applies_delay(
    monkeypatch,
    capsys,
):
    delays = []
    monkeypatch.setattr(eval_cli.time, "sleep", delays.append)
    observer = eval_cli.make_step_observer(OneStepEnv(), verbose=True, delay=0.2)
    record = StepRecord(
        episode=1,
        step=2,
        observation=np.array(
            [0.12345679, 0.0, 0.5, -0.5, 0.25],
            dtype=np.float32,
        ),
        action=np.array([0.5, -0.5], dtype=np.float32),
        reward=1.25,
        terminated=False,
        truncated=False,
        info={"distance": 8.0},
    )

    observer(record)

    assert delays == [0.2]
    output = capsys.readouterr().out
    assert "step  2" in output
    assert (
        "observation=[villager_x_norm=0.123456791, villager_z_norm=0, "
        "resource_x_norm=0.5, resource_z_norm=-0.5, distance_norm=0.25]"
        in output
    )
    assert "target=(384,128)" in output
    assert "dist=8.0" in output


def test_agent_view_observer_updates_before_pre_action_delay():
    events = []
    agent_view = FakeAgentView(events)
    observer = eval_cli.make_agent_view_observer(agent_view, delay=0.2)
    decision = DecisionRecord(
        episode=0,
        step=0,
        observation=np.zeros(5, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
    )

    observer(decision)

    assert events[0][0] == "view"
    assert events[0][1] is decision
    assert events[1] == ("pause", 0.2)


def test_agent_view_observer_runs_without_delay():
    agent_view = FakeAgentView()
    observer = eval_cli.make_agent_view_observer(agent_view, delay=0.0)
    decision = DecisionRecord(
        episode=0,
        step=0,
        observation=np.zeros(5, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
    )

    observer(decision)

    assert agent_view.records == [decision]


def test_main_opens_updates_and_closes_opt_in_agent_view(
    tmp_path: Path,
    monkeypatch,
):
    env = OneStepEnv()
    agent_view = FakeAgentView()
    opened_for = []
    monkeypatch.setattr(eval_cli, "build_environment", lambda config, **kwargs: env)
    monkeypatch.setattr(
        eval_cli,
        "open_agent_view",
        lambda selected_env: opened_for.append(selected_env) or agent_view,
    )

    exit_code = eval_cli.main(
        ["--experiment", str(write_experiment(tmp_path)), "--agent-view"]
    )

    assert exit_code == 0
    assert opened_for == [env]
    assert len(agent_view.records) == 1
    assert agent_view.close_calls == 1
    assert env.closed is True


def test_main_closes_agent_view_when_evaluation_fails(tmp_path: Path, monkeypatch):
    env = OneStepEnv()
    agent_view = FakeAgentView()
    monkeypatch.setattr(eval_cli, "build_environment", lambda config, **kwargs: env)
    monkeypatch.setattr(eval_cli, "open_agent_view", lambda selected_env: agent_view)
    monkeypatch.setattr(
        eval_cli,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("evaluation failed")),
    )

    with pytest.raises(RuntimeError, match="evaluation failed"):
        eval_cli.main(
            ["--experiment", str(write_experiment(tmp_path)), "--agent-view"]
        )

    assert agent_view.close_calls == 1
    assert env.closed is True


def test_main_reports_agent_view_startup_failure_and_closes_env(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    env = OneStepEnv()
    monkeypatch.setattr(eval_cli, "build_environment", lambda config, **kwargs: env)

    def fail_to_open(selected_env):
        raise eval_cli.AgentViewUnavailable("no graphical display")

    monkeypatch.setattr(eval_cli, "open_agent_view", fail_to_open)

    with pytest.raises(SystemExit) as error:
        eval_cli.main(
            ["--experiment", str(write_experiment(tmp_path)), "--agent-view"]
        )

    assert error.value.code == 2
    assert "no graphical display" in capsys.readouterr().err
    assert env.closed is True


def test_learned_agent_without_model_has_an_actionable_cli_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    env = OneStepEnv()
    monkeypatch.setattr(
        eval_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )

    with pytest.raises(SystemExit) as error:
        eval_cli.main(["--experiment", str(write_experiment(tmp_path, SB3_EXPERIMENT))])

    assert error.value.code == 2
    assert "--model" in capsys.readouterr().err
    assert env.closed is False


def test_model_loading_requires_explicit_trust_acknowledgement(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    env = OneStepEnv()
    monkeypatch.setattr(
        eval_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )

    with pytest.raises(SystemExit) as error:
        eval_cli.main(
            [
                "--experiment",
                str(write_experiment(tmp_path, SB3_EXPERIMENT)),
                "--model",
                str(tmp_path / "downloaded-model.zip"),
            ]
        )

    assert error.value.code == 2
    assert "--trust-model" in capsys.readouterr().err
    assert env.closed is False


def test_trusted_model_is_loaded_through_the_registered_adapter(
    tmp_path: Path,
    monkeypatch,
):
    env = OneStepEnv()
    model_path = tmp_path / "model.zip"
    loaded = []
    monkeypatch.setattr(
        eval_cli,
        "build_environment",
        lambda config, **kwargs: env,
    )
    monkeypatch.setattr(
        eval_cli,
        "load_policy",
        lambda agent, path: loaded.append((agent.name, path)) or GatherOraclePolicy(),
    )

    exit_code = eval_cli.main(
        [
            "--experiment",
            str(write_experiment(tmp_path, SB3_EXPERIMENT)),
            "--model",
            str(model_path),
            "--trust-model",
            "--mode",
            "deterministic",
        ]
    )

    assert exit_code == 0
    assert loaded == [("sb3_sac", model_path)]
    assert env.closed is True
