from pathlib import Path

import numpy as np
import pytest

import rl.eval as eval_cli
from rl.agents.baselines import GatherOraclePolicy
from rl.experiments.evaluation import DecisionRecord, StepRecord
from rl.gather.engine_observer import EngineObserverFrame, EngineObserverUnavailable
from rl.gather.rollout_recording import AgentViewRolloutRecorder


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

    def capture_agent_frame(self):
        return EngineObserverFrame(1, 1, b"P6\n1 1\n255\n\x10\x20\x30")

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
        info={
            "distance": 8.0,
            "resource_stock": 14.0,
            "resource_stock_delta": 4.0,
            "carried_resource": 10.0,
            "carried_resource_delta": 10.0,
            "carried_resource_delta_reward": 2.0,
            "gather_cycle_no_click_reward": 0.02,
            "click_gather_cycle_penalty": 1.0,
        },
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
    assert "stock=14.0 dstock=+4.0" in output
    assert "carried=10.0 dcarried=+10.0" in output
    assert "parts=[carry=+2.00 wait=+0.02 interrupt=-1.00]" in output


def test_verbose_observer_reports_assignment_categories(capsys):
    env = OneStepEnv()
    env.action_mode = "assignment_click"
    observer = eval_cli.make_step_observer(env, verbose=True, delay=0.0)
    record = StepRecord(
        episode=0,
        step=1,
        observation=np.zeros(5, dtype=np.float32),
        action=np.array([0, 2, 1, 0], dtype=np.int64),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )

    observer(record)

    assert "assignments=['NO_CLICK', 'TREE_1', 'TREE_0', 'NO_CLICK']" in (
        capsys.readouterr().out
    )


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


def test_main_records_agent_view_rollout_without_opening_tk(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    env = OneStepEnv()
    output_dir = tmp_path / "rollout"
    monkeypatch.setattr(eval_cli, "build_environment", lambda config, **kwargs: env)

    exit_code = eval_cli.main(
        [
            "--experiment",
            str(write_experiment(tmp_path)),
            "--record-agent-view",
            str(output_dir),
        ]
    )

    index = output_dir / "deterministic/index.html"
    metadata = output_dir / "deterministic/metadata.jsonl"
    frame = output_dir / "deterministic/frames/ep000-step0000.png"
    assert exit_code == 0
    assert f"rollout: {index}" in capsys.readouterr().out
    assert index.exists()
    assert metadata.read_text(encoding="utf-8").count("\n") == 1
    assert frame.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert env.closed is True


def test_rollout_recorder_retries_transient_observer_unavailability(
    tmp_path: Path,
    monkeypatch,
):
    sleeps = []

    class FlakyCaptureEnv:
        def __init__(self):
            self.calls = 0

        def capture_agent_frame(self):
            self.calls += 1
            if self.calls == 1:
                raise EngineObserverUnavailable("renderer warming up")
            return EngineObserverFrame(1, 1, b"P6\n1 1\n255\n\x10\x20\x30")

    monkeypatch.setattr("rl.gather.rollout_recording.time.sleep", sleeps.append)
    env = FlakyCaptureEnv()
    recorder = AgentViewRolloutRecorder(
        tmp_path / "rollout",
        env,
        capture_attempts=2,
        capture_retry_delay=0.1,
    )

    recorder.observe(
        DecisionRecord(
            episode=0,
            step=0,
            observation=np.zeros(5, dtype=np.float32),
            action=np.zeros(3, dtype=np.float32),
        )
    )

    assert env.calls == 2
    assert sleeps == [0.1]
    assert (tmp_path / "rollout/frames/ep000-step0000.png").exists()


def test_rollout_recorder_falls_back_to_schematic_frame(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    class UnavailableCaptureEnv:
        def capture_agent_frame(self):
            raise EngineObserverUnavailable("no renderer")

    monkeypatch.setattr("rl.gather.rollout_recording.time.sleep", lambda _delay: None)
    recorder = AgentViewRolloutRecorder(
        tmp_path / "rollout",
        UnavailableCaptureEnv(),
        capture_attempts=2,
        fallback_to_schematic=True,
    )

    recorder.observe(
        DecisionRecord(
            episode=0,
            step=0,
            observation=np.array([0.0, 0.0, 0.5, -0.5, 0.25], dtype=np.float32),
            action=np.array([0.5, -0.5, 1.0], dtype=np.float32),
        )
    )

    metadata = (tmp_path / "rollout/metadata.jsonl").read_text(encoding="utf-8")
    assert '"frame_source": "schematic"' in metadata
    assert (tmp_path / "rollout/frames/ep000-step0000.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert "recording schematic" in capsys.readouterr().out


def test_rollout_recorder_requires_engine_frames_by_default(tmp_path: Path):
    class UnavailableCaptureEnv:
        def capture_agent_frame(self):
            raise EngineObserverUnavailable("no renderer")

    recorder = AgentViewRolloutRecorder(
        tmp_path / "rollout",
        UnavailableCaptureEnv(),
        capture_attempts=1,
    )

    with pytest.raises(EngineObserverUnavailable, match="did not produce a frame"):
        recorder.observe(
            DecisionRecord(
                episode=0,
                step=0,
                observation=np.zeros(5, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
            )
        )


def test_rollout_recorder_encodes_recorded_frames_to_mp4(
    tmp_path: Path,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        "rl.gather.rollout_recording.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    monkeypatch.setattr(
        "rl.gather.rollout_recording.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    recorder = AgentViewRolloutRecorder(tmp_path / "rollout", OneStepEnv())
    recorder.observe(
        DecisionRecord(
            episode=0,
            step=0,
            observation=np.zeros(5, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
        )
    )

    video_path = recorder.write_video(tmp_path / "rollout.mp4")

    assert video_path == tmp_path / "rollout.mp4"
    assert calls == [
        (
            [
                "/usr/bin/ffmpeg",
                "-y",
                "-framerate",
                "4",
                "-pattern_type",
                "glob",
                "-i",
                str(tmp_path / "rollout/frames/*.png"),
                "-pix_fmt",
                "yuv420p",
                str(tmp_path / "rollout.mp4"),
            ],
            {"check": True},
        )
    ]


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


def test_rollout_recorder_writes_one_frame_per_simulation_turn(tmp_path: Path):
    class CaptureEnv:
        def capture_agent_frame(self):
            return EngineObserverFrame(1, 1, b"P6\n1 1\n255\n\x10\x20\x30")

    recorder = AgentViewRolloutRecorder(tmp_path / "rollout", CaptureEnv())

    for _ in range(3):
        recorder.observe_sim_turn()

    assert recorder.turn_frame_count == 3
    frames = sorted(path.name for path in (tmp_path / "rollout/turn_frames").iterdir())
    assert frames == ["turn00000.png", "turn00001.png", "turn00002.png"]
    assert (tmp_path / "rollout/turn_frames/turn00000.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_rollout_video_uses_gstreamer_when_ffmpeg_is_missing(
    tmp_path: Path,
    monkeypatch,
):
    class CaptureEnv:
        def capture_agent_frame(self):
            return EngineObserverFrame(1, 1, b"P6\n1 1\n255\n\x10\x20\x30")

    commands = []
    monkeypatch.setattr(
        "rl.gather.rollout_recording.shutil.which",
        lambda name: None if name == "ffmpeg" else "/usr/bin/gst-launch-1.0",
    )
    monkeypatch.setattr(
        "rl.gather.rollout_recording.subprocess.run",
        lambda command, check: commands.append(command),
    )
    recorder = AgentViewRolloutRecorder(tmp_path / "rollout", CaptureEnv())
    recorder.observe_sim_turn()

    written = recorder.write_video(tmp_path / "rollout.mp4")

    assert written == tmp_path / "rollout.webm"
    assert commands[0][0] == "/usr/bin/gst-launch-1.0"
    assert "vp8enc" in commands[0]
    assert f"location={tmp_path / 'rollout/turn_frames/turn%05d.png'}" in commands[0]


def test_rollout_video_prefers_ffmpeg_and_per_turn_frames(tmp_path: Path, monkeypatch):
    class CaptureEnv:
        def capture_agent_frame(self):
            return EngineObserverFrame(1, 1, b"P6\n1 1\n255\n\x10\x20\x30")

    commands = []
    monkeypatch.setattr(
        "rl.gather.rollout_recording.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    monkeypatch.setattr(
        "rl.gather.rollout_recording.subprocess.run",
        lambda command, check: commands.append(command),
    )
    recorder = AgentViewRolloutRecorder(tmp_path / "rollout", CaptureEnv())
    recorder.observe_sim_turn()

    written = recorder.write_video(tmp_path / "rollout.mp4")

    assert written == tmp_path / "rollout.mp4"
    assert str(tmp_path / "rollout/turn_frames/*.png") in commands[0]
    assert "20" in commands[0]
