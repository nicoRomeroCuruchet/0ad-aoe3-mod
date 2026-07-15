import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_dry_run(*arguments: str) -> str:
    result = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _fake_uv_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    capture_path = tmp_path / "uv-arguments.txt"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$UV_CAPTURE"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["UV"] = "uv"
    environment["UV_CAPTURE"] = str(capture_path)
    return environment, capture_path


def _run_with_fake_uv(tmp_path: Path, *arguments: str) -> list[str]:
    environment, capture_path = _fake_uv_environment(tmp_path)
    subprocess.run(
        ["make", "--no-print-directory", *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return capture_path.read_text(encoding="utf-8").splitlines()


def test_make_targets_are_thin_wrappers_around_the_canonical_tools():
    assert "uv sync --locked" in _make_dry_run("setup")
    assert "uv run --locked pytest --cov" in _make_dry_run("test")
    assert "uv run --locked ruff check rl" in _make_dry_run("lint")
    assert (
        "./run_game.sh --require-rl-observer --rl-interface=127.0.0.1:6000"
        in _make_dry_run("server")
    )
    assert "./engine/build_observer.sh" in _make_dry_run("engine-observer")


def test_make_agent_targets_forward_parameters(tmp_path):
    oracle = _run_with_fake_uv(
        tmp_path, "oracle", "EPISODES=3", "ARGS=--delay 0.5 --verbose"
    )
    training = _run_with_fake_uv(
        tmp_path, "train", "STEPS=321", "ARGS=--out /tmp/model"
    )

    assert oracle[-5:] == ["--episodes", "3", "--delay", "0.5", "--verbose"]
    assert training[-5:] == [
        "--agent-view",
        "--timesteps",
        "321",
        "--out",
        "/tmp/model",
    ]


def test_make_agent_targets_preserve_config_defaults(tmp_path):
    assert "--episodes" not in _run_with_fake_uv(tmp_path, "oracle")
    assert "--timesteps" not in _run_with_fake_uv(tmp_path, "train")


def test_make_train_can_disable_the_default_agent_view(tmp_path):
    training = _run_with_fake_uv(tmp_path, "train", "NO_AGENT_VIEW=1")

    assert "--agent-view" not in training
    assert "--delay" not in training


def test_make_train_rejects_view_arguments_in_headless_mode(tmp_path):
    environment, capture_path = _fake_uv_environment(tmp_path)

    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "train",
            "NO_AGENT_VIEW=1",
            "ARGS=--agent-view --delay 0.5",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cannot include --agent-view or --delay" in result.stderr
    assert not capture_path.exists()


def test_make_eval_requires_and_quotes_a_model_path(tmp_path):
    missing_model = subprocess.run(
        ["make", "--no-print-directory", "eval"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing_model.returncode != 0
    assert "MODEL=path/to/model" in missing_model.stderr

    untrusted_model = subprocess.run(
        ["make", "--no-print-directory", "eval", "MODEL=rl/runs/my-run/model"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert untrusted_model.returncode != 0
    assert "TRUST_MODEL=1" in untrusted_model.stderr

    evaluation = _run_with_fake_uv(
        tmp_path,
        "eval",
        "MODEL=rl/runs/my run/model",
        "TRUST_MODEL=1",
        "EPISODES=2",
        "ARGS=--mode both",
    )
    model_index = evaluation.index("--model")
    assert evaluation[model_index + 1] == "rl/runs/my run/model"
    assert evaluation[-5:] == ["--trust-model", "--episodes", "2", "--mode", "both"]


def test_make_runtime_values_cannot_inject_shell_commands(tmp_path):
    environment, capture_path = _fake_uv_environment(tmp_path)
    marker = tmp_path / "injected"

    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "oracle",
            "EPISODES=2",
            f"ARGS=--mode both; touch {marker}",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not marker.exists()
    assert "both;" in capture_path.read_text(encoding="utf-8").splitlines()


def test_make_runtime_values_do_not_expand_make_functions(tmp_path):
    environment, capture_path = _fake_uv_environment(tmp_path)
    marker = tmp_path / "make-expanded"

    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "eval",
            f"MODEL=$(shell touch {marker})rl/runs/model",
            "TRUST_MODEL=1",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not marker.exists()
    assert "$(shell touch" in capture_path.read_text(encoding="utf-8")
