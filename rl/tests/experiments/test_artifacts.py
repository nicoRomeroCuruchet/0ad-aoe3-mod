import json
from pathlib import Path

import numpy as np
import pytest

from rl.agents.base import AgentSpec, EpisodeResult
from rl.experiments.artifacts import (
    create_run_artifacts,
    record_run,
    write_json,
)
from rl.experiments.config import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from rl.experiments.evaluation import EvaluationReport


def test_create_run_artifacts_makes_a_predictable_isolated_layout(tmp_path: Path):
    artifacts = create_run_artifacts(
        tmp_path,
        "m0-sb3-sac",
        run_id="20260714T120000Z",
    )

    assert artifacts.run_dir == tmp_path / "m0-sb3-sac-20260714T120000Z"
    assert artifacts.run_dir.is_dir()
    assert artifacts.model_path == artifacts.run_dir / "model"
    assert artifacts.best_model_path == artifacts.run_dir / "best_model"
    assert artifacts.config_path == artifacts.run_dir / "resolved_config.json"
    assert artifacts.metrics_path == artifacts.run_dir / "metrics.json"
    assert artifacts.metadata_path == artifacts.run_dir / "metadata.json"
    assert artifacts.training_log_dir == artifacts.run_dir / "training"


@pytest.mark.parametrize("run_name", ["", "../escape", "has spaces", "/absolute"])
def test_create_run_artifacts_rejects_unsafe_run_names(
    tmp_path: Path,
    run_name: str,
):
    with pytest.raises(ValueError, match="run name"):
        create_run_artifacts(tmp_path, run_name, run_id="safe")


def test_create_run_artifacts_refuses_to_reuse_a_run_directory(tmp_path: Path):
    create_run_artifacts(tmp_path, "experiment", run_id="same")

    with pytest.raises(FileExistsError):
        create_run_artifacts(tmp_path, "experiment", run_id="same")


def test_write_json_creates_human_readable_deterministic_output(tmp_path: Path):
    destination = tmp_path / "nested" / "metrics.json"

    write_json(destination, {"successes": 2, "rewards": [1.5, 2.0]})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "rewards": [1.5, 2.0],
        "successes": 2,
    }
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_write_json_redacts_common_secret_fields(tmp_path: Path):
    destination = tmp_path / "config.json"

    write_json(
        destination,
        {
            "agent": {
                "api_key": "do-not-write-this",
                "access_token": "also-secret",
                "learning_rate": 0.1,
                "callbacks": [{"password": "nested-secret"}],
            }
        },
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "agent": {
            "access_token": "<redacted>",
            "api_key": "<redacted>",
            "callbacks": [{"password": "<redacted>"}],
            "learning_rate": 0.1,
        }
    }


def test_record_run_serializes_resolved_config_metrics_and_metadata(tmp_path: Path):
    artifacts = create_run_artifacts(tmp_path, "experiment", run_id="one")
    config = ExperimentConfig(
        environment=EnvironmentConfig(
            name="zero_ad_gather",
            parameters={"scenario": "rl/reset_config.json", "horizon": 50},
        ),
        agent=AgentSpec(
            name="sb3_sac",
            parameters={"policy": "MlpPolicy", "net_arch": [64, 64]},
        ),
        training=TrainingConfig(
            total_steps=1_000,
            seed=7,
            log_interval=3,
            solved_window_episodes=20,
            solved_success_rate=0.8,
            solved_min_steps=500,
            solved_check_interval_steps=250,
        ),
        evaluation=EvaluationConfig(episodes=1, deterministic=True, seed=8),
    )
    report = EvaluationReport(
        episodes=(
            EpisodeResult(
                episode=0,
                total_reward=4.5,
                steps=3,
                terminated=True,
                truncated=False,
                final_info={
                    "distance": np.float32(1.25),
                    "position": np.array([2.0, 3.0], dtype=np.float32),
                },
            ),
        )
    )

    record_run(
        artifacts,
        config,
        report,
        metadata={"model_path": "rl/runs/example/model"},
    )

    assert json.loads(artifacts.config_path.read_text(encoding="utf-8")) == {
        "agent": {
            "name": "sb3_sac",
            "net_arch": [64, 64],
            "policy": "MlpPolicy",
        },
        "environment": {
            "horizon": 50,
            "name": "zero_ad_gather",
            "scenario": "rl/reset_config.json",
        },
        "evaluation": {"deterministic": True, "episodes": 1, "seed": 8},
        "training": {
            "log_interval": 3,
            "seed": 7,
            "solved_min_steps": 500,
            "solved_check_interval_steps": 250,
            "solved_success_rate": 0.8,
            "solved_window_episodes": 20,
            "total_steps": 1_000,
        },
    }
    assert json.loads(artifacts.metrics_path.read_text(encoding="utf-8")) == {
        "episodes": [
            {
                "episode": 0,
                "final_info": {"distance": 1.25, "position": [2.0, 3.0]},
                "steps": 3,
                "terminated": True,
                "total_reward": 4.5,
                "truncated": False,
            }
        ],
        "summary": {
            "episode_count": 1,
            "mean_steps": 3.0,
            "mean_total_reward": 4.5,
            "success_rate": 1.0,
        },
    }
    assert json.loads(artifacts.metadata_path.read_text(encoding="utf-8")) == {
        "model_path": "rl/runs/example/model"
    }
