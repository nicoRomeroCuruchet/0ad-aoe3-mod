import json
from pathlib import Path

import pytest

from rl.experiments.artifacts import create_run_artifacts, write_json


def test_create_run_artifacts_makes_a_predictable_isolated_layout(tmp_path: Path):
    artifacts = create_run_artifacts(
        tmp_path,
        "m0-sb3-sac",
        run_id="20260714T120000Z",
    )

    assert artifacts.run_dir == tmp_path / "m0-sb3-sac-20260714T120000Z"
    assert artifacts.run_dir.is_dir()
    assert artifacts.model_path == artifacts.run_dir / "model"
    assert artifacts.config_path == artifacts.run_dir / "experiment.toml"
    assert artifacts.metrics_path == artifacts.run_dir / "metrics.json"
    assert artifacts.metadata_path == artifacts.run_dir / "metadata.json"


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
