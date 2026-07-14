"""Filesystem layout for reproducible experiment outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunArtifacts:
    """Paths owned by one training run."""

    run_dir: Path
    model_path: Path
    config_path: Path
    metrics_path: Path
    metadata_path: Path


def create_run_artifacts(
    root: str | Path,
    run_name: str,
    *,
    run_id: str | None = None,
) -> RunArtifacts:
    """Create a fresh run directory and return its conventional paths."""

    if not _SAFE_COMPONENT.fullmatch(run_name):
        raise ValueError(
            "run name must contain only letters, numbers, '.', '_' or '-'",
        )

    selected_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if not _SAFE_COMPONENT.fullmatch(selected_run_id):
        raise ValueError("run id contains unsafe path characters")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    run_dir = root_path / f"{run_name}-{selected_run_id}"
    run_dir.mkdir(exist_ok=False)

    return RunArtifacts(
        run_dir=run_dir,
        model_path=run_dir / "model",
        config_path=run_dir / "experiment.toml",
        metrics_path=run_dir / "metrics.json",
        metadata_path=run_dir / "metadata.json",
    )


def write_json(destination: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a stable, human-readable JSON document."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination_path.write_text(serialized, encoding="utf-8")
