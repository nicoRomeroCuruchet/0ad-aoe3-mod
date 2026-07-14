"""Filesystem layout for reproducible experiment outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import ExperimentConfig
from .evaluation import EvaluationReport


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
    }
)


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
        config_path=run_dir / "resolved_config.json",
        metrics_path=run_dir / "metrics.json",
        metadata_path=run_dir / "metadata.json",
    )


def write_json(destination: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a stable, human-readable JSON document."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_plain_value(payload), indent=2, sort_keys=True) + "\n"
    destination_path.write_text(serialized, encoding="utf-8")


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if _is_sensitive_key(key) else _plain_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_password", "_secret", "_token")
    )


def _config_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "environment": {
            "name": config.environment.name,
            **_plain_value(config.environment.parameters),
        },
        "agent": {
            "name": config.agent.name,
            **_plain_value(config.agent.parameters),
        },
        "training": {
            "total_steps": config.training.total_steps,
            "seed": config.training.seed,
        },
        "evaluation": {
            "episodes": config.evaluation.episodes,
            "deterministic": config.evaluation.deterministic,
            "seed": config.evaluation.seed,
        },
    }


def _metrics_payload(report: EvaluationReport) -> dict[str, Any]:
    episodes = [
        {
            "episode": result.episode,
            "total_reward": result.total_reward,
            "steps": result.steps,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "final_info": _plain_value(result.final_info),
        }
        for result in report.episodes
    ]
    return {
        "summary": {
            "episode_count": report.episode_count,
            "mean_total_reward": report.mean_total_reward,
            "mean_steps": report.mean_steps,
            "success_rate": report.success_rate,
        },
        "episodes": episodes,
    }


def record_run(
    artifacts: RunArtifacts,
    config: ExperimentConfig,
    report: EvaluationReport,
    *,
    metadata: Mapping[str, Any],
) -> None:
    """Persist the resolved config, evaluation metrics, and run metadata."""

    record_run_context(artifacts, config, metadata=metadata)
    record_evaluation(artifacts, report)


def record_run_context(
    artifacts: RunArtifacts,
    config: ExperimentConfig,
    *,
    metadata: Mapping[str, Any],
) -> None:
    """Persist reproducibility context independently of final evaluation."""

    write_json(artifacts.config_path, _config_payload(config))
    write_json(artifacts.metadata_path, metadata)


def record_evaluation(
    artifacts: RunArtifacts,
    report: EvaluationReport,
) -> None:
    """Persist evaluation metrics after the checkpoint is already durable."""

    write_json(artifacts.metrics_path, _metrics_payload(report))
