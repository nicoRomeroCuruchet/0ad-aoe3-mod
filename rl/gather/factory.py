"""Factories for constructing gather environments from tracked scenarios."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

from .env import ZeroADGatherEnv
from .team_env import ZeroADTeamGatherEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = REPO_ROOT / "rl"
MAX_SCENARIO_BYTES = 1024 * 1024


def _validated_scenario_path(scenario_path: str | PathLike[str]) -> Path:
    scenario_root = SCENARIO_ROOT.resolve()
    supplied_path = Path(scenario_path)
    candidate = (
        supplied_path
        if supplied_path.is_absolute()
        else REPO_ROOT.resolve() / supplied_path
    )
    resolved = candidate.resolve(strict=True)

    if not resolved.is_relative_to(scenario_root):
        raise ValueError("scenario path must stay inside the repository RL directory")
    relative_path = resolved.relative_to(scenario_root)
    if len(relative_path.parts) > 1 and relative_path.parts[0] != "scenarios":
        raise ValueError(
            "nested scenarios must be stored under rl/scenarios",
        )
    if resolved.suffix.lower() != ".json":
        raise ValueError("scenario must be a JSON file")
    if not resolved.is_file():
        raise ValueError("scenario must be a regular file")
    if resolved.stat().st_size > MAX_SCENARIO_BYTES:
        raise ValueError(
            f"scenario is too large (maximum {MAX_SCENARIO_BYTES} bytes)",
        )
    return resolved


def make_gather_env(
    scenario_path: str | PathLike[str], **env_parameters: Any
) -> ZeroADGatherEnv:
    """Read a repo-owned scenario config and forward environment parameters."""

    validated_path = _validated_scenario_path(scenario_path)
    scenario_bytes = validated_path.read_bytes()
    if len(scenario_bytes) > MAX_SCENARIO_BYTES:
        raise ValueError(
            f"scenario is too large (maximum {MAX_SCENARIO_BYTES} bytes)",
        )
    scenario_config = scenario_bytes.decode("utf-8")
    try:
        json.loads(scenario_config)
    except json.JSONDecodeError as error:
        raise ValueError("scenario must contain valid JSON") from error
    return ZeroADGatherEnv(scenario_config, **env_parameters)


def make_team_gather_env(
    scenario_path: str | PathLike[str], **env_parameters: Any
) -> ZeroADTeamGatherEnv:
    """Read a repo-owned scenario config and forward team parameters."""

    validated_path = _validated_scenario_path(scenario_path)
    scenario_bytes = validated_path.read_bytes()
    if len(scenario_bytes) > MAX_SCENARIO_BYTES:
        raise ValueError(
            f"scenario is too large (maximum {MAX_SCENARIO_BYTES} bytes)",
        )
    scenario_config = scenario_bytes.decode("utf-8")
    try:
        json.loads(scenario_config)
    except json.JSONDecodeError as error:
        raise ValueError("scenario must contain valid JSON") from error
    return ZeroADTeamGatherEnv(scenario_config, **env_parameters)
