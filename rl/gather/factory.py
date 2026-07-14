"""Factories for constructing gather environments from tracked scenarios."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from .env import ZeroADGatherEnv


def make_gather_env(
    scenario_path: str | PathLike[str], **env_parameters: Any
) -> ZeroADGatherEnv:
    """Read a scenario config and forward its environment parameters."""
    scenario_config = Path(scenario_path).read_text(encoding="utf-8")
    return ZeroADGatherEnv(scenario_config, **env_parameters)
