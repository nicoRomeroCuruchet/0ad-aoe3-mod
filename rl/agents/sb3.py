"""Stable-Baselines3 adapters for the repo's small agent contracts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import Policy, TrainRequest


class SB3DependencyError(RuntimeError):
    """Raised when an SB3 agent is selected without its optional dependency."""


def _load_sac_class() -> type[Any]:
    try:
        module = import_module("stable_baselines3")
    except ModuleNotFoundError as error:
        raise SB3DependencyError(
            "Stable-Baselines3 is required for 'sb3_sac'; "
            "run 'uv sync --locked' before using this agent",
        ) from error
    return module.SAC


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class SB3Policy:
    """Expose an SB3 model through the library-independent Policy API."""

    model: Any

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> np.ndarray:
        action, _state = self.model.predict(
            observation,
            deterministic=deterministic,
        )
        return np.asarray(action, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))


class SB3SACTrainer:
    """Train SAC while keeping SB3 details out of experiment orchestration."""

    def fit(self, request: TrainRequest) -> Policy:
        sac_class = _load_sac_class()
        parameters = _thaw(request.agent.parameters)
        policy_name = parameters.pop("policy", "MlpPolicy")
        parameters["seed"] = request.seed
        model = sac_class(policy_name, request.env, **parameters)
        model.learn(total_timesteps=request.total_steps)
        return SB3Policy(model)


def load_sb3_sac_policy(path: str | Path) -> SB3Policy:
    """Load a trusted serialized SAC model as a common Policy.

    SB3 checkpoints can contain cloudpickled Python objects. Callers must not
    pass files from an untrusted source because loading can execute code.
    """

    model = _load_sac_class().load(str(path))
    return SB3Policy(model)
