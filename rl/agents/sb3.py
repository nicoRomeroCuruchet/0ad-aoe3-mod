"""Stable-Baselines3 adapters for the repo's small agent contracts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import numpy as np

from .base import Policy, TrainRequest


class SB3DependencyError(RuntimeError):
    """Raised when an SB3 agent is selected without its optional dependency."""


BEST_MODEL_EPISODES = 10


def _load_sac_class() -> type[Any]:
    try:
        module = import_module("stable_baselines3")
    except ModuleNotFoundError as error:
        raise SB3DependencyError(
            "Stable-Baselines3 is required for 'sb3_sac'; "
            "run 'uv sync --locked' before using this agent",
        ) from error
    return module.SAC


def _load_logger_configure() -> Any:
    try:
        module = import_module("stable_baselines3.common.logger")
    except ModuleNotFoundError as error:
        raise SB3DependencyError(
            "Stable-Baselines3 is required for 'sb3_sac'; "
            "run 'uv sync --locked' before using this agent",
        ) from error
    return module.configure


def _load_base_callback_class() -> type[Any]:
    try:
        module = import_module("stable_baselines3.common.callbacks")
    except ModuleNotFoundError as error:
        raise SB3DependencyError(
            "Stable-Baselines3 is required for 'sb3_sac'; "
            "run 'uv sync --locked' before using this agent",
        ) from error
    return module.BaseCallback


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _episode_rewards(infos: object) -> tuple[float, ...]:
    if not isinstance(infos, list):
        return ()
    rewards = []
    for info in infos:
        if not isinstance(info, Mapping):
            continue
        episode = info.get("episode")
        if isinstance(episode, Mapping) and "r" in episode:
            rewards.append(float(episode["r"]))
    return tuple(rewards)


def _replay_buffer_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.suffix == ".zip":
        checkpoint = checkpoint.with_suffix("")
    return checkpoint.with_name(f"{checkpoint.name}.replay_buffer.pkl")


def _save_checkpoint(model: Any, path: str | Path) -> None:
    model.save(str(path))
    model.save_replay_buffer(str(_replay_buffer_path(path)))


def _build_best_reward_callback(path: Path) -> Any:
    base_callback = _load_base_callback_class()

    class BestRewardCheckpointCallback(base_callback):
        def __init__(self, destination: Path) -> None:
            super().__init__()
            self.destination = destination
            self.pending_rewards: tuple[float, ...] = ()
            self.best_mean_reward = float("-inf")

        def _on_step(self) -> bool:
            for reward in _episode_rewards(self.locals.get("infos")):
                self.pending_rewards = (*self.pending_rewards, reward)
                if len(self.pending_rewards) < BEST_MODEL_EPISODES:
                    continue
                mean_reward = fmean(self.pending_rewards)
                self.pending_rewards = ()
                if mean_reward <= self.best_mean_reward:
                    continue
                self.best_mean_reward = mean_reward
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                _save_checkpoint(self.model, self.destination)
                print(
                    f"best_model: {self.destination} "
                    f"mean_episode_reward={mean_reward:.6g} "
                    f"episodes={BEST_MODEL_EPISODES}",
                    flush=True,
                )
            return True

    return BestRewardCheckpointCallback(path)


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
        _save_checkpoint(self.model, path)


class SB3SACTrainer:
    """Train SAC while keeping SB3 details out of experiment orchestration."""

    def fit(self, request: TrainRequest) -> Policy:
        sac_class = _load_sac_class()
        parameters = _thaw(request.agent.parameters)
        policy_name = parameters.pop("policy", "MlpPolicy")
        parameters["seed"] = request.seed
        if request.resume_from is None:
            model = sac_class(policy_name, request.env, **parameters)
            learn_kwargs = {}
        else:
            # SB3 restores the serialized policy architecture itself and checks
            # policy_kwargs for exact equality, including its injected defaults.
            parameters.pop("policy_kwargs", None)
            model = sac_class.load(
                str(request.resume_from),
                env=request.env,
                **parameters,
            )
            replay_buffer = _replay_buffer_path(request.resume_from)
            if replay_buffer.is_file():
                model.load_replay_buffer(str(replay_buffer))
            else:
                warmup_steps = int(getattr(model, "learning_starts", 0))
                model.learning_starts = int(model.num_timesteps) + warmup_steps
                print(
                    f"replay_buffer: {replay_buffer} not found; "
                    f"collecting {warmup_steps} fresh transitions before updates",
                    flush=True,
                )
            set_random_seed = getattr(model, "set_random_seed", None)
            if callable(set_random_seed):
                set_random_seed(request.seed)
            learn_kwargs = {"reset_num_timesteps": False}
        if request.log_dir is not None:
            request.log_dir.mkdir(parents=True, exist_ok=True)
            configure_logger = _load_logger_configure()
            model.set_logger(
                configure_logger(str(request.log_dir), ["stdout", "csv", "json"])
            )
        model.learn(
            total_timesteps=request.total_steps,
            log_interval=request.log_interval,
            callback=(
                None
                if request.best_model_path is None
                else _build_best_reward_callback(request.best_model_path)
            ),
            **learn_kwargs,
        )
        return SB3Policy(model)


def load_sb3_sac_policy(path: str | Path) -> SB3Policy:
    """Load a trusted serialized SAC model as a common Policy.

    SB3 checkpoints can contain cloudpickled Python objects. Callers must not
    pass files from an untrusted source because loading can execute code.
    """

    model = _load_sac_class().load(str(path))
    return SB3Policy(model)
