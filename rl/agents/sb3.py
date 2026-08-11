"""Stable-Baselines3 adapters for the repo's small agent contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
import json
import math
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from .base import Policy, TrainingOutcome, TrainRequest


class SB3DependencyError(RuntimeError):
    """Raised when an SB3 agent is selected without its optional dependency."""


BEST_MODEL_EPISODES = 10
CHECKPOINT_MANIFEST_VERSION = 1


def _import_sb3(module_name: str, agent_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        raise SB3DependencyError(
            f"Stable-Baselines3 is required for '{agent_name}'; "
            "run 'uv sync --locked' before using this agent",
        ) from error


def _load_sac_class() -> type[Any]:
    return _import_sb3("stable_baselines3", "sb3_sac").SAC


def _load_ppo_class() -> type[Any]:
    return _import_sb3("stable_baselines3", "sb3_ppo").PPO


def _load_logger_configure() -> Any:
    return _import_sb3("stable_baselines3.common.logger", "sb3_sac").configure


def _load_base_callback_class() -> type[Any]:
    return _import_sb3("stable_baselines3.common.callbacks", "sb3_sac").BaseCallback


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


def _checkpoint_base(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.suffix == ".zip":
        checkpoint = checkpoint.with_suffix("")
    return checkpoint


def _model_archive_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.suffix:
        return checkpoint
    return checkpoint.with_name(f"{checkpoint.name}.zip")


def _replay_buffer_path(path: str | Path) -> Path:
    checkpoint = _checkpoint_base(path)
    return checkpoint.with_name(f"{checkpoint.name}.replay_buffer.pkl")


def _checkpoint_manifest_path(path: str | Path) -> Path:
    checkpoint = _checkpoint_base(path)
    return checkpoint.with_name(f"{checkpoint.name}.checkpoint.json")


def _checkpoint_pending_path(path: str | Path) -> Path:
    checkpoint = _checkpoint_base(path)
    return checkpoint.with_name(f"{checkpoint.name}.checkpoint.pending.json")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_payload(
    model: Any,
    archive: Path,
    replay: Path | None,
) -> dict[str, object]:
    num_timesteps = getattr(model, "num_timesteps", None)
    return {
        "version": CHECKPOINT_MANIFEST_VERSION,
        "status": "complete",
        "num_timesteps": (
            None if num_timesteps is None else int(num_timesteps)
        ),
        "model_sha256": _file_sha256(archive),
        "replay_buffer_sha256": None if replay is None else _file_sha256(replay),
    }


def _validate_checkpoint_pair(path: str | Path) -> int | None:
    archive = _model_archive_path(path)
    replay = _replay_buffer_path(path)
    manifest_path = _checkpoint_manifest_path(path)
    pending_path = _checkpoint_pending_path(path)
    if not manifest_path.is_file():
        if pending_path.exists():
            raise RuntimeError("checkpoint transaction is incomplete")
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("checkpoint manifest is unreadable") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint manifest must contain an object")
    version = payload.get("version")
    if (
        type(version) is not int
        or version != CHECKPOINT_MANIFEST_VERSION
        or payload.get("status") != "complete"
    ):
        raise RuntimeError("checkpoint manifest is incomplete or unsupported")
    replay_hash = payload.get("replay_buffer_sha256")
    if not archive.is_file() or (replay_hash is not None and not replay.is_file()):
        raise RuntimeError("checkpoint pair is incomplete")
    expected_hashes = {archive: payload.get("model_sha256")}
    if replay_hash is not None:
        expected_hashes[replay] = replay_hash
    for artifact, expected_hash in expected_hashes.items():
        if not isinstance(expected_hash, str) or _file_sha256(artifact) != expected_hash:
            raise RuntimeError(f"checkpoint checksum mismatch for {artifact.name}")
    num_timesteps = payload.get("num_timesteps")
    if num_timesteps is None:
        return None
    if (
        not isinstance(num_timesteps, int)
        or isinstance(num_timesteps, bool)
        or num_timesteps < 0
    ):
        raise RuntimeError("checkpoint manifest has an invalid timestep count")
    return num_timesteps


def _save_checkpoint(model: Any, path: str | Path) -> None:
    archive = _model_archive_path(path)
    checkpoint = _checkpoint_base(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    temporary_archive = checkpoint.with_name(
        f".{checkpoint.name}.{generation}.zip"
    )
    temporary_replay = checkpoint.with_name(
        f".{checkpoint.name}.{generation}.replay_buffer.pkl"
    )
    pending_path = _checkpoint_pending_path(checkpoint)
    pending_preexisted = pending_path.exists()
    replacements_started = False
    # On-policy algorithms such as PPO keep no replay buffer to serialize.
    save_replay_buffer = getattr(model, "save_replay_buffer", None)
    try:
        model.save(str(temporary_archive))
        if save_replay_buffer is not None:
            save_replay_buffer(str(temporary_replay))
        if not temporary_archive.is_file() or (
            save_replay_buffer is not None and not temporary_replay.is_file()
        ):
            raise RuntimeError("checkpoint writer did not create both artifacts")
        payload = _checkpoint_payload(
            model,
            temporary_archive,
            temporary_replay if save_replay_buffer is not None else None,
        )
        _write_json_atomically(
            pending_path,
            {**payload, "status": "pending"},
        )
        replacements_started = True
        temporary_archive.replace(archive)
        if save_replay_buffer is not None:
            temporary_replay.replace(_replay_buffer_path(checkpoint))
        else:
            _replay_buffer_path(checkpoint).unlink(missing_ok=True)
        _write_json_atomically(_checkpoint_manifest_path(checkpoint), payload)
        pending_path.unlink(missing_ok=True)
    except BaseException:
        if not replacements_started and not pending_preexisted:
            pending_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_replay.unlink(missing_ok=True)


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
    training_outcome: TrainingOutcome | None = None

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


def _validated_success_rate(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise RuntimeError("solve evaluator must return a finite rate in [0, 1]")
    return float(value)


def _learn_once(
    model: Any,
    request: TrainRequest,
    callback: Any,
    *,
    reset_num_timesteps: bool,
) -> TrainingOutcome:
    start_steps = 0 if reset_num_timesteps else int(model.num_timesteps)
    model.learn(
        total_timesteps=request.total_steps,
        log_interval=request.log_interval,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    end_steps = int(model.num_timesteps)
    return TrainingOutcome(
        stop_reason="steps_complete",
        start_num_timesteps=start_steps,
        end_num_timesteps=end_steps,
        steps_this_run=end_steps - start_steps,
    )


def _run_solve_check(model: Any, request: TrainRequest) -> float:
    success_rate = _validated_success_rate(
        request.solve_evaluator(SB3Policy(model))
    )
    print(
        "solve_check: "
        f"success_rate={success_rate:.0%} "
        f"episodes={request.solved_window_episodes} "
        f"num_timesteps={int(model.num_timesteps)}",
        flush=True,
    )
    current_env = model.get_env()
    if current_env is None:
        raise RuntimeError("the model lost its training environment")
    model.set_env(current_env, force_reset=True)
    return success_rate


def _learn_until_solved(
    model: Any,
    request: TrainRequest,
    callback: Any,
    *,
    reset_num_timesteps: bool,
) -> TrainingOutcome:
    start_steps = 0 if reset_num_timesteps else int(model.num_timesteps)
    solve_checks = 0
    last_success_rate = None
    stop_reason = "safety_cap"

    if not reset_num_timesteps and int(model.num_timesteps) >= request.solved_min_steps:
        last_success_rate = _run_solve_check(model, request)
        solve_checks = 1
        if last_success_rate >= request.solved_success_rate:
            return TrainingOutcome(
                stop_reason="solved",
                start_num_timesteps=start_steps,
                end_num_timesteps=start_steps,
                steps_this_run=0,
                solve_checks=solve_checks,
                last_success_rate=last_success_rate,
            )

    while int(model.num_timesteps) - start_steps < request.total_steps:
        steps_completed = int(model.num_timesteps) - start_steps
        remaining_steps = request.total_steps - steps_completed
        tranche_steps = min(request.solved_check_interval_steps, remaining_steps)
        before_steps = int(model.num_timesteps)
        model.learn(
            total_timesteps=tranche_steps,
            log_interval=request.log_interval,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        reset_num_timesteps = False
        current_steps = int(model.num_timesteps)
        if current_steps <= before_steps:
            raise RuntimeError("training made no timestep progress")
        if request.checkpoint_path is not None:
            request.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(model, request.checkpoint_path)
        if current_steps < request.solved_min_steps:
            continue

        last_success_rate = _run_solve_check(model, request)
        solve_checks += 1
        if last_success_rate >= request.solved_success_rate:
            stop_reason = "solved"
            break

    end_steps = int(model.num_timesteps)
    return TrainingOutcome(
        stop_reason=stop_reason,
        start_num_timesteps=start_steps,
        end_num_timesteps=end_steps,
        steps_this_run=end_steps - start_steps,
        solve_checks=solve_checks,
        last_success_rate=last_success_rate,
    )


class SB3Trainer:
    """Train one SB3 algorithm while keeping SB3 out of the orchestration."""

    def _algorithm_class(self) -> type[Any]:
        raise NotImplementedError

    def fit(self, request: TrainRequest) -> Policy:
        algorithm_class = self._algorithm_class()
        parameters = _thaw(request.agent.parameters)
        policy_name = parameters.pop("policy", "MlpPolicy")
        parameters["seed"] = request.seed
        if request.resume_from is None:
            model = algorithm_class(policy_name, request.env, **parameters)
            reset_num_timesteps = True
        else:
            expected_num_timesteps = _validate_checkpoint_pair(request.resume_from)
            # SB3 restores the serialized policy architecture itself and checks
            # policy_kwargs for exact equality, including its injected defaults.
            parameters.pop("policy_kwargs", None)
            model = algorithm_class.load(
                str(request.resume_from),
                env=request.env,
                **parameters,
            )
            if (
                expected_num_timesteps is not None
                and int(model.num_timesteps) != expected_num_timesteps
            ):
                raise RuntimeError(
                    "checkpoint timestep count does not match its manifest"
                )
            load_replay_buffer = getattr(model, "load_replay_buffer", None)
            replay_buffer = _replay_buffer_path(request.resume_from)
            if load_replay_buffer is None:
                pass  # On-policy algorithms carry no experience across runs.
            elif replay_buffer.is_file():
                load_replay_buffer(str(replay_buffer))
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
            reset_num_timesteps = False
        if request.log_dir is not None:
            request.log_dir.mkdir(parents=True, exist_ok=True)
            configure_logger = _load_logger_configure()
            model.set_logger(
                configure_logger(str(request.log_dir), ["stdout", "csv", "json"])
            )
        callback = (
            None
            if request.best_model_path is None
            else _build_best_reward_callback(request.best_model_path)
        )
        if request.solve_evaluator is None:
            outcome = _learn_once(
                model,
                request,
                callback,
                reset_num_timesteps=reset_num_timesteps,
            )
        else:
            outcome = _learn_until_solved(
                model,
                request,
                callback,
                reset_num_timesteps=reset_num_timesteps,
            )
        return SB3Policy(model, training_outcome=outcome)


class SB3SACTrainer(SB3Trainer):
    """Train SAC, the off-policy baseline for these milestones."""

    def _algorithm_class(self) -> type[Any]:
        return _load_sac_class()


class SB3PPOTrainer(SB3Trainer):
    """Train PPO, the on-policy comparison for the same milestones."""

    def _algorithm_class(self) -> type[Any]:
        return _load_ppo_class()


def _load_sb3_policy(path: str | Path, algorithm_class: type[Any]) -> SB3Policy:
    expected_num_timesteps = _validate_checkpoint_pair(path)
    model = algorithm_class.load(str(path))
    if (
        expected_num_timesteps is not None
        and int(model.num_timesteps) != expected_num_timesteps
    ):
        raise RuntimeError("checkpoint timestep count does not match its manifest")
    return SB3Policy(model)


def load_sb3_sac_policy(path: str | Path) -> SB3Policy:
    """Load a trusted serialized SAC model as a common Policy.

    SB3 checkpoints can contain cloudpickled Python objects. Callers must not
    pass files from an untrusted source because loading can execute code.
    """

    return _load_sb3_policy(path, _load_sac_class())


def load_sb3_ppo_policy(path: str | Path) -> SB3Policy:
    """Load a trusted serialized PPO model as a common Policy.

    The same deserialization warning as `load_sb3_sac_policy` applies.
    """

    return _load_sb3_policy(path, _load_ppo_class())
