"""Train one registered agent from a tracked experiment config."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Sequence

from rl.agents.registry import ensure_can_save, save_policy
from rl.experiments.artifacts import (
    create_run_artifacts,
    record_evaluation,
    record_run_context,
)
from rl.experiments.config import (
    EnvironmentConfig,
    ExperimentConfig,
    TrainingConfig,
    load_experiment_config,
)
from rl.experiments.environments import build_environment
from rl.experiments.evaluation import evaluate
from rl.experiments.training import train_policy


DEFAULT_EXPERIMENT = Path("rl/configs/m0_sb3_sac.toml")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def apply_overrides(
    config: ExperimentConfig,
    *,
    uri: str | None,
    total_steps: int | None,
) -> ExperimentConfig:
    """Return a new config with optional machine/run-specific CLI values."""

    environment_parameters = dict(config.environment.parameters)
    if uri is not None:
        environment_parameters["uri"] = uri

    return ExperimentConfig(
        environment=EnvironmentConfig(
            name=config.environment.name,
            parameters=environment_parameters,
        ),
        agent=config.agent,
        training=TrainingConfig(
            total_steps=(
                config.training.total_steps if total_steps is None else total_steps
            ),
            seed=config.training.seed,
        ),
        evaluation=config.evaluation,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="tracked TOML experiment definition",
    )
    parser.add_argument(
        "--uri",
        help="override the configured 0 A.D. RL server URI",
    )
    parser.add_argument(
        "--timesteps",
        type=_positive_integer,
        help="override training.total_steps",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("rl/runs"),
        help="directory for ignored run artifacts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="optional checkpoint path override (without .zip for SB3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow --out to replace an existing checkpoint",
    )
    parser.add_argument(
        "--allow-remote-server",
        action="store_true",
        help=(
            "allow sending the scenario to a non-loopback RL server; "
            "use only when you trust it"
        ),
    )
    return parser


def _close_environment(env: object) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _checkpoint_candidates(path: Path) -> tuple[Path, ...]:
    candidates = {path, Path(f"{path}.zip")}
    if path.name:
        candidates.add(path.with_suffix(".zip"))
    return tuple(sorted(candidates, key=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    loaded_config = load_experiment_config(args.experiment)
    config = apply_overrides(
        loaded_config,
        uri=args.uri,
        total_steps=args.timesteps,
    )
    ensure_can_save(config.agent)
    if args.out is not None and not args.force:
        existing = next(
            (path for path in _checkpoint_candidates(args.out) if path.exists()),
            None,
        )
        if existing is not None:
            parser.error(
                f"checkpoint already exists at {existing}; pass --force to replace it"
            )

    artifacts = create_run_artifacts(args.run_root, args.experiment.stem)
    model_path = args.out if args.out is not None else artifacts.model_path
    env = build_environment(
        config.environment,
        allow_remote=args.allow_remote_server,
    )

    try:
        policy = train_policy(config, env)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_policy(config.agent, policy, model_path)
        metadata = {
            "agent": config.agent.name,
            "experiment_file": str(args.experiment),
            "model_path": str(model_path),
        }
        record_run_context(
            artifacts,
            config,
            metadata={
                **metadata,
                "python": platform.python_version(),
                "status": "checkpoint_saved",
            },
        )
        report = evaluate(
            env,
            policy,
            episodes=config.evaluation.episodes,
            deterministic=config.evaluation.deterministic,
            seed=config.evaluation.seed,
        )
        record_evaluation(artifacts, report)
        record_run_context(
            artifacts,
            config,
            metadata={
                **metadata,
                "python": platform.python_version(),
                "status": "complete",
            },
        )
    finally:
        _close_environment(env)

    print(f"run: {artifacts.run_dir}")
    print(f"modelo: {model_path}")
    print(
        "evaluación: "
        f"reward_medio={report.mean_total_reward:.1f} "
        f"éxito={report.success_rate:.0%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
