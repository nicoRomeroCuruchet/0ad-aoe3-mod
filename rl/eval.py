"""Evaluate any registered policy against a configured environment."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.agents.registry import (
    AgentCapabilityError,
    build_policy,
    ensure_can_build,
    load_policy,
)
from rl.experiments.config import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    load_experiment_config,
)
from rl.experiments.environments import build_environment
from rl.experiments.evaluation import (
    DecisionObserver,
    DecisionRecord,
    EvaluationReport,
    StepObserver,
    StepRecord,
    evaluate,
)
from rl.gather.agent_view import (
    AgentView,
    AgentViewUnavailable,
    open_agent_view,
)
from rl.gather.core import denormalize_action


DEFAULT_EXPERIMENT = Path("rl/configs/m0_oracle.toml")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def apply_overrides(
    config: ExperimentConfig,
    *,
    uri: str | None,
    episodes: int | None,
    save_replay: bool,
) -> ExperimentConfig:
    """Return a new config with local evaluation overrides applied."""

    environment_parameters = dict(config.environment.parameters)
    if uri is not None:
        environment_parameters["uri"] = uri
    if save_replay:
        environment_parameters["save_replay"] = True

    return ExperimentConfig(
        environment=EnvironmentConfig(
            name=config.environment.name,
            parameters=environment_parameters,
        ),
        agent=config.agent,
        training=config.training,
        evaluation=EvaluationConfig(
            episodes=config.evaluation.episodes if episodes is None else episodes,
            deterministic=config.evaluation.deterministic,
            seed=config.evaluation.seed,
        ),
    )


def _format_observation(env: object, observation: object) -> str:
    values = np.asarray(observation).reshape(-1)
    labels = tuple(getattr(env, "observation_labels", ()))
    if len(labels) == len(values):
        contents = ", ".join(
            f"{label}={float(value):.9g}"
            for label, value in zip(labels, values, strict=True)
        )
    else:
        contents = ", ".join(f"{float(value):.9g}" for value in values)
    return f"observation=[{contents}]"


def make_step_observer(
    env: object,
    *,
    verbose: bool,
    delay: float,
) -> StepObserver | None:
    """Build an optional display observer without coupling it to an agent."""

    if not verbose and delay == 0:
        return None

    def observe(record: StepRecord) -> None:
        if delay:
            time.sleep(delay)
        if not verbose:
            return

        map_size_m = getattr(env, "map_size_m", None)
        if map_size_m is None:
            target = f"action={record.action.tolist()}"
        else:
            x, z = denormalize_action(record.action, map_size_m)
            target = f"target=({x:.0f},{z:.0f})"
        distance = record.info.get("distance")
        distance_text = "" if distance is None else f" dist={float(distance):.1f}"
        observation_text = _format_observation(env, record.observation)
        print(
            f"    step {record.step:2d}: {observation_text} {target}{distance_text} "
            f"reward={record.reward:+.2f}"
        )

    return observe


def make_agent_view_observer(
    agent_view: AgentView,
    *,
    delay: float,
) -> DecisionObserver:
    """Update the local view before advancing the environment."""

    def observe(record: DecisionRecord) -> None:
        agent_view.update(record)
        if delay:
            agent_view.pause(delay)

    return observe


def _print_report(report: EvaluationReport, mode_label: str) -> None:
    for result in report.episodes:
        distance = result.final_info.get("distance")
        distance_text = "" if distance is None else f" dist_final={float(distance):.1f}"
        print(
            f"ep {result.episode:2d}: reward={result.total_reward:.1f}"
            f"{distance_text} alcanzado={result.terminated}"
        )
    print(
        f"-- {report.episode_count} episodios ({mode_label}): "
        f"reward medio={report.mean_total_reward:.1f} | "
        f"pasos medios={report.mean_steps:.1f} | "
        f"éxito={report.success_rate:.0%} --"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help="tracked TOML experiment definition",
    )
    parser.add_argument("--model", type=Path, help="checkpoint for a learned agent")
    parser.add_argument(
        "--trust-model",
        action="store_true",
        help=(
            "allow Python-based checkpoint deserialization; use only for a model "
            "you created or otherwise trust"
        ),
    )
    parser.add_argument("--uri", help="override the configured 0 A.D. RL server URI")
    parser.add_argument("--episodes", type=_positive_integer)
    parser.add_argument(
        "--mode",
        choices=("configured", "deterministic", "stochastic", "both"),
        default="configured",
    )
    parser.add_argument("--verbose", action="store_true", help="print every step")
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="open a Polites-centered window showing the exact policy observation",
    )
    parser.add_argument(
        "--delay",
        type=_non_negative_float,
        default=0.0,
        help=(
            "seconds between decisions; with --agent-view, pause before the action"
        ),
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="ask 0 A.D. to save each evaluated episode as a replay",
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model is not None and not args.trust_model:
        parser.error(
            "loading an SB3 checkpoint can execute code; pass --trust-model "
            "only if you trust its source"
        )
    loaded_config = load_experiment_config(args.experiment)
    config = apply_overrides(
        loaded_config,
        uri=args.uri,
        episodes=args.episodes,
        save_replay=args.replay,
    )
    try:
        if args.model is None:
            ensure_can_build(config.agent)
            policy = None
        else:
            policy = load_policy(config.agent, args.model)
    except AgentCapabilityError as error:
        parser.error(f"{error}; pass --model for a learned agent")

    env = build_environment(
        config.environment,
        allow_remote=args.allow_remote_server,
    )

    agent_view: AgentView | None = None
    try:
        if policy is None:
            policy = build_policy(config.agent, env, seed=config.evaluation.seed)

        if args.agent_view:
            try:
                agent_view = open_agent_view(env)
            except AgentViewUnavailable as error:
                parser.error(str(error))

        observer = make_step_observer(
            env,
            verbose=args.verbose,
            delay=0.0 if agent_view is not None else args.delay,
        )
        decision_observer = (
            None
            if agent_view is None
            else make_agent_view_observer(agent_view, delay=args.delay)
        )
        modes = []
        if args.mode == "configured":
            configured_label = (
                "determinista" if config.evaluation.deterministic else "estocástica"
            )
            modes.append((config.evaluation.deterministic, configured_label))
        if args.mode in ("deterministic", "both"):
            modes.append((True, "determinista"))
        if args.mode in ("stochastic", "both"):
            modes.append((False, "estocástica"))

        for deterministic, label in modes:
            report = evaluate(
                env,
                policy,
                episodes=config.evaluation.episodes,
                deterministic=deterministic,
                seed=config.evaluation.seed,
                decision_observer=decision_observer,
                observer=observer,
            )
            _print_report(report, label)
    finally:
        try:
            if agent_view is not None:
                agent_view.close()
        finally:
            _close_environment(env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
