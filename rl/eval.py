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
    EvaluationReport,
    StepObserver,
    StepRecord,
    evaluate,
)
from rl.gather.agent_view import (
    AgentView,
    AgentViewUnavailable,
    make_agent_view_observer,
    open_agent_view,
)
from rl.gather.assignment_actions import (
    JOINT_ASSIGNMENT_CLICK_MODE,
    joint_assignment_from_index,
)
from rl.gather.core import denormalize_action
from rl.gather.rollout_recording import AgentViewRolloutRecorder, mode_recording_dir


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

        action_mode = getattr(env, "action_mode", None)
        map_size_m = getattr(env, "map_size_m", None)
        if action_mode == "assignment_click":
            assignments = np.asarray(record.action).reshape(-1)
            labels = [
                "NO_CLICK" if int(choice) == 0 else f"TREE_{int(choice) - 1}"
                for choice in assignments
            ]
            target = f"assignments={labels}"
        elif action_mode == JOINT_ASSIGNMENT_CLICK_MODE:
            assignments = joint_assignment_from_index(
                record.action,
                villager_count=int(getattr(env, "villager_count")),
                resource_count=int(getattr(env, "resource_count")),
            )
            labels = [
                "NO_CLICK" if int(choice) == 0 else f"TREE_{int(choice) - 1}"
                for choice in assignments
            ]
            target = f"joint_assignment={labels}"
        elif map_size_m is None:
            target = f"action={record.action.tolist()}"
        else:
            x, z = denormalize_action(record.action, map_size_m)
            target = f"target=({x:.0f},{z:.0f})"
            action_values = np.asarray(record.action).reshape(-1)
            if action_values.size >= 3:
                target += f" click_signal={float(action_values[2]):+.2f}"
        distance = record.info.get("distance")
        distance_text = "" if distance is None else f" dist={float(distance):.1f}"
        command = record.info.get("command")
        command_text = "" if command is None else f" command={command}"
        stock = record.info.get("resource_stock")
        stock_delta = record.info.get("resource_stock_delta")
        stock_text = (
            ""
            if stock is None
            else f" stock={float(stock):.1f} dstock={float(stock_delta or 0.0):+.1f}"
        )
        carried = record.info.get("carried_resource")
        carried_delta = record.info.get("carried_resource_delta")
        carried_text = (
            ""
            if carried is None
            else (
                f" carried={float(carried):.1f}"
                f" dcarried={float(carried_delta or 0.0):+.1f}"
            )
        )
        reward_parts = []
        for key, label in (
            ("distance_shaping_reward", "dist"),
            ("gather_ready_reward", "ready"),
            ("carried_resource_delta_reward", "carry"),
            ("gather_cycle_no_click_reward", "wait"),
            ("carrying_no_click_reward", "hold"),
        ):
            value = record.info.get(key)
            if value:
                reward_parts.append(f"{label}={float(value):+.2f}")
        click_penalty = record.info.get("click_gather_cycle_penalty")
        if click_penalty:
            reward_parts.append(f"interrupt={-float(click_penalty):+.2f}")
        reward_parts_text = (
            "" if not reward_parts else f" parts=[{' '.join(reward_parts)}]"
        )
        observation_text = _format_observation(env, record.observation)
        print(
            f"    step {record.step:2d}: {observation_text} {target}{distance_text}"
            f"{command_text}{stock_text}{carried_text}{reward_parts_text}"
            f" reward={record.reward:+.2f}"
        )

    return observe


def _combine_decision_observers(
    *observers: DecisionObserver | None,
) -> DecisionObserver | None:
    active = tuple(observer for observer in observers if observer is not None)
    if not active:
        return None

    def observe(record):
        for observer in active:
            observer(record)

    return observe


def _print_report(report: EvaluationReport, mode_label: str) -> None:
    for result in report.episodes:
        distance = result.final_info.get("distance")
        distance_text = "" if distance is None else f" dist_final={float(distance):.1f}"
        stock_delta = result.final_info.get("episode_resource_stock_delta")
        stock_text = (
            ""
            if stock_delta is None
            else f" stock_delta={float(stock_delta):.1f}"
        )
        print(
            f"ep {result.episode:2d}: reward={result.total_reward:.1f}"
            f"{distance_text}{stock_text} alcanzado={result.terminated}"
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
        help="open a Polites-centered window rendered by the 0 A.D. engine",
    )
    parser.add_argument(
        "--delay",
        type=_non_negative_float,
        default=0.0,
        help=("seconds between decisions; with --agent-view, pause before the action"),
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="ask 0 A.D. to save each evaluated episode as a replay",
    )
    parser.add_argument(
        "--record-agent-view",
        type=Path,
        help=(
            "write engine-rendered rollout frames and an HTML player to this directory"
        ),
    )
    parser.add_argument(
        "--record-agent-view-video",
        type=Path,
        help="encode recorded agent-view frames to this MP4 file with ffmpeg",
    )
    parser.add_argument(
        "--record-sim-turns",
        action="store_true",
        help=(
            "capture one frame per simulation turn instead of per decision; "
            "needed for smooth rollout video"
        ),
    )
    parser.add_argument(
        "--allow-schematic-recording",
        action="store_true",
        help="fall back to schematic frames if the engine renderer cannot capture",
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
    if args.record_sim_turns and (
        args.record_agent_view is None and args.record_agent_view_video is None
    ):
        parser.error(
            "--record-sim-turns requires --record-agent-view "
            "or --record-agent-view-video"
        )
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
        view_observer = (
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
            recording_dir = args.record_agent_view
            if recording_dir is None and args.record_agent_view_video is not None:
                recording_dir = args.record_agent_view_video.parent / (
                    f"{args.record_agent_view_video.stem}-frames"
                )
            recorder = (
                None
                if recording_dir is None
                else AgentViewRolloutRecorder(
                    mode_recording_dir(
                        recording_dir,
                        deterministic=deterministic,
                    ),
                    env,
                    fallback_to_schematic=args.allow_schematic_recording,
                )
            )
            if args.record_sim_turns and recorder is not None:
                env.sim_frame_observer = recorder.observe_sim_turn
            try:
                report = evaluate(
                    env,
                    policy,
                    episodes=config.evaluation.episodes,
                    deterministic=deterministic,
                    seed=config.evaluation.seed,
                    decision_observer=_combine_decision_observers(
                        view_observer,
                        None if recorder is None else recorder.observe,
                    ),
                    observer=observer,
                )
            finally:
                if args.record_sim_turns and recorder is not None:
                    env.sim_frame_observer = None
            _print_report(report, label)
            if recorder is not None:
                print(f"rollout: {recorder.write_index()}")
                if args.record_agent_view_video is not None:
                    video_path = (
                        args.record_agent_view_video
                        if len(modes) == 1
                        else args.record_agent_view_video.with_name(
                            f"{args.record_agent_view_video.stem}-"
                            f"{'deterministic' if deterministic else 'stochastic'}"
                            f"{args.record_agent_view_video.suffix}"
                        )
                    )
                    print(f"video: {recorder.write_video(video_path)}")
    finally:
        try:
            if agent_view is not None:
                agent_view.close()
        finally:
            _close_environment(env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
