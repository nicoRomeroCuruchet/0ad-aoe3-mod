from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from rl.agents.base import (
    AgentSpec,
    EpisodeResult,
    Policy,
    TrainingOutcome,
    Trainer,
    TrainRequest,
)


class ConstantPolicy:
    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> np.ndarray:
        del observation, deterministic
        return np.zeros(2, dtype=np.float32)


class NoOpTrainer:
    def fit(self, request: TrainRequest) -> Policy:
        del request
        return ConstantPolicy()


def test_policy_and_trainer_are_runtime_checkable_protocols():
    assert isinstance(ConstantPolicy(), Policy)
    assert isinstance(NoOpTrainer(), Trainer)


def test_agent_spec_is_frozen_and_defensively_freezes_parameters():
    source = {
        "hidden_sizes": [256, 256],
        "optimizer": {"learning_rate": 3e-4},
    }

    spec = AgentSpec(name="custom_sac", parameters=source)
    source["hidden_sizes"].append(128)
    source["optimizer"]["learning_rate"] = 1.0

    assert spec.parameters["hidden_sizes"] == (256, 256)
    assert spec.parameters["optimizer"]["learning_rate"] == 3e-4
    with pytest.raises(TypeError):
        spec.parameters["new_parameter"] = True
    with pytest.raises(FrozenInstanceError):
        spec.name = "td3"


@pytest.mark.parametrize("name", ["", "   "])
def test_agent_spec_rejects_empty_names(name):
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        AgentSpec(name=name)


def test_train_request_is_frozen_and_validates_its_budget():
    request = TrainRequest(
        env=object(),
        agent=AgentSpec(name="custom_sac"),
        total_steps=1_000,
        seed=7,
        log_dir=Path("rl/runs/example/training"),
        log_interval=2,
        best_model_path=Path("rl/runs/example/best_model"),
        checkpoint_path=Path("rl/runs/example/model"),
        resume_from=Path("rl/runs/previous/best_model"),
        solved_window_episodes=20,
        solved_success_rate=0.8,
        solved_min_steps=1_000,
        solved_check_interval_steps=250,
        solve_evaluator=lambda policy: 1.0,
    )

    assert request.total_steps == 1_000
    assert request.log_interval == 2
    assert request.best_model_path == Path("rl/runs/example/best_model")
    assert request.checkpoint_path == Path("rl/runs/example/model")
    assert request.resume_from == Path("rl/runs/previous/best_model")
    assert request.solved_window_episodes == 20
    assert request.solved_success_rate == 0.8
    assert request.solved_min_steps == 1_000
    assert request.solved_check_interval_steps == 250
    assert callable(request.solve_evaluator)
    with pytest.raises(FrozenInstanceError):
        request.seed = 8

    with pytest.raises(ValueError, match="total_steps must be a positive integer"):
        TrainRequest(
            env=object(),
            agent=AgentSpec(name="custom_sac"),
            total_steps=0,
            seed=7,
        )
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        TrainRequest(
            env=object(),
            agent=AgentSpec(name="custom_sac"),
            total_steps=10,
            seed=-1,
        )
    with pytest.raises(ValueError, match="log_interval must be a positive integer"):
        TrainRequest(
            env=object(),
            agent=AgentSpec(name="custom_sac"),
            total_steps=10,
            seed=7,
            log_interval=0,
        )
    with pytest.raises(ValueError, match="log_dir"):
        TrainRequest(
            env=object(),
            agent=AgentSpec(name="custom_sac"),
            total_steps=10,
            seed=7,
            log_dir="rl/runs/example/training",
        )
    with pytest.raises(ValueError, match="best_model_path"):
        TrainRequest(
            env=object(),
            agent=AgentSpec(name="custom_sac"),
            total_steps=10,
            seed=7,
            best_model_path="rl/runs/example/best_model",
        )
    with pytest.raises(ValueError, match="resume_from"):
        TrainRequest(
            env=object(),
            agent=AgentSpec(name="custom_sac"),
            total_steps=10,
            seed=7,
            resume_from="rl/runs/previous/best_model",
        )
    with pytest.raises(ValueError, match="checkpoint_path"):
        TrainRequest(
            env=object(),
            agent=AgentSpec(name="custom_sac"),
            total_steps=10,
            seed=7,
            checkpoint_path="rl/runs/example/model",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"solved_window_episodes": 0, "solved_success_rate": 0.8}, "window"),
        ({"solved_window_episodes": True, "solved_success_rate": 0.8}, "window"),
        ({"solved_window_episodes": 10}, "must be configured together"),
        ({"solved_success_rate": 0.8}, "must be configured together"),
        ({"solved_window_episodes": 10, "solved_success_rate": 0.0}, "success rate"),
        ({"solved_window_episodes": 10, "solved_success_rate": 1.01}, "success rate"),
        ({"solved_window_episodes": 10, "solved_success_rate": float("nan")}, "success rate"),
        ({"solved_window_episodes": 10, "solved_success_rate": True}, "success rate"),
        (
            {
                "solved_window_episodes": 10,
                "solved_success_rate": 0.8,
                "solved_min_steps": -1,
            },
            "minimum steps",
        ),
        ({"solved_min_steps": 1}, "requires solved stopping"),
        (
            {"solved_window_episodes": 10, "solved_success_rate": 0.8},
            "check interval",
        ),
        (
            {
                "solved_window_episodes": 10,
                "solved_success_rate": 0.8,
                "solved_check_interval_steps": True,
            },
            "check interval",
        ),
        (
            {
                "solved_window_episodes": 10,
                "solved_success_rate": 0.8,
                "solved_check_interval_steps": 5,
            },
            "evaluator",
        ),
    ],
)
def test_train_request_rejects_invalid_solved_stopping(overrides, message):
    values = {
        "env": object(),
        "agent": AgentSpec(name="custom_sac"),
        "total_steps": 10,
        "seed": 7,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        TrainRequest(**values)


@pytest.mark.parametrize("stop_reason", ["solved", "safety_cap", "steps_complete"])
def test_training_outcome_is_frozen_and_validated(stop_reason):
    outcome = TrainingOutcome(
        stop_reason=stop_reason,
        start_num_timesteps=10,
        end_num_timesteps=30,
        steps_this_run=20,
        solve_checks=2,
        last_success_rate=0.8,
    )

    assert outcome.stop_reason == stop_reason
    with pytest.raises(FrozenInstanceError):
        outcome.steps_this_run = 21


def test_episode_result_is_frozen_and_defensively_freezes_final_info():
    source_info = {"distance": 3.5, "labels": ["success"]}
    result = EpisodeResult(
        episode=2,
        total_reward=12.0,
        steps=4,
        terminated=True,
        truncated=False,
        final_info=source_info,
    )
    source_info["distance"] = 100.0
    source_info["labels"].append("changed")

    assert result.final_info["distance"] == 3.5
    assert result.final_info["labels"] == ("success",)
    with pytest.raises(TypeError):
        result.final_info["distance"] = 1.0
    with pytest.raises(FrozenInstanceError):
        result.steps = 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"episode": -1}, "episode must be a non-negative integer"),
        ({"steps": -1}, "steps must be a non-negative integer"),
        ({"terminated": 1}, "terminated must be a boolean"),
        ({"truncated": 0}, "truncated must be a boolean"),
    ],
)
def test_episode_result_rejects_invalid_metadata(overrides, message):
    values = {
        "episode": 0,
        "total_reward": 0.0,
        "steps": 0,
        "terminated": False,
        "truncated": False,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        EpisodeResult(**values)
