from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rl.agents.base import (
    AgentSpec,
    EpisodeResult,
    Policy,
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
    )

    assert request.total_steps == 1_000
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
