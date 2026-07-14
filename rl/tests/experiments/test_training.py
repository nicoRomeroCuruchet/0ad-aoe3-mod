import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rl.agents.base import AgentSpec, TrainRequest
from rl.experiments.config import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from rl.experiments.training import DecisionObserverEnv, train_policy


class ConstantPolicy:
    def act(self, observation, *, deterministic):
        del observation, deterministic
        return np.zeros(1, dtype=np.float32)


class RecordingTrainer:
    def __init__(self):
        self.requests = []

    def fit(self, request):
        self.requests.append(request)
        return ConstantPolicy()


class RecordingEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, events):
        super().__init__()
        self.events = events

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return np.full(5, 0.25, dtype=np.float32), {}

    def step(self, action):
        self.events.append(("environment_step", np.asarray(action).copy()))
        return np.full(5, 0.5, dtype=np.float32), 0.0, False, False, {}


def experiment_config():
    return ExperimentConfig(
        environment=EnvironmentConfig(
            name="zero_ad_gather",
            parameters={"scenario": "unused.json"},
        ),
        agent=AgentSpec(name="student_algorithm", parameters={"learning_rate": 0.1}),
        training=TrainingConfig(total_steps=321, seed=7),
        evaluation=EvaluationConfig(episodes=2, deterministic=True, seed=100),
    )


def test_train_policy_only_depends_on_the_common_trainer_contract():
    config = experiment_config()
    env = object()
    trainer = RecordingTrainer()

    policy = train_policy(
        config,
        env,
        trainer_builder=lambda agent: trainer,
    )

    assert isinstance(policy, ConstantPolicy)
    assert trainer.requests == [
        TrainRequest(
            env=env,
            agent=config.agent,
            total_steps=321,
            seed=7,
        )
    ]


def test_train_policy_wraps_the_environment_when_observing_decisions():
    config = experiment_config()
    env = RecordingEnv([])
    trainer = RecordingTrainer()

    train_policy(
        config,
        env,
        trainer_builder=lambda agent: trainer,
        decision_observer=lambda record: None,
    )

    assert len(trainer.requests) == 1
    assert isinstance(trainer.requests[0].env, DecisionObserverEnv)
    assert trainer.requests[0].env.env is env


def test_decision_observer_env_reports_each_training_decision_before_step():
    events = []
    records = []
    env = RecordingEnv(events)

    def observe(record):
        records.append(record)
        events.append(("decision", record))

    observed_env = DecisionObserverEnv(env, observe)
    first_action = np.array([0.1, -0.2], dtype=np.float32)
    second_action = np.array([0.3, -0.4], dtype=np.float32)

    observed_env.reset(seed=7)
    observed_env.step(first_action)
    observed_env.step(second_action)

    assert observed_env.observation_space is env.observation_space
    assert observed_env.action_space is env.action_space
    assert [event[0] for event in events] == [
        "decision",
        "environment_step",
        "decision",
        "environment_step",
    ]
    assert [(record.episode, record.step) for record in records] == [(0, 0), (0, 1)]
    np.testing.assert_array_equal(records[0].observation, np.full(5, 0.25))
    np.testing.assert_array_equal(records[0].action, first_action)
    np.testing.assert_array_equal(records[1].observation, np.full(5, 0.5))
    np.testing.assert_array_equal(records[1].action, second_action)


def test_decision_observer_env_resets_episode_and_requires_an_observation():
    records = []
    observed_env = DecisionObserverEnv(RecordingEnv([]), records.append)
    action = np.zeros(2, dtype=np.float32)

    with pytest.raises(RuntimeError, match="reset"):
        observed_env.step(action)

    observed_env.reset()
    observed_env.step(action)
    observed_env.reset()
    observed_env.step(action)

    assert [(record.episode, record.step) for record in records] == [(0, 0), (1, 0)]
