import numpy as np

from rl.agents.base import AgentSpec, TrainRequest
from rl.experiments.config import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from rl.experiments.training import TrainingResult, run_training, train_policy


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


class OneStepEnv:
    def __init__(self):
        self.reset_seeds = []

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.ones(1, dtype=np.float32), 2.5, True, False, {"distance": 0.0}


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


def test_run_training_constructs_env_trains_and_evaluates_with_shared_contracts():
    config = experiment_config()
    env = OneStepEnv()
    trainer = RecordingTrainer()
    built_from = []

    result = run_training(
        config,
        environment_builder=lambda selected: built_from.append(selected) or env,
        trainer_builder=lambda agent: trainer,
    )

    assert isinstance(result, TrainingResult)
    assert isinstance(result.policy, ConstantPolicy)
    assert result.environment is env
    assert result.evaluation.episode_count == 2
    assert result.evaluation.mean_total_reward == 2.5
    assert env.reset_seeds == [100, 101]
    assert built_from == [config.environment]


def test_run_training_uses_an_injected_environment_without_constructing_one():
    config = experiment_config()
    env = OneStepEnv()
    trainer = RecordingTrainer()

    result = run_training(
        config,
        env=env,
        environment_builder=lambda _config: (_ for _ in ()).throw(AssertionError),
        trainer_builder=lambda agent: trainer,
    )

    assert result.environment is env
