import numpy as np

from rl.agents.base import AgentSpec, TrainRequest
from rl.experiments.config import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from rl.experiments.training import train_policy


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
