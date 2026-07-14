from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rl.experiments.evaluation import (
    EvaluationReport,
    StepRecord,
    evaluate,
    run_episode,
)


class TwoStepEnv:
    def __init__(self):
        self.actions = []
        self.reset_seeds = []
        self._step = 0

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        self._step = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self._step += 1
        terminated = self._step == 2
        return (
            np.array([float(self._step)], dtype=np.float32),
            float(self._step),
            terminated,
            False,
            {"distance": float(2 - self._step)},
        )


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def act(self, observation, *, deterministic):
        self.calls.append((np.asarray(observation).copy(), deterministic))
        return np.array([observation[0]], dtype=np.float32)


class TruncatingEnv:
    def reset(self, *, seed=None):
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        return (
            np.array([1.0], dtype=np.float32),
            -1.0,
            False,
            True,
            {"reason": "limit"},
        )


def test_run_episode_uses_policy_and_records_terminal_transition():
    env = TwoStepEnv()
    policy = RecordingPolicy()

    result = run_episode(
        env,
        policy,
        episode=4,
        deterministic=True,
        seed=41,
    )

    assert result.episode == 4
    assert result.total_reward == 3.0
    assert result.steps == 2
    assert result.terminated is True
    assert result.truncated is False
    assert result.final_info == {"distance": 0.0}
    assert env.reset_seeds == [41]
    assert [deterministic for _, deterministic in policy.calls] == [True, True]

    with pytest.raises(FrozenInstanceError):
        result.steps = 3


def test_run_episode_preserves_truncation_status():
    result = run_episode(
        TruncatingEnv(),
        RecordingPolicy(),
        episode=0,
        deterministic=False,
    )

    assert result.total_reward == -1.0
    assert result.steps == 1
    assert result.terminated is False
    assert result.truncated is True
    assert result.final_info == {"reason": "limit"}


def test_run_episode_emits_immutable_algorithm_independent_step_records():
    records = []

    run_episode(
        TwoStepEnv(),
        RecordingPolicy(),
        episode=3,
        deterministic=True,
        observer=records.append,
    )

    assert [record.step for record in records] == [0, 1]
    assert all(isinstance(record, StepRecord) for record in records)
    assert records[-1].episode == 3
    assert records[-1].reward == 2.0
    assert records[-1].terminated is True
    assert records[-1].info == {"distance": 0.0}
    assert records[-1].action.flags.writeable is False

    with pytest.raises(FrozenInstanceError):
        records[-1].step = 10


def test_evaluate_returns_immutable_results_and_aggregates():
    env = TwoStepEnv()
    policy = RecordingPolicy()

    report = evaluate(
        env,
        policy,
        episodes=3,
        deterministic=False,
        seed=10,
    )

    assert isinstance(report.episodes, tuple)
    assert [result.episode for result in report.episodes] == [0, 1, 2]
    assert report.episode_count == 3
    assert report.mean_total_reward == 3.0
    assert report.mean_steps == 2.0
    assert report.success_rate == 1.0
    assert env.reset_seeds == [10, 11, 12]

    with pytest.raises(FrozenInstanceError):
        report.episodes = ()


def test_evaluate_forwards_one_observer_across_episodes():
    records = []

    evaluate(
        TwoStepEnv(),
        RecordingPolicy(),
        episodes=2,
        deterministic=False,
        observer=records.append,
    )

    assert [(record.episode, record.step) for record in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_evaluation_report_freezes_directly_supplied_results():
    episode = run_episode(
        TruncatingEnv(),
        RecordingPolicy(),
        episode=0,
        deterministic=True,
    )
    supplied_results = [episode]

    report = EvaluationReport(episodes=supplied_results)
    supplied_results.clear()

    assert report.episodes == (episode,)


def test_evaluation_report_requires_results():
    with pytest.raises(ValueError, match="at least one episode"):
        EvaluationReport(episodes=())


@pytest.mark.parametrize("episodes", [0, -1])
def test_evaluate_requires_at_least_one_episode(episodes):
    with pytest.raises(ValueError, match="episodes must be positive"):
        evaluate(
            TwoStepEnv(),
            RecordingPolicy(),
            episodes=episodes,
            deterministic=True,
        )
