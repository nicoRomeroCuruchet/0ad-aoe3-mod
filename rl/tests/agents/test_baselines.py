import numpy as np
import pytest
from gymnasium import spaces

from rl.agents.baselines import GatherOraclePolicy, RandomPolicy


def test_random_policy_is_reproducible_for_a_seed():
    action_space = spaces.Box(
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 3.0], dtype=np.float32),
        dtype=np.float32,
    )
    first = RandomPolicy(action_space, seed=17)
    second = RandomPolicy(action_space, seed=17)
    observation = np.zeros(5, dtype=np.float32)

    first_actions = [first.act(observation, deterministic=False) for _ in range(3)]
    second_actions = [second.act(observation, deterministic=True) for _ in range(3)]

    for first_action, second_action in zip(first_actions, second_actions):
        np.testing.assert_array_equal(first_action, second_action)
        assert first_action.dtype == action_space.dtype
        assert action_space.contains(first_action)


def test_random_policy_rejects_unbounded_action_spaces():
    action_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(2,),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="finite bounds"):
        RandomPolicy(action_space, seed=0)


def test_gather_oracle_targets_resource_coordinates():
    observation = np.array(
        [-0.5, 0.25, 0.75, -0.25, 0.8],
        dtype=np.float32,
    )
    policy = GatherOraclePolicy()

    action = policy.act(observation, deterministic=True)

    np.testing.assert_array_equal(
        action,
        np.array([0.75, -0.25], dtype=np.float32),
    )
    action[0] = 0.0
    assert observation[2] == 0.75


def test_gather_oracle_validates_observation_shape():
    policy = GatherOraclePolicy()

    with pytest.raises(ValueError, match="at least four values"):
        policy.act(np.zeros(3, dtype=np.float32), deterministic=False)
