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


def test_gather_oracle_can_emit_a_click_signal():
    observation = np.array(
        [-0.5, 0.25, 0.75, -0.25, 0.8],
        dtype=np.float32,
    )
    policy = GatherOraclePolicy(action_size=3)

    action = policy.act(observation, deterministic=True)

    np.testing.assert_array_equal(
        action,
        np.array([0.75, -0.25, 1.0], dtype=np.float32),
    )


def test_gather_oracle_validates_observation_shape():
    policy = GatherOraclePolicy()

    with pytest.raises(ValueError, match="at least four values"):
        policy.act(np.zeros(3, dtype=np.float32), deterministic=False)


def test_gather_oracle_rejects_unsupported_action_sizes():
    with pytest.raises(ValueError, match="action_size"):
        GatherOraclePolicy(action_size=4)


def test_team_oracle_sends_empty_villagers_to_their_nearest_free_tree():
    from rl.agents.baselines import TeamGatherOraclePolicy
    from rl.gather.observation import (
        TeamObservationScales,
        TeamSnapshot,
        build_team_observation,
    )

    snapshot = TeamSnapshot(
        villager_xz=((100.0, 100.0), (200.0, 100.0)),
        resource_xz=((300.0, 100.0), (400.0, 100.0)),
        resource_remaining=(200.0, 200.0),
        carried=(0.0, 0.0),
        target_index=(0, 1),
        gather_cycle_active=(False, False),
        dropsite_xz=(50.0, 150.0),
        stock=0.0,
    )
    observation = build_team_observation(
        snapshot,
        TeamObservationScales(
            map_size_m=512.0,
            carried_resource_scale=20.0,
            stock_scale=1000.0,
            resource_amount_scale=200.0,
        ),
    )

    action = TeamGatherOraclePolicy(2, 2).act(observation, deterministic=True)

    assert action.shape == (6,)
    assert action[0] == pytest.approx(2.0 * 300.0 / 512.0 - 1.0, abs=1e-3)
    assert action[1] == pytest.approx(2.0 * 100.0 / 512.0 - 1.0, abs=1e-3)
    assert action[2] == pytest.approx(1.0)
    assert action[3] == pytest.approx(2.0 * 400.0 / 512.0 - 1.0, abs=1e-3)


def test_team_oracle_sends_loaded_villagers_to_the_dropsite():
    from rl.agents.baselines import TeamGatherOraclePolicy
    from rl.gather.observation import (
        TeamObservationScales,
        TeamSnapshot,
        build_team_observation,
    )

    snapshot = TeamSnapshot(
        villager_xz=((100.0, 100.0), (200.0, 100.0)),
        resource_xz=((300.0, 100.0), (400.0, 100.0)),
        resource_remaining=(200.0, 200.0),
        carried=(20.0, 0.0),
        target_index=(0, 1),
        gather_cycle_active=(False, False),
        dropsite_xz=(50.0, 150.0),
        stock=0.0,
    )
    observation = build_team_observation(
        snapshot,
        TeamObservationScales(
            map_size_m=512.0,
            carried_resource_scale=20.0,
            stock_scale=1000.0,
            resource_amount_scale=200.0,
        ),
    )

    action = TeamGatherOraclePolicy(2, 2).act(observation, deterministic=True)

    assert action[0] == pytest.approx(2.0 * 50.0 / 512.0 - 1.0, abs=1e-3)
    assert action[1] == pytest.approx(2.0 * 150.0 / 512.0 - 1.0, abs=1e-3)
