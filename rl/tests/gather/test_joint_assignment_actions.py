"""Contracts for the coordinated, collision-free M2 assignment action."""

from dataclasses import dataclass

import numpy as np
import pytest
from gymnasium import spaces

import rl.gather.assignment_actions as assignment_actions
from rl.gather.team_env import ZeroADTeamGatherEnv


@dataclass(frozen=True)
class FakeUnit:
    coordinates: tuple[float, float]
    entity_id: int

    def position(self):
        return self.coordinates

    def id(self):
        return self.entity_id


class FakeState:
    def __init__(self, villagers, resources, dropsite):
        self.villagers = villagers
        self.resources = resources
        self.dropsite = dropsite

    def units(self, *, owner, entity_type):
        return {
            (1, "polites"): self.villagers,
            (0, "tree"): self.resources,
            (1, "storehouse"): [self.dropsite],
        }[(owner, entity_type)]


class FakeActions:
    def __init__(self):
        self.calls = []

    def walk(self, units, x, z):
        self.calls.append(("walk", units[0].id(), x, z))
        return ("walk", units[0].id())

    def gather(self, units, resource):
        self.calls.append(("gather", units[0].id(), resource.id()))
        return ("gather", units[0].id())

    def returnresource(self, units, dropsite):
        self.calls.append(("returnresource", units[0].id(), dropsite.id()))
        return ("returnresource", units[0].id())


class FakeGame:
    def __init__(self, state):
        self.current_state = state
        self.step_calls = []

    def reset(self, _scenario_config, *, save_replay=False):
        del save_replay
        return self.current_state

    def step(self, commands=None):
        self.step_calls.append(commands)
        return self.current_state

    def step_many(self, commands=None, *, turns):
        self.step_calls.append((commands, turns))
        return self.current_state

    def evaluate(self, _expression):
        return {
            "stock": {"wood": 300.0},
            "carried": {str(11 + index): 0.0 for index in range(4)},
            "remaining": {str(21 + index): 200.0 for index in range(4)},
        }


def _joint_env():
    villagers = [
        FakeUnit((50.0 + 20.0 * index, 80.0), 11 + index)
        for index in range(4)
    ]
    resources = [
        FakeUnit((180.0 + 40.0 * index, 240.0), 21 + index)
        for index in range(4)
    ]
    state = FakeState(villagers, resources, FakeUnit((100.0, 100.0), 31))
    game = FakeGame(state)
    actions = FakeActions()
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=4,
        resource_count=4,
        action_mode="joint_assignment_click",
        map_size_m=512.0,
        horizon=3,
        sim_steps_per_action=1,
        gather_command_distance=12.0,
        game=game,
        actions=actions,
    )
    return env, actions


def test_joint_assignment_space_enumerates_all_209_partial_injective_4x4_actions():
    action_space = assignment_actions.joint_assignment_action_space(
        villager_count=4,
        resource_count=4,
    )

    assert isinstance(action_space, spaces.Discrete)
    assert action_space.n == 209
    assert action_space.start == 0


def test_every_joint_assignment_index_round_trips_and_never_reuses_a_tree():
    action_space = assignment_actions.joint_assignment_action_space(
        villager_count=4,
        resource_count=4,
    )
    decoded = []

    for index in range(action_space.n):
        assignment = assignment_actions.joint_assignment_from_index(
            index,
            villager_count=4,
            resource_count=4,
        )
        nonzero_categories = assignment[assignment != 0]

        assert assignment.shape == (4,)
        assert np.issubdtype(assignment.dtype, np.integer)
        assert np.all((0 <= assignment) & (assignment <= 4))
        assert len(nonzero_categories) == len(np.unique(nonzero_categories))
        assert assignment_actions.joint_assignment_to_index(
            assignment,
            villager_count=4,
            resource_count=4,
        ) == index
        decoded.append(tuple(int(category) for category in assignment))

    assert len(set(decoded)) == 209


def test_joint_assignment_encoder_rejects_duplicate_nonzero_tree_categories():
    with pytest.raises(ValueError):
        assignment_actions.joint_assignment_to_index(
            np.array([1, 1, 0, 0], dtype=np.int64),
            villager_count=4,
            resource_count=4,
        )


def test_joint_assignment_environment_exposes_a_discrete_209_action_space():
    env, _actions = _joint_env()

    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 209
    assert env.action_space.start == 0


@pytest.mark.parametrize("action_type", [int, np.int64], ids=["python-int", "np-int64"])
def test_joint_assignment_scalar_action_decodes_to_one_gather_per_assigned_villager(
    action_type,
):
    env, actions = _joint_env()
    assignment = np.array([1, 2, 3, 4], dtype=np.int64)
    action = action_type(
        assignment_actions.joint_assignment_to_index(
            assignment,
            villager_count=4,
            resource_count=4,
        )
    )
    assert env.action_space.contains(action)
    env.reset()

    env.step(action)

    assert actions.calls == [
        ("gather", 11, 21),
        ("gather", 12, 22),
        ("gather", 13, 23),
        ("gather", 14, 24),
    ]


def test_joint_assignment_execution_keeps_an_active_tree_reserved():
    env, actions = _joint_env()
    env.reset()
    # Villager 0 is still chopping TREE_2. The global categorical may contain
    # a stale candidate that names that tree for villager 1, but execution must
    # reject it just as the policy's observed-state action mask does.
    env._target_index = (2, None, None, None)
    env._cycle_active = (True, False, False, False)
    assignment = np.array([0, 3, 0, 0], dtype=np.int64)
    action = assignment_actions.joint_assignment_to_index(
        assignment,
        villager_count=4,
        resource_count=4,
    )

    env.step(action)

    assert actions.calls == []
