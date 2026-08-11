from dataclasses import dataclass

import numpy as np
import pytest

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
            (1, "polites"): list(self.villagers),
            (0, "tree"): list(self.resources),
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
    def __init__(self, state, snapshots):
        self.current_state = state
        self.snapshots = list(snapshots)
        self.evaluate_calls = []
        self.step_calls = []

    def reset(self, scenario_config, *, save_replay=False):
        return self.current_state

    def step(self, commands=None):
        self.step_calls.append(commands)
        return self.current_state

    def step_many(self, commands=None, *, turns):
        self.step_calls.append((commands, turns))
        return self.current_state

    def evaluate(self, expression):
        self.evaluate_calls.append(expression)
        if self.snapshots:
            return self.snapshots.pop(0)
        return {"stock": {"wood": 0.0}, "carried": {}, "remaining": {}}


def _backend(stock=300.0, count=12):
    villagers = [FakeUnit((100.0, 100.0), 11), FakeUnit((200.0, 100.0), 12)]
    resources = [FakeUnit((300.0, 100.0), 21), FakeUnit((400.0, 100.0), 22)]
    state = FakeState(villagers, resources, FakeUnit((50.0, 150.0), 31))
    snapshot = {
        "stock": {"wood": stock},
        "carried": {"11": 0.0, "12": 0.0},
        "remaining": {"21": 200.0, "22": 200.0},
    }
    return FakeGame(state, [snapshot] * count), FakeActions()


def _env(**overrides):
    game, actions = _backend()
    parameters = {
        "villager_count": 2,
        "resource_count": 2,
        "map_size_m": 512.0,
        "horizon": 3,
        "sim_steps_per_action": 2,
        "gather_command_distance": 12.0,
        "stock_success_threshold": 40.0,
        "game": game,
        "actions": actions,
    }
    parameters.update(overrides)
    return ZeroADTeamGatherEnv("scenario", **parameters), game, actions


def _target(x, z, map_size_m=512.0):
    return 2.0 * x / map_size_m - 1.0, 2.0 * z / map_size_m - 1.0


def test_spaces_follow_the_configured_counts():
    env, _game, _actions = _env()

    assert env.observation_space.shape == (2, 21)
    assert env.action_space.shape == (6,)


def test_reset_returns_one_slice_per_villager():
    env, _game, _actions = _env()

    observation, info = env.reset()

    assert observation.shape == (2, 21)
    assert info["resource_stock"] == pytest.approx(300.0)


def test_each_action_slot_drives_its_own_villager():
    env, _game, actions = _env()
    env.reset()
    tree_x, tree_z = _target(300.0, 100.0)
    drop_x, drop_z = _target(50.0, 150.0)
    action = np.array([tree_x, tree_z, 1.0, drop_x, drop_z, 1.0], dtype=np.float32)

    env.step(action)

    assert ("gather", 11, 21) in actions.calls
    assert ("returnresource", 12, 31) in actions.calls


def test_no_click_emits_no_command_for_that_villager():
    env, _game, actions = _env()
    env.reset()
    tree_x, tree_z = _target(300.0, 100.0)
    action = np.array([tree_x, tree_z, 1.0, 0.0, 0.0, -1.0], dtype=np.float32)

    env.step(action)

    driven = {call[1] for call in actions.calls}
    assert 11 in driven
    assert 12 not in driven


def test_one_batched_evaluation_per_step():
    env, game, _actions = _env()
    env.reset()
    before = len(game.evaluate_calls)

    env.step(np.zeros(6, dtype=np.float32))

    assert len(game.evaluate_calls) - before == 1


def test_episode_terminates_when_collective_stock_reaches_the_threshold():
    game, actions = _backend()
    game.snapshots = [
        {
            "stock": {"wood": 300.0},
            "carried": {},
            "remaining": {"21": 200.0, "22": 200.0},
        },
        {
            "stock": {"wood": 345.0},
            "carried": {},
            "remaining": {"21": 200.0, "22": 200.0},
        },
    ]
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=2,
        resource_count=2,
        map_size_m=512.0,
        horizon=5,
        sim_steps_per_action=1,
        stock_success_threshold=40.0,
        game=game,
        actions=actions,
    )
    env.reset()

    _observation, reward, terminated, truncated, info = env.step(
        np.zeros(6, dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(45.0)
    assert info["episode_resource_stock_delta"] == pytest.approx(45.0)


def test_horizon_truncates_without_termination():
    env, _game, _actions = _env()
    env.reset()

    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is False
        assert truncated is False

    _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
    assert terminated is False
    assert truncated is True


def test_unexpected_unit_count_fails_loudly():
    from rl.gather.roster import RosterError

    game, actions = _backend()
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=4,
        resource_count=2,
        map_size_m=512.0,
        horizon=3,
        sim_steps_per_action=1,
        game=game,
        actions=actions,
    )

    with pytest.raises(RosterError, match="expected 4 villagers, found 2"):
        env.reset()


def test_slot_order_survives_a_shuffled_engine_listing():
    env, game, actions = _env()
    env.reset()
    game.current_state.villagers.reverse()
    tree_x, tree_z = _target(300.0, 100.0)
    action = np.array([tree_x, tree_z, 1.0, 0.0, 0.0, -1.0], dtype=np.float32)

    env.step(action)

    # Slot 0 is entity 11 regardless of the order the engine listed units in.
    assert ("gather", 11, 21) in actions.calls
