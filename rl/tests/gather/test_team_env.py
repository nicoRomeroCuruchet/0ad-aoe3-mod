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


def test_regathering_the_same_tree_is_not_an_interruption():
    env, _game, _actions = _env()
    env.reset()
    tree_x, tree_z = _target(300.0, 100.0)
    action = np.array([tree_x, tree_z, 1.0, 0.0, 0.0, -1.0], dtype=np.float32)

    env.step(action)
    _observation, _reward, _terminated, _truncated, info = env.step(action)

    assert info["click_gather_cycle_penalty"] == pytest.approx(0.0)


def test_retargeting_another_tree_mid_cycle_is_an_interruption():
    env, _game, _actions = _env()
    env.reset()
    first_x, first_z = _target(300.0, 100.0)
    second_x, second_z = _target(400.0, 100.0)

    env.step(np.array([first_x, first_z, 1.0, 0.0, 0.0, -1.0], dtype=np.float32))
    _observation, _reward, _terminated, _truncated, info = env.step(
        np.array([second_x, second_z, 1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    )

    # One of two villagers interrupted, averaged over the team.
    assert info["click_gather_cycle_penalty"] == pytest.approx(0.5)


def test_return_resource_falls_back_when_the_client_lacks_the_action():
    class ActionsWithoutReturn:
        def __init__(self):
            self.calls = []

        def walk(self, units, x, z):
            self.calls.append(("walk", units[0].id()))
            return ("walk", units[0].id())

        def gather(self, units, resource):
            self.calls.append(("gather", units[0].id()))
            return ("gather", units[0].id())

    game, _actions = _backend()
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=2,
        resource_count=2,
        map_size_m=512.0,
        horizon=3,
        sim_steps_per_action=1,
        gather_command_distance=12.0,
        game=game,
        actions=ActionsWithoutReturn(),
    )
    env.reset()
    drop_x, drop_z = _target(50.0, 150.0)

    env.step(np.array([drop_x, drop_z, 1.0, 0.0, 0.0, -1.0], dtype=np.float32))

    command = game.step_calls[-1][0][0]
    assert command["type"] == "returnresource"
    assert command["entities"] == [11]
    assert command["target"] == 31


def test_sim_frame_observer_reports_every_simulation_turn():
    ticks = []
    game, actions = _backend()
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=2,
        resource_count=2,
        map_size_m=512.0,
        horizon=3,
        sim_steps_per_action=3,
        game=game,
        actions=actions,
        sim_frame_observer=lambda: ticks.append(len(game.step_calls)),
    )
    env.reset()

    env.step(np.zeros(6, dtype=np.float32))

    assert ticks == [1, 2, 3]


def test_capture_agent_frame_follows_the_configured_slot():
    class FakeObserver:
        def __init__(self):
            self.captured = []

        def capture(self, entity_id):
            self.captured.append(entity_id)
            return "frame"

    observer = FakeObserver()
    game, actions = _backend()
    env = ZeroADTeamGatherEnv(
        "scenario",
        villager_count=2,
        resource_count=2,
        map_size_m=512.0,
        horizon=3,
        sim_steps_per_action=1,
        observer_villager_slot=1,
        game=game,
        actions=actions,
        engine_observer=observer,
    )
    env.reset()

    assert env.capture_agent_frame() == "frame"
    assert observer.captured == [12]


def test_observer_slot_must_address_a_configured_villager():
    game, actions = _backend()

    with pytest.raises(ValueError, match="observer_villager_slot"):
        ZeroADTeamGatherEnv(
            "scenario",
            villager_count=2,
            resource_count=2,
            observer_villager_slot=5,
            game=game,
            actions=actions,
        )


def _delivery_backend(stocks, carried_sequence):
    villagers = [FakeUnit((100.0, 100.0), 11), FakeUnit((200.0, 100.0), 12)]
    resources = [FakeUnit((300.0, 100.0), 21), FakeUnit((400.0, 100.0), 22)]
    state = FakeState(villagers, resources, FakeUnit((50.0, 150.0), 31))
    snapshots = [
        {
            "stock": {"wood": stock},
            "carried": {"11": carried[0], "12": carried[1]},
            "remaining": {"21": 200.0, "22": 200.0},
        }
        for stock, carried in zip(stocks, carried_sequence)
    ]
    return FakeGame(state, snapshots), FakeActions()


def _delivery_env(game, actions, **overrides):
    parameters = {
        "villager_count": 2,
        "resource_count": 2,
        "map_size_m": 512.0,
        "horizon": 10,
        "sim_steps_per_action": 1,
        "stock_success_threshold": 40.0,
        "min_delivery_per_villager": 20.0,
        "game": game,
        "actions": actions,
    }
    parameters.update(overrides)
    return ZeroADTeamGatherEnv("scenario", **parameters)


def test_one_villager_delivering_everything_does_not_finish_the_episode():
    # Villager 0 hauls two loads; villager 1 never carries anything.
    game, actions = _delivery_backend(
        stocks=[300.0, 300.0, 320.0, 320.0, 340.0],
        carried_sequence=[(0.0, 0.0), (20.0, 0.0), (0.0, 0.0), (20.0, 0.0), (0.0, 0.0)],
    )
    env = _delivery_env(game, actions)
    env.reset()

    for _ in range(4):
        _obs, _reward, terminated, _truncated, info = env.step(
            np.zeros(6, dtype=np.float32)
        )

    assert info["episode_resource_stock_delta"] == pytest.approx(40.0)
    assert info["delivered_per_villager"][0] == pytest.approx(40.0)
    assert info["delivered_per_villager"][1] == pytest.approx(0.0)
    assert info["working_villagers"] == 1
    assert terminated is False


def test_episode_finishes_once_every_villager_has_delivered():
    # Both villagers haul one load each.
    game, actions = _delivery_backend(
        stocks=[300.0, 300.0, 340.0],
        carried_sequence=[(0.0, 0.0), (20.0, 20.0), (0.0, 0.0)],
    )
    env = _delivery_env(game, actions)
    env.reset()

    env.step(np.zeros(6, dtype=np.float32))
    _obs, _reward, terminated, _truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert info["delivered_per_villager"] == pytest.approx((20.0, 20.0))
    assert info["working_villagers"] == 2
    assert info["min_delivered"] == pytest.approx(20.0)
    assert terminated is True


def test_participation_requirement_is_off_by_default():
    game, actions = _delivery_backend(
        stocks=[300.0, 340.0],
        carried_sequence=[(0.0, 0.0), (0.0, 0.0)],
    )
    env = _delivery_env(game, actions, min_delivery_per_villager=0.0)
    env.reset()

    _obs, _reward, terminated, _truncated, _info = env.step(
        np.zeros(6, dtype=np.float32)
    )

    assert terminated is True


def test_last_remaining_is_exposed_for_the_live_view():
    env, _game, _actions = _env()
    env.reset()

    assert env._last_remaining == (200.0, 200.0)

    env.step(np.zeros(6, dtype=np.float32))

    assert env._last_remaining == (200.0, 200.0)
