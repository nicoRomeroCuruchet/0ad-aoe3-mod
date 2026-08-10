from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class FakeUnit:
    coordinates: tuple[float, float]
    entity_id: int

    def position(self):
        return self.coordinates

    def id(self):
        return self.entity_id


@dataclass(frozen=True)
class FakeState:
    villager: FakeUnit
    resource: FakeUnit
    storehouse: FakeUnit | None = None

    def units(self, *, owner, entity_type):
        units_by_query = {
            (1, "polites"): [self.villager],
            (0, "tree"): [self.resource],
            (1, "storehouse"): [] if self.storehouse is None else [self.storehouse],
        }
        return units_by_query[(owner, entity_type)]


class FakeGame:
    def __init__(self, initial_state, state_after_action):
        self.initial_state = initial_state
        self.state_after_action = state_after_action
        self.current_state = initial_state
        self.reset_calls = []
        self.step_calls = []
        self.evaluate_calls = []
        self.stock_values = []
        self.carried_values = []
        self.close_calls = 0

    def reset(self, scenario_config, *, save_replay=False):
        self.reset_calls.append((scenario_config, save_replay))
        self.current_state = self.initial_state

    def step(self, commands=None):
        self.step_calls.append(commands)
        if commands is not None:
            self.current_state = self.state_after_action
        return self.current_state

    def evaluate(self, expression):
        self.evaluate_calls.append(expression)
        if "GetResourceCounts" in expression and "GetCarryingStatus" in expression:
            stock = self.stock_values.pop(0) if self.stock_values else 0.0
            carried = self.carried_values.pop(0) if self.carried_values else 0.0
            return {"stock": {"wood": stock}, "carried": carried}
        if "GetCarryingStatus" in expression:
            return self.carried_values.pop(0) if self.carried_values else 0.0
        return {"wood": self.stock_values.pop(0) if self.stock_values else 0.0}

    def close(self):
        self.close_calls += 1


class FakeActions:
    def __init__(self):
        self.walk_calls = []
        self.gather_calls = []
        self.returnresource_calls = []

    def walk(self, units, x, z):
        command = ("walk", tuple(units), x, z)
        self.walk_calls.append((units, x, z))
        return command

    def gather(self, units, resource):
        command = ("gather", tuple(units), resource)
        self.gather_calls.append((units, resource))
        return command

    def returnresource(self, units, dropsite):
        command = ("returnresource", tuple(units), dropsite)
        self.returnresource_calls.append((units, dropsite))
        return command


@pytest.fixture
def fake_backend():
    initial_state = FakeState(
        villager=FakeUnit((0.0, 0.0), 1),
        resource=FakeUnit((100.0, 0.0), 2),
        storehouse=FakeUnit((0.0, 100.0), 3),
    )
    state_after_action = FakeState(
        villager=FakeUnit((25.0, 0.0), 1),
        resource=initial_state.resource,
        storehouse=initial_state.storehouse,
    )
    return FakeGame(initial_state, state_after_action), FakeActions()
