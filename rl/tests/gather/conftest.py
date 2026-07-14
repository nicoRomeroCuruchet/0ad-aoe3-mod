from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class FakeUnit:
    coordinates: tuple[float, float]

    def position(self):
        return self.coordinates


@dataclass(frozen=True)
class FakeState:
    villager: FakeUnit
    resource: FakeUnit

    def units(self, *, owner, type):
        units_by_query = {
            (1, "polites"): [self.villager],
            (0, "tree"): [self.resource],
        }
        return units_by_query[(owner, type)]


class FakeGame:
    def __init__(self, initial_state, state_after_action):
        self.initial_state = initial_state
        self.state_after_action = state_after_action
        self.current_state = initial_state
        self.reset_calls = []
        self.step_calls = []
        self.close_calls = 0

    def reset(self, scenario_config, *, save_replay=False):
        self.reset_calls.append((scenario_config, save_replay))
        self.current_state = self.initial_state

    def step(self, commands=None):
        self.step_calls.append(commands)
        if commands is not None:
            self.current_state = self.state_after_action
        return self.current_state

    def close(self):
        self.close_calls += 1


class FakeActions:
    def __init__(self):
        self.walk_calls = []

    def walk(self, units, x, z):
        command = ("walk", tuple(units), x, z)
        self.walk_calls.append((units, x, z))
        return command


@pytest.fixture
def fake_backend():
    initial_state = FakeState(
        villager=FakeUnit((0.0, 0.0)),
        resource=FakeUnit((100.0, 0.0)),
    )
    state_after_action = FakeState(
        villager=FakeUnit((25.0, 0.0)),
        resource=initial_state.resource,
    )
    return FakeGame(initial_state, state_after_action), FakeActions()
