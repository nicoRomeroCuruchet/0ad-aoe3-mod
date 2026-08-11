from dataclasses import dataclass

import pytest

from rl.gather.roster import Roster, RosterError, build_roster


@dataclass(frozen=True)
class FakeUnit:
    coordinates: tuple[float, float]
    entity_id: int

    def position(self):
        return self.coordinates

    def id(self):
        return self.entity_id


class FakeState:
    def __init__(self, villagers, resources, dropsites):
        self._units = {
            (1, "polites"): list(villagers),
            (0, "tree"): list(resources),
            (1, "storehouse"): list(dropsites),
        }

    def units(self, *, owner, entity_type):
        return self._units[(owner, entity_type)]


def _state(villager_ids, resource_ids):
    villagers = [FakeUnit((float(i), 0.0), i) for i in villager_ids]
    resources = [FakeUnit((0.0, float(i)), i) for i in resource_ids]
    return FakeState(villagers, resources, [FakeUnit((5.0, 5.0), 99)])


def test_roster_orders_entities_by_id():
    roster = build_roster(_state([30, 10, 20], [7, 5]), villager_count=3, resource_count=2)

    assert isinstance(roster, Roster)
    assert [unit.id() for unit in roster.villagers] == [10, 20, 30]
    assert [unit.id() for unit in roster.resources] == [5, 7]
    assert roster.dropsite.id() == 99


def test_roster_slot_order_survives_a_shuffled_engine_listing():
    first = build_roster(_state([30, 10, 20], [7, 5]), villager_count=3, resource_count=2)
    second = build_roster(_state([20, 30, 10], [5, 7]), villager_count=3, resource_count=2)

    assert [unit.id() for unit in first.villagers] == [
        unit.id() for unit in second.villagers
    ]
    assert [unit.id() for unit in first.resources] == [
        unit.id() for unit in second.resources
    ]


def test_roster_rejects_an_unexpected_villager_count():
    with pytest.raises(RosterError, match="expected 4 villagers, found 3"):
        build_roster(_state([1, 2, 3], [4, 5]), villager_count=4, resource_count=2)


def test_roster_rejects_an_unexpected_resource_count():
    with pytest.raises(RosterError, match="expected 3 resources, found 2"):
        build_roster(_state([1, 2], [4, 5]), villager_count=2, resource_count=3)


def test_roster_allows_a_missing_dropsite():
    state = FakeState([FakeUnit((0.0, 0.0), 1)], [FakeUnit((1.0, 1.0), 2)], [])

    roster = build_roster(state, villager_count=1, resource_count=1)

    assert roster.dropsite is None
