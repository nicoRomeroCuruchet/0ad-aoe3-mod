"""Stable slot assignment between action indices and engine entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RosterError(ValueError):
    """Raised when the scenario does not match the configured unit counts."""


@dataclass(frozen=True, slots=True)
class Roster:
    """Entities addressed by slot for the whole episode."""

    villagers: tuple[Any, ...]
    resources: tuple[Any, ...]
    dropsite: Any | None


def _sorted_by_id(units: list[Any]) -> tuple[Any, ...]:
    return tuple(sorted(units, key=lambda unit: int(unit.id())))


def _require_count(units: tuple[Any, ...], expected: int, noun: str) -> None:
    if len(units) != expected:
        raise RosterError(f"expected {expected} {noun}, found {len(units)}")


def build_roster(
    state: Any,
    *,
    villager_count: int,
    resource_count: int,
    villager_type: str = "polites",
    resource_type: str = "tree",
    dropsite_type: str = "storehouse",
) -> Roster:
    """Order entities by engine id so action slots stay put across steps.

    The engine does not promise a stable ordering from ``state.units(...)``, and a
    reordering would silently repoint an action slot at a different villager.
    Sorting by entity id is the slot assignment.
    """

    villagers = _sorted_by_id(state.units(owner=1, entity_type=villager_type))
    resources = _sorted_by_id(state.units(owner=0, entity_type=resource_type))
    _require_count(villagers, villager_count, "villagers")
    _require_count(resources, resource_count, "resources")
    dropsites = _sorted_by_id(state.units(owner=1, entity_type=dropsite_type))
    return Roster(
        villagers=villagers,
        resources=resources,
        dropsite=dropsites[0] if dropsites else None,
    )


def refresh_roster(
    state: Any,
    *,
    previous: Roster,
    villager_count: int,
    resource_count: int,
    villager_type: str = "polites",
    resource_type: str = "tree",
    dropsite_type: str = "storehouse",
) -> Roster:
    """Refresh live entities while retaining slots for depleted supplies.

    0 A.D. deletes a ``ResourceSupply`` entity once its amount reaches zero.
    M2's categorical tree actions and relational observation rows must remain
    stable after that deletion, so a disappeared resource retains its reset-time
    object (and therefore its id and position). Its queried remaining amount is
    zero; callers must not issue it a gather command.
    """

    villagers = _sorted_by_id(state.units(owner=1, entity_type=villager_type))
    _require_count(villagers, villager_count, "villagers")
    if len(previous.resources) != resource_count:
        raise RosterError(
            "previous roster does not match the configured resource count"
        )

    live_resources = _sorted_by_id(state.units(owner=0, entity_type=resource_type))
    live_by_id = {int(unit.id()): unit for unit in live_resources}
    previous_ids = tuple(int(unit.id()) for unit in previous.resources)
    unknown = sorted(set(live_by_id) - set(previous_ids))
    if unknown:
        raise RosterError(
            "resource ids changed during an episode: "
            + ", ".join(str(resource_id) for resource_id in unknown)
        )
    resources = tuple(
        live_by_id.get(int(resource.id()), resource) for resource in previous.resources
    )

    dropsites = _sorted_by_id(state.units(owner=1, entity_type=dropsite_type))
    return Roster(
        villagers=villagers,
        resources=resources,
        dropsite=dropsites[0] if dropsites else None,
    )
