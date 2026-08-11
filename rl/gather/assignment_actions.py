"""Categorical assignment actions translated to the map-click boundary."""

from __future__ import annotations

from functools import lru_cache
from math import comb, perm
from numbers import Integral, Real

import numpy as np
from gymnasium import spaces

from .core import normalize_coord


RAW_CLICK_MODE = "raw_click"
ASSIGNMENT_CLICK_MODE = "assignment_click"
JOINT_ASSIGNMENT_CLICK_MODE = "joint_assignment_click"
ACTION_MODES = frozenset(
    {RAW_CLICK_MODE, ASSIGNMENT_CLICK_MODE, JOINT_ASSIGNMENT_CLICK_MODE}
)

# The task only needs the 4-by-4 case (209 actions), but retaining a bound
# here prevents a typo in a generic experiment config from materialising an
# exponential-sized categorical table.
MAX_JOINT_ASSIGNMENT_ACTIONS = 16_384


def _positive_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def assignment_action_space(
    *,
    villager_count: int,
    resource_count: int,
) -> spaces.MultiDiscrete:
    """Return one ``NO_CLICK/tree-slot`` categorical choice per villager."""

    villagers = _positive_count("villager_count", villager_count)
    resources = _positive_count("resource_count", resource_count)
    return spaces.MultiDiscrete(
        np.full(villagers, resources + 1, dtype=np.int64),
        dtype=np.int64,
    )


def _joint_assignment_count(villager_count: int, resource_count: int) -> int:
    """Return the number of partial injective villager-to-tree matchings."""

    return sum(
        comb(villager_count, assigned) * perm(resource_count, assigned)
        for assigned in range(min(villager_count, resource_count) + 1)
    )


@lru_cache(maxsize=None)
def _joint_assignment_table(
    villager_count: int,
    resource_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate stable collision-free assignment categories.

    A row contains one category per villager: zero means ``NO_CLICK`` and
    ``j + 1`` chooses tree slot ``j``.  Every non-zero category occurs at
    most once.  Lexicographic traversal keeps checkpoint/action meanings
    stable across processes.
    """

    action_count = _joint_assignment_count(villager_count, resource_count)
    if action_count > MAX_JOINT_ASSIGNMENT_ACTIONS:
        raise ValueError(
            "joint assignment action space has "
            f"{action_count} actions, exceeding the {MAX_JOINT_ASSIGNMENT_ACTIONS} limit"
        )

    rows: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], used: frozenset[int]) -> None:
        if len(prefix) == villager_count:
            rows.append(prefix)
            return
        visit((*prefix, 0), used)
        for resource in range(1, resource_count + 1):
            if resource not in used:
                visit((*prefix, resource), used | {resource})

    visit((), frozenset())
    return tuple(rows)


def joint_assignment_action_count(
    *,
    villager_count: int,
    resource_count: int,
) -> int:
    """Return the number of valid collision-free team assignments.

    Calling this validates the requested table size as well as the positive
    villager/resource counts, so it is safe to use when validating a policy.
    """

    villagers = _positive_count("villager_count", villager_count)
    resources = _positive_count("resource_count", resource_count)
    return len(_joint_assignment_table(villagers, resources))


def joint_assignment_action_space(
    *,
    villager_count: int,
    resource_count: int,
) -> spaces.Discrete:
    """Return one categorical action covering all valid team assignments."""

    return spaces.Discrete(
        joint_assignment_action_count(
            villager_count=villager_count,
            resource_count=resource_count,
        ),
        start=0,
        dtype=np.int64,
    )


def _joint_index(value: object) -> int:
    """Accept a scalar integer (including SB3's zero-dimensional ndarray)."""

    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.shape == (1,):
            value = value[0].item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("joint assignment index must be an integer scalar")
    return int(value)


def joint_assignment_from_index(
    index: object,
    *,
    villager_count: int,
    resource_count: int,
) -> np.ndarray:
    """Decode a categorical action into a stable per-villager assignment."""

    villagers = _positive_count("villager_count", villager_count)
    resources = _positive_count("resource_count", resource_count)
    table = _joint_assignment_table(villagers, resources)
    action_index = _joint_index(index)
    if not 0 <= action_index < len(table):
        raise ValueError(
            f"joint assignment index must be between 0 and {len(table) - 1}"
        )
    return np.asarray(table[action_index], dtype=np.int64)


def joint_assignment_to_index(
    assignments: object,
    *,
    villager_count: int,
    resource_count: int,
) -> int:
    """Encode one partial injective assignment as its categorical index."""

    villagers = _positive_count("villager_count", villager_count)
    resources = _positive_count("resource_count", resource_count)
    values = np.asarray(assignments)
    if values.shape != (villagers,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(
            "joint assignments must be an integer array with one value per villager"
        )
    if np.any(values < 0) or np.any(values > resources):
        raise ValueError(
            f"joint assignment categories must be between 0 and {resources}"
        )
    assigned = values[values != 0]
    if len(assigned) != len(np.unique(assigned)):
        raise ValueError("joint assignments cannot assign a tree to two villagers")
    table = _joint_assignment_table(villagers, resources)
    try:
        return table.index(tuple(int(value) for value in values))
    except ValueError as error:  # Defensive: the injection check above proves it.
        raise ValueError("joint assignment is not a valid partial matching") from error


def _validated_map_size(map_size_m: object) -> float:
    if (
        isinstance(map_size_m, bool)
        or not isinstance(map_size_m, Real)
        or not np.isfinite(float(map_size_m))
        or float(map_size_m) <= 0.0
    ):
        raise ValueError("map_size_m must be finite and positive")
    return float(map_size_m)


def _validated_resource_xz(
    resource_xz: object,
    map_size_m: float,
) -> np.ndarray:
    coordinates = np.asarray(resource_xz)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("resource_xz must have shape (resource_count, 2)")
    if coordinates.shape[0] == 0:
        raise ValueError("resource_xz must contain at least one resource")
    if not np.issubdtype(coordinates.dtype, np.number):
        raise ValueError("resource coordinates must be numeric")
    values = coordinates.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("resource coordinates must be finite")
    if np.any(values < 0.0) or np.any(values > map_size_m):
        raise ValueError("resource coordinates must lie within the map")
    return values


def assignment_to_raw_click(
    assignments: object,
    resource_xz: object,
    *,
    map_size_m: float,
) -> np.ndarray:
    """Decode assignments into the environment's existing raw click vector.

    Category zero emits no click. Category ``j + 1`` emits an exact click on
    stable resource slot ``j``. Inputs are only read; the returned vector owns
    its memory.
    """

    map_size = _validated_map_size(map_size_m)
    coordinates = _validated_resource_xz(resource_xz, map_size)
    values = np.asarray(assignments)
    if values.ndim != 1:
        raise ValueError("assignments must be a one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("assignments must have an integer dtype")

    resource_count = coordinates.shape[0]
    if np.any(values < 0) or np.any(values > resource_count):
        raise ValueError(
            f"assignment categories must be between 0 and {resource_count}"
        )

    decoded = np.zeros(3 * values.size, dtype=np.float32)
    decoded[2::3] = -1.0
    for villager, category in enumerate(values):
        if category == 0:
            continue
        x, z = coordinates[int(category) - 1]
        base = 3 * villager
        decoded[base] = normalize_coord(x, map_size)
        decoded[base + 1] = normalize_coord(z, map_size)
        decoded[base + 2] = 1.0
    return decoded
