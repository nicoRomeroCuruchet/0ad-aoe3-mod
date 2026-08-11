"""Team observation assembly for multi-villager gather environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    GATHER_LIFECYCLE_OBSERVATION_LABELS,
    distance,
    normalize_coord,
    normalize_non_negative,
)


# Indices 0-9 retain M1's field names and order for diagnostics. M2 uses a
# separate categorical policy, so it does not assume M1's continuous weights or
# distance scale are transferable.
TEAM_CORE_LABELS = GATHER_LIFECYCLE_OBSERVATION_LABELS

RELATIONAL_FIELDS = (
    "dx_norm",
    "dz_norm",
    "dist_norm",
    "is_current_target",
    "remaining_norm",
)


@dataclass(frozen=True, slots=True)
class TeamObservationScales:
    """Normalization constants shared by every slice."""

    map_size_m: float
    carried_resource_scale: float
    stock_scale: float
    resource_amount_scale: float


@dataclass(frozen=True, slots=True)
class TeamSnapshot:
    """Everything one observation needs, already read from the engine."""

    villager_xz: tuple[tuple[float, float], ...]
    resource_xz: tuple[tuple[float, float], ...]
    resource_remaining: tuple[float, ...]
    carried: tuple[float, ...]
    target_index: tuple[int | None, ...]
    gather_cycle_active: tuple[bool, ...]
    dropsite_xz: tuple[float, float]
    stock: float

    def __post_init__(self) -> None:
        per_villager = {
            len(self.villager_xz),
            len(self.carried),
            len(self.target_index),
            len(self.gather_cycle_active),
        }
        if len(per_villager) != 1:
            raise ValueError("per-villager fields must have equal length")
        if len(self.resource_xz) != len(self.resource_remaining):
            raise ValueError("per-resource fields must have equal length")
        if not self.resource_xz:
            raise ValueError("a team snapshot needs at least one resource")
        if any(
            index is not None
            and (index < 0 or index >= len(self.resource_xz))
            for index in self.target_index
        ):
            raise ValueError("target_index out of range")


def team_observation_labels(
    villager_count: int,
    resource_count: int,
) -> tuple[str, ...]:
    """Names for one villager's slice, in slice order."""

    del villager_count
    relational = tuple(
        f"tree{index}_{field}"
        for index in range(resource_count)
        for field in RELATIONAL_FIELDS
    )
    return (*TEAM_CORE_LABELS, "agent_id_norm", *relational)


def build_team_observation(
    snapshot: TeamSnapshot,
    scales: TeamObservationScales,
) -> np.ndarray:
    """Return one ``(villager_count, slice_dim)`` float32 observation."""

    villager_count = len(snapshot.villager_xz)
    stock_norm = normalize_non_negative(snapshot.stock, scales.stock_scale)
    distance_scale_m = np.sqrt(2.0) * scales.map_size_m
    dropsite = (
        normalize_coord(snapshot.dropsite_xz[0], scales.map_size_m),
        normalize_coord(snapshot.dropsite_xz[1], scales.map_size_m),
    )
    slices = []
    for villager in range(villager_count):
        villager_xz = snapshot.villager_xz[villager]
        target_index = snapshot.target_index[villager]
        if target_index is None:
            target_fields = (0.0, 0.0, 0.0)
        else:
            target = snapshot.resource_xz[target_index]
            target_fields = (
                normalize_coord(target[0], scales.map_size_m),
                normalize_coord(target[1], scales.map_size_m),
                distance(villager_xz, target) / distance_scale_m,
            )
        values = [
            normalize_coord(villager_xz[0], scales.map_size_m),
            normalize_coord(villager_xz[1], scales.map_size_m),
            *target_fields,
            normalize_non_negative(
                snapshot.carried[villager],
                scales.carried_resource_scale,
            ),
            stock_norm,
            dropsite[0],
            dropsite[1],
            float(snapshot.gather_cycle_active[villager]),
            0.0 if villager_count == 1 else villager / (villager_count - 1),
        ]
        for resource, resource_xz in enumerate(snapshot.resource_xz):
            own_distance = distance(villager_xz, resource_xz)
            values.extend(
                [
                    (resource_xz[0] - villager_xz[0]) / scales.map_size_m,
                    (resource_xz[1] - villager_xz[1]) / scales.map_size_m,
                    own_distance / distance_scale_m,
                    float(target_index == resource),
                    normalize_non_negative(
                        snapshot.resource_remaining[resource],
                        scales.resource_amount_scale,
                    ),
                ]
            )
        slices.append(values)
    return np.array(slices, dtype=np.float32)
