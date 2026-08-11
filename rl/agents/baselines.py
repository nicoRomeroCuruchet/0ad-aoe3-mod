"""Non-learning policies used to validate and benchmark environments."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


class RandomPolicy:
    """Sample reproducible actions from a finite continuous action space."""

    def __init__(self, action_space: spaces.Box, *, seed: int = 0) -> None:
        if not np.issubdtype(action_space.dtype, np.floating):
            raise TypeError("RandomPolicy requires a floating-point Box")
        if not np.all(np.isfinite(action_space.low)) or not np.all(
            np.isfinite(action_space.high)
        ):
            raise ValueError("RandomPolicy requires finite bounds")

        self._low = np.asarray(action_space.low).copy()
        self._high = np.asarray(action_space.high).copy()
        self._dtype = action_space.dtype
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> np.ndarray:
        """Return the next seeded random action.

        ``deterministic`` is accepted to satisfy the common policy contract.
        Random baselines remain random in either evaluation mode; reproducibility
        is controlled by the constructor seed.
        """

        _ = observation, deterministic
        return self._rng.uniform(self._low, self._high).astype(
            self._dtype,
            copy=False,
        )


class GatherOraclePolicy:
    """Target the resource coordinates exposed by the M0 gather observation."""

    def __init__(self, action_size: int = 2) -> None:
        if action_size not in {2, 3}:
            raise ValueError("gather oracle action_size must be 2 or 3")
        self._action_size = action_size

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> np.ndarray:
        """Return ``[resource_x, resource_z]`` from the observation."""

        _ = deterministic
        values = np.asarray(observation)
        if values.ndim != 1 or values.size < 4:
            raise ValueError("gather observations must contain at least four values")
        action = np.empty(self._action_size, dtype=np.float32)
        action[:2] = values[2:4]
        if self._action_size == 3:
            # Third dimension is a generic click/no-click signal in M1.
            action[2] = 1.0
        return action


class TeamGatherOraclePolicy:
    """Greedy assignment ceiling for the M2 team gather environment.

    Empty villagers claim their nearest unclaimed tree; loaded villagers head
    for the dropsite. It does not learn: it exists to prove the environment's
    slot ordering, batching, and reward are wired correctly before training.
    """

    CORE_WIDTH = 11
    RELATIONAL_WIDTH = 5
    CARRIED_INDEX = 5
    DROPSITE_X_INDEX = 7
    DROPSITE_Z_INDEX = 8

    def __init__(self, villager_count: int = 4, resource_count: int = 4) -> None:
        if villager_count <= 0 or resource_count <= 0:
            raise ValueError("team oracle needs positive counts")
        self._villager_count = villager_count
        self._resource_count = resource_count

    def _tree_offsets(self, slice_values):
        offsets = []
        for resource in range(self._resource_count):
            base = self.CORE_WIDTH + resource * self.RELATIONAL_WIDTH
            offsets.append(
                (
                    float(slice_values[base]),
                    float(slice_values[base + 1]),
                    float(slice_values[base + 2]),
                )
            )
        return offsets

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> np.ndarray:
        """Return `[x, z, click]` per villager as one flat action."""

        del deterministic
        values = np.asarray(observation, dtype=np.float32)
        expected = (
            self._villager_count,
            self.CORE_WIDTH + self.RELATIONAL_WIDTH * self._resource_count,
        )
        if values.shape != expected:
            raise ValueError(f"observation shape {values.shape} != {expected}")

        action = np.zeros(3 * self._villager_count, dtype=np.float32)
        claimed: set[int] = set()
        for villager in range(self._villager_count):
            slice_values = values[villager]
            base = 3 * villager
            action[base + 2] = 1.0
            if slice_values[self.CARRIED_INDEX] > 0.0:
                action[base] = slice_values[self.DROPSITE_X_INDEX]
                action[base + 1] = slice_values[self.DROPSITE_Z_INDEX]
                continue
            offsets = self._tree_offsets(slice_values)
            order = sorted(
                range(self._resource_count),
                key=lambda resource: offsets[resource][2],
            )
            choice = next(
                (resource for resource in order if resource not in claimed),
                order[0],
            )
            claimed.add(choice)
            dx, dz, _dist = offsets[choice]
            action[base] = np.clip(slice_values[0] + 2.0 * dx, -1.0, 1.0)
            action[base + 1] = np.clip(slice_values[1] + 2.0 * dz, -1.0, 1.0)
        return action
