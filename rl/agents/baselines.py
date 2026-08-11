"""Non-learning policies used to validate and benchmark environments."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl.gather.assignment_actions import (
    ASSIGNMENT_CLICK_MODE,
    JOINT_ASSIGNMENT_CLICK_MODE,
    RAW_CLICK_MODE,
    joint_assignment_to_index,
)


class RandomPolicy:
    """Sample reproducible actions from a supported finite action space."""

    def __init__(
        self,
        action_space: spaces.Box | spaces.Discrete | spaces.MultiDiscrete,
        *,
        seed: int = 0,
    ) -> None:
        if isinstance(action_space, spaces.MultiDiscrete):
            self._mode = "multidiscrete"
            self._low = np.asarray(action_space.start).copy()
            self._high = self._low + np.asarray(action_space.nvec)
        elif isinstance(action_space, spaces.Discrete):
            self._mode = "discrete"
            self._low = int(action_space.start)
            self._high = self._low + int(action_space.n)
        elif isinstance(action_space, spaces.Box):
            if not np.issubdtype(action_space.dtype, np.floating):
                raise TypeError("RandomPolicy requires a floating-point Box")
            if not np.all(np.isfinite(action_space.low)) or not np.all(
                np.isfinite(action_space.high)
            ):
                raise ValueError("RandomPolicy requires finite bounds")
            self._mode = "box"
            self._low = np.asarray(action_space.low).copy()
            self._high = np.asarray(action_space.high).copy()
        else:
            raise TypeError(
                "RandomPolicy requires a Box, Discrete, or MultiDiscrete space"
            )
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
        if self._mode == "multidiscrete":
            return self._rng.integers(
                self._low,
                self._high,
                dtype=self._dtype,
            )
        if self._mode == "discrete":
            return np.asarray(
                self._rng.integers(self._low, self._high),
                dtype=self._dtype,
            )
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
    CYCLE_ACTIVE_INDEX = 9
    CURRENT_TARGET_OFFSET = 3
    REMAINING_OFFSET = 4

    def __init__(
        self,
        villager_count: int = 4,
        resource_count: int = 4,
        *,
        action_mode: str = "raw_click",
    ) -> None:
        if villager_count <= 0 or resource_count <= 0:
            raise ValueError("team oracle needs positive counts")
        if action_mode not in {
            RAW_CLICK_MODE,
            ASSIGNMENT_CLICK_MODE,
            JOINT_ASSIGNMENT_CLICK_MODE,
        }:
            raise ValueError("team oracle action_mode is invalid")
        self._villager_count = villager_count
        self._resource_count = resource_count
        self._action_mode = action_mode

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

    def _current_target(self, slice_values) -> int | None:
        """Return this villager's live selected tree, if the observation has one."""

        for resource in range(self._resource_count):
            base = self.CORE_WIDTH + resource * self.RELATIONAL_WIDTH
            if (
                slice_values[base + self.CURRENT_TARGET_OFFSET] > 0.5
                and slice_values[base + self.REMAINING_OFFSET] > 0.0
            ):
                return resource
        return None

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

        if self._action_mode in {
            ASSIGNMENT_CLICK_MODE,
            JOINT_ASSIGNMENT_CLICK_MODE,
        }:
            assignment = self._assignment_action(values)
            if self._action_mode == JOINT_ASSIGNMENT_CLICK_MODE:
                return np.asarray(
                    joint_assignment_to_index(
                        assignment,
                        villager_count=self._villager_count,
                        resource_count=self._resource_count,
                    ),
                    dtype=np.int64,
                )
            return assignment

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

    def _assignment_action(self, values: np.ndarray) -> np.ndarray:
        action = np.zeros(self._villager_count, dtype=np.int64)
        claimed = {
            target
            for villager in range(self._villager_count)
            if (
                values[villager, self.CARRIED_INDEX] > 0.0
                or values[villager, self.CYCLE_ACTIVE_INDEX] > 0.0
            )
            for target in (self._current_target(values[villager]),)
            if target is not None
        }
        for villager in range(self._villager_count):
            slice_values = values[villager]
            carrying = slice_values[self.CARRIED_INDEX] > 0.0
            cycle_active = slice_values[self.CYCLE_ACTIVE_INDEX] > 0.0
            busy = carrying or cycle_active
            if busy:
                # Re-sending the exact same gather target is explicitly
                # non-disruptive in the environment. It makes the oracle a
                # reliable ceiling even if UnitAI stalls an active finite-tree
                # cycle after another villager has delivered.
                if cycle_active and not carrying:
                    target = self._current_target(slice_values)
                    if target is not None:
                        action[villager] = target + 1
                continue
            offsets = self._tree_offsets(slice_values)
            available = [
                resource
                for resource in range(self._resource_count)
                if resource not in claimed
                and slice_values[
                    self.CORE_WIDTH
                    + resource * self.RELATIONAL_WIDTH
                    + self.REMAINING_OFFSET
                ]
                > 0.0
            ]
            if not available:
                continue
            choice = min(
                available,
                key=lambda resource: offsets[resource][2],
            )
            claimed.add(choice)
            action[villager] = choice + 1
        return action
