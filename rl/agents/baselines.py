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
        return np.asarray(values[2:4], dtype=np.float32).copy()
