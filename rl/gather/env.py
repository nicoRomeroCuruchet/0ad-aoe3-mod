"""Gymnasium environment backed by a running 0 A.D. simulation."""

from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import (
    build_observation,
    denormalize_action,
    distance,
    gather_reward,
    is_reached,
    xz,
)

VILLAGER_TYPE = "polites"
RESOURCE_TYPE = "tree"


def _load_zero_ad():
    try:
        return importlib.import_module("zero_ad")
    except ModuleNotFoundError as exc:
        if exc.name != "zero_ad":
            raise
        raise ModuleNotFoundError(
            "zero_ad is required for the live 0 A.D. backend; install it or "
            "inject both game and actions into ZeroADGatherEnv"
        ) from exc


def _resolve_backend(uri: str, game: Any, actions: Any) -> tuple[Any, Any]:
    if game is not None and actions is not None:
        return game, actions

    zero_ad = _load_zero_ad()
    resolved_game = game if game is not None else zero_ad.ZeroAD(uri)
    resolved_actions = actions if actions is not None else zero_ad.actions
    return resolved_game, resolved_actions


class ZeroADGatherEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_labels = (
        "villager_x_norm",
        "villager_z_norm",
        "resource_x_norm",
        "resource_z_norm",
        "distance_norm",
    )

    def __init__(
        self,
        scenario_config,
        uri="http://localhost:6000",
        map_size_m=512.0,
        horizon=50,
        reach_threshold=12.0,
        sim_steps_per_action=10,
        save_replay=False,
        game=None,
        actions=None,
    ):
        # reach_threshold=12: el aldeano no puede pisar el arbol (obstaculo solido);
        # se frena a ~9.5m del centro, asi que "llegar" se cuenta a <12m.
        super().__init__()
        self.game, self.actions = _resolve_backend(uri, game, actions)
        self.scenario_config = scenario_config
        self.save_replay = save_replay
        self.map_size_m = map_size_m
        self.horizon = horizon
        self.reach_threshold = reach_threshold
        self.sim_steps_per_action = sim_steps_per_action
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._step_count = 0
        self._prev_dist = None
        self._closed = False

    def _positions(self, state):
        v = xz(state.units(owner=1, entity_type=VILLAGER_TYPE)[0].position())
        r = xz(state.units(owner=0, entity_type=RESOURCE_TYPE)[0].position())
        return v, r

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(self.scenario_config, save_replay=self.save_replay)
        state = self.game.step()  # un tick para que las entidades existan
        v, r = self._positions(state)
        self._prev_dist = distance(v, r)
        self._step_count = 0
        return build_observation(v, r, self.map_size_m), {}

    def step(self, action):
        x, z = denormalize_action(action, self.map_size_m)
        villager = self.game.current_state.units(
            owner=1,
            entity_type=VILLAGER_TYPE,
        )[0]
        cmd = self.actions.walk([villager], x, z)
        state = self.game.step([cmd])
        for _ in range(self.sim_steps_per_action - 1):
            state = self.game.step()
        v, r = self._positions(state)
        cur_dist = distance(v, r)
        reward = gather_reward(self._prev_dist, cur_dist)
        self._prev_dist = cur_dist
        self._step_count += 1
        terminated = is_reached(cur_dist, self.reach_threshold)
        truncated = self._step_count >= self.horizon
        obs = build_observation(v, r, self.map_size_m)
        return obs, reward, terminated, truncated, {"distance": cur_dist}

    def close(self):
        """Release the optional live/injected backend exactly once."""

        if self._closed:
            return
        self._closed = True
        close_backend = getattr(self.game, "close", None)
        if callable(close_backend):
            close_backend()
        super().close()
