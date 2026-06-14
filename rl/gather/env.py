import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zero_ad

from .core import (xz, distance, denormalize_action, build_observation,
                   gather_reward, is_reached)

VILLAGER_TYPE = "polites"
RESOURCE_TYPE = "tree"


class ZeroADGatherEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_config, uri="http://localhost:6000",
                 map_size_m=512.0, horizon=50, reach_threshold=12.0,
                 sim_steps_per_action=10, save_replay=False):
        # reach_threshold=12: el aldeano no puede pisar el arbol (obstaculo solido);
        # se frena a ~9.5m del centro, asi que "llegar" se cuenta a <12m.
        super().__init__()
        self.game = zero_ad.ZeroAD(uri)
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

    def _positions(self, state):
        v = xz(state.units(owner=1, type=VILLAGER_TYPE)[0].position())
        r = xz(state.units(owner=0, type=RESOURCE_TYPE)[0].position())
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
        villager = self.game.current_state.units(owner=1, type=VILLAGER_TYPE)[0]
        cmd = zero_ad.actions.walk([villager], x, z)
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
