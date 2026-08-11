"""Gymnasium environment for a team of villagers gathering one resource."""

from __future__ import annotations

import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import denormalize_action, distance, is_reached, nearest_index, xz
from .engine_observer import EngineObserverClient, EngineObserverUnavailable
from .env import (
    DROPSITE_TYPE,
    RECOVERABLE_BACKEND_ERRORS,
    RESOURCE_TYPE,
    VILLAGER_TYPE,
    _resolve_backend,
    parse_team_snapshot,
    team_snapshot_expression,
)
from .observation import (
    TeamObservationScales,
    TeamSnapshot,
    build_team_observation,
    team_observation_labels,
)
from .reward import TeamRewardScales, VillagerRewardInputs, compose_team_reward
from .roster import Roster, build_roster


class ZeroADTeamGatherEnv(gym.Env):
    """Drive ``villager_count`` villagers with one action vector."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_config,
        *,
        villager_count: int = 4,
        resource_count: int = 4,
        uri: str = "http://localhost:6000",
        map_size_m: float = 512.0,
        horizon: int = 120,
        sim_steps_per_action: int = 80,
        gather_command_distance: float = 12.0,
        click_action_threshold: float = 0.0,
        stock_resource: str = "wood",
        stock_player: int = 1,
        stock_success_threshold: float = 80.0,
        min_delivery_per_villager: float = 0.0,
        carried_resource_observation_scale: float = 20.0,
        stock_observation_scale: float = 1000.0,
        resource_amount_scale: float = 200.0,
        distance_shaping_scale: float = 0.02,
        carried_resource_delta_reward_scale: float = 0.2,
        click_gather_cycle_penalty: float = 1.0,
        backend_retries: int = 0,
        backend_retry_delay: float = 1.0,
        save_replay: bool = False,
        observer_villager_slot: int = 0,
        game: Any = None,
        actions: Any = None,
        engine_observer: Any = None,
        sim_frame_observer: Callable[[], None] | None = None,
        backend_factory: Callable[[str], tuple[Any, Any]] | None = None,
    ) -> None:
        super().__init__()
        if villager_count <= 0 or resource_count <= 0:
            raise ValueError("villager_count and resource_count must be positive")
        if not 0 <= observer_villager_slot < villager_count:
            raise ValueError("observer_villager_slot must address a configured villager")
        if sim_frame_observer is not None and not callable(sim_frame_observer):
            raise ValueError("sim_frame_observer must be callable or None")
        self.observer_villager_slot = observer_villager_slot
        self.sim_frame_observer = sim_frame_observer
        self.uri = uri
        self._backend_factory = backend_factory
        self._uses_injected_backend = game is not None and actions is not None
        self.game, self.actions = _resolve_backend(uri, game, actions, backend_factory)
        self.engine_observer = (
            engine_observer if engine_observer is not None else EngineObserverClient(uri)
        )
        self.scenario_config = scenario_config
        self.save_replay = save_replay
        self.villager_count = villager_count
        self.resource_count = resource_count
        self.map_size_m = map_size_m
        self.horizon = horizon
        self.sim_steps_per_action = sim_steps_per_action
        self.gather_command_distance = gather_command_distance
        self.click_action_threshold = click_action_threshold
        self.stock_resource = stock_resource
        self.stock_player = stock_player
        self.stock_success_threshold = stock_success_threshold
        # A total-wood threshold alone is satisfiable by one villager making
        # several trips, which is not a team task. Requiring a per-villager
        # delivery makes participation part of the success criterion.
        self.min_delivery_per_villager = min_delivery_per_villager
        self.backend_retries = backend_retries
        self.backend_retry_delay = backend_retry_delay
        self.observation_scales = TeamObservationScales(
            map_size_m=map_size_m,
            carried_resource_scale=carried_resource_observation_scale,
            stock_scale=stock_observation_scale,
            resource_amount_scale=resource_amount_scale,
        )
        self.reward_scales = TeamRewardScales(
            distance_shaping_scale=distance_shaping_scale,
            carried_resource_delta_reward_scale=carried_resource_delta_reward_scale,
            click_gather_cycle_penalty=click_gather_cycle_penalty,
        )
        self.observation_labels = team_observation_labels(
            villager_count,
            resource_count,
        )
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(villager_count, len(self.observation_labels)),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(3 * villager_count,),
            dtype=np.float32,
        )
        self._roster: Roster | None = None
        self._step_count = 0
        self._initial_stock = 0.0
        self._previous_stock = 0.0
        self._previous_carried: tuple[float, ...] = ()
        self._previous_distance: tuple[float, ...] = ()
        self._target_index: tuple[int, ...] = ()
        self._cycle_active: tuple[bool, ...] = ()
        self._delivered: tuple[float, ...] = ()
        self._closed = False

    # ------------------------------------------------------------------ engine

    def _roster_for(self, state: Any) -> Roster:
        return build_roster(
            state,
            villager_count=self.villager_count,
            resource_count=self.resource_count,
            villager_type=VILLAGER_TYPE,
            resource_type=RESOURCE_TYPE,
            dropsite_type=DROPSITE_TYPE,
        )

    def _read_engine(
        self,
        roster: Roster,
    ) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
        villager_ids = tuple(int(unit.id()) for unit in roster.villagers)
        resource_ids = tuple(int(unit.id()) for unit in roster.resources)
        payload = self.game.evaluate(
            team_snapshot_expression(
                self.stock_player,
                villager_ids,
                resource_ids,
                self.stock_resource,
            )
        )
        return parse_team_snapshot(
            payload,
            self.stock_resource,
            villager_ids,
            resource_ids,
        )

    def _close_backend(self) -> None:
        close_backend = getattr(self.game, "close", None)
        if callable(close_backend):
            close_backend()

    def _recover_backend(self) -> None:
        if self._uses_injected_backend and self._backend_factory is None:
            return
        self._close_backend()
        self.game, self.actions = _resolve_backend(
            self.uri,
            None,
            None,
            self._backend_factory,
        )

    # ------------------------------------------------------------- observation

    def _snapshot(
        self,
        roster: Roster,
        stock: float,
        carried: tuple[float, ...],
        remaining: tuple[float, ...],
    ) -> TeamSnapshot:
        if roster.dropsite is None:
            raise RuntimeError("the team environment requires a storehouse")
        return TeamSnapshot(
            villager_xz=tuple(xz(unit.position()) for unit in roster.villagers),
            resource_xz=tuple(xz(unit.position()) for unit in roster.resources),
            resource_remaining=remaining,
            carried=carried,
            target_index=self._target_index,
            gather_cycle_active=self._cycle_active,
            dropsite_xz=xz(roster.dropsite.position()),
            stock=stock,
        )

    # ------------------------------------------------------------------- reset

    def _reset_once(self, *, seed=None):
        super().reset(seed=seed)
        state = self.game.reset(self.scenario_config, save_replay=self.save_replay)
        if state is None:
            state = self.game.step()
        roster = self._roster_for(state)
        self._roster = roster
        villager_xz = tuple(xz(unit.position()) for unit in roster.villagers)
        resource_xz = tuple(xz(unit.position()) for unit in roster.resources)
        self._target_index = tuple(
            nearest_index(villager, resource_xz) for villager in villager_xz
        )
        self._cycle_active = tuple(False for _ in roster.villagers)
        self._delivered = tuple(0.0 for _ in roster.villagers)
        stock, carried, remaining = self._read_engine(roster)
        self._initial_stock = stock
        self._previous_stock = stock
        self._previous_carried = carried
        self._previous_distance = tuple(
            distance(villager_xz[index], resource_xz[self._target_index[index]])
            for index in range(self.villager_count)
        )
        self._step_count = 0
        observation = build_team_observation(
            self._snapshot(roster, stock, carried, remaining),
            self.observation_scales,
        )
        return observation, {
            "resource_stock": stock,
            "episode_resource_stock_delta": 0.0,
        }

    def reset(self, *, seed=None, options=None):
        del options
        for attempt in range(self.backend_retries + 1):
            try:
                return self._reset_once(seed=seed)
            except RECOVERABLE_BACKEND_ERRORS:
                if attempt >= self.backend_retries:
                    raise
                self._recover_backend()
                if self.backend_retry_delay:
                    time.sleep(self.backend_retry_delay)
        raise RuntimeError("unreachable reset retry state")

    # -------------------------------------------------------------------- step

    def _commands(
        self,
        action: np.ndarray,
        roster: Roster,
    ) -> tuple[list[Any], list[bool]]:
        commands: list[Any] = []
        interrupted: list[bool] = []
        resource_xz = [xz(unit.position()) for unit in roster.resources]
        dropsite_xz = xz(roster.dropsite.position())
        targets = list(self._target_index)
        cycles = list(self._cycle_active)
        for villager in range(self.villager_count):
            base = 3 * villager
            x, z = denormalize_action(action[base : base + 2], self.map_size_m)
            clicked = float(action[base + 2]) > self.click_action_threshold
            interrupted.append(False)
            if not clicked:
                continue
            unit = roster.villagers[villager]
            hit_resource = nearest_index((x, z), resource_xz)
            hits_resource = is_reached(
                distance((x, z), resource_xz[hit_resource]),
                self.gather_command_distance,
            )
            hits_dropsite = is_reached(
                distance((x, z), dropsite_xz),
                self.gather_command_distance,
            )
            if hits_dropsite and not hits_resource:
                commands.append(self._return_resource_command(unit, roster.dropsite))
                continue
            if hits_resource:
                # Re-issuing the same gather order is not an interruption: the
                # villager keeps working the tree it already has. Only pointing
                # it at a different tree throws away work in progress.
                if cycles[villager] and targets[villager] != hit_resource:
                    interrupted[villager] = True
                targets[villager] = hit_resource
                cycles[villager] = True
                commands.append(
                    self._gather_command(unit, roster.resources[hit_resource])
                )
                continue
            if cycles[villager]:
                interrupted[villager] = True
            cycles[villager] = False
            commands.append(self.actions.walk([unit], x, z))
        self._target_index = tuple(targets)
        self._cycle_active = tuple(cycles)
        return commands, interrupted

    def _return_resource_command(self, unit: Any, dropsite: Any) -> Any:
        """Order a deposit, falling back to a raw command for older clients."""

        return_resource = getattr(self.actions, "returnresource", None)
        if callable(return_resource):
            return return_resource([unit], dropsite)
        return {
            "type": "returnresource",
            "entities": [unit.id()],
            "target": dropsite.id(),
            "queued": False,
        }

    def _gather_command(self, unit: Any, resource: Any) -> Any:
        gather = getattr(self.actions, "gather")
        try:
            return gather([unit], resource)
        except TypeError as error:
            try:
                return gather([unit], [resource])
            except TypeError:
                raise error

    def _step_once(self, action):
        if self._roster is None:
            raise RuntimeError("the environment must be reset before stepping")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != 3 * self.villager_count:
            raise ValueError("action must hold three values per villager")
        commands, interrupted = self._commands(values, self._roster)
        payload = commands or None
        step_many = getattr(self.game, "step_many", None)
        if self.sim_frame_observer is not None:
            # Recording wants one frame per simulation turn, so give up the
            # batched fast path and report every turn as it is simulated.
            state = self.game.step(payload)
            self.sim_frame_observer()
            for _ in range(self.sim_steps_per_action - 1):
                state = self.game.step()
                self.sim_frame_observer()
        elif callable(step_many):
            state = step_many(payload, turns=self.sim_steps_per_action)
        else:
            state = self.game.step(payload)
            for _ in range(self.sim_steps_per_action - 1):
                state = self.game.step()

        roster = self._roster_for(state)
        self._roster = roster
        stock, carried, remaining = self._read_engine(roster)
        villager_xz = tuple(xz(unit.position()) for unit in roster.villagers)
        resource_xz = tuple(xz(unit.position()) for unit in roster.resources)
        dropsite_xz = xz(roster.dropsite.position())

        villagers = []
        current_distance = []
        for index in range(self.villager_count):
            carrying = self._previous_carried[index] > 0.0
            target_xz = (
                dropsite_xz if carrying else resource_xz[self._target_index[index]]
            )
            now = distance(villager_xz[index], target_xz)
            current_distance.append(now)
            villagers.append(
                VillagerRewardInputs(
                    distance_closed_m=self._previous_distance[index] - now,
                    carried_resource_delta=(
                        carried[index] - self._previous_carried[index]
                    ),
                    interrupted_gather_cycle=interrupted[index],
                )
            )
        terms = compose_team_reward(
            stock - self._previous_stock,
            tuple(villagers),
            self.reward_scales,
        )

        # A villager's carried load only falls when it deposits, so the drop is
        # that villager's contribution to the team total.
        self._delivered = tuple(
            self._delivered[index]
            + max(0.0, self._previous_carried[index] - carried[index])
            for index in range(self.villager_count)
        )

        # A deposit ends whichever cycles were running, mirroring M1's rule that
        # the cycle finishes when wood actually lands in the player's stock.
        if stock > self._previous_stock:
            self._cycle_active = tuple(False for _ in self._cycle_active)
        self._previous_stock = stock
        self._previous_carried = carried
        self._previous_distance = tuple(current_distance)
        self._step_count += 1

        episode_delta = stock - self._initial_stock
        everyone_delivered = min(self._delivered) >= self.min_delivery_per_villager
        terminated = (
            episode_delta >= self.stock_success_threshold and everyone_delivered
        )
        truncated = not terminated and self._step_count >= self.horizon
        observation = build_team_observation(
            self._snapshot(roster, stock, carried, remaining),
            self.observation_scales,
        )
        info = {
            "resource_stock": stock,
            "resource_stock_delta": terms.stock_delta,
            "episode_resource_stock_delta": episode_delta,
            "distance_shaping_reward": terms.distance_shaping,
            "carried_resource_delta_reward": terms.carried_delta,
            "click_gather_cycle_penalty": terms.click_penalty,
            "commands": len(commands),
            "delivered_per_villager": self._delivered,
            "min_delivered": min(self._delivered),
            "working_villagers": sum(
                1 for delivered in self._delivered if delivered > 0.0
            ),
        }
        return observation, terms.total(), terminated, truncated, info

    def step(self, action):
        try:
            return self._step_once(action)
        except RECOVERABLE_BACKEND_ERRORS:
            if self.backend_retries <= 0:
                raise
            self._recover_backend()
            if self.backend_retry_delay:
                time.sleep(self.backend_retry_delay)
            observation, reset_info = self.reset()
            info = dict(reset_info)
            info["interrupted"] = True
            return observation, 0.0, False, True, info

    def capture_agent_frame(self):
        """Capture the engine-rendered view centred on one team member."""

        state = getattr(self.game, "current_state", None)
        if state is None:
            raise EngineObserverUnavailable(
                "the engine observer needs a current game state before capture"
            )
        roster = self._roster if self._roster is not None else self._roster_for(state)
        unit = roster.villagers[self.observer_villager_slot]
        entity_id = getattr(unit, "id", None)
        if not callable(entity_id):
            raise EngineObserverUnavailable(
                "the zero_ad entity does not expose an engine entity ID"
            )
        return self.engine_observer.capture(entity_id())

    def close(self):
        """Release the backend exactly once."""

        if self._closed:
            return
        self._closed = True
        self._close_backend()
        super().close()
