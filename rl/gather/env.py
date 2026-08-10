"""Gymnasium environment backed by a running 0 A.D. simulation."""

from __future__ import annotations

import importlib
import json
import subprocess
import time
from collections.abc import Mapping
from numbers import Real
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .engine_observer import EngineObserverClient, EngineObserverUnavailable
from .core import (
    GATHER_OBSERVATION_LABELS,
    GATHER_RESOURCE_OBSERVATION_LABELS,
    build_observation,
    denormalize_action,
    distance,
    gather_reward,
    is_reached,
    stock_delta_reward,
    xz,
)

VILLAGER_TYPE = "polites"
RESOURCE_TYPE = "tree"
DROPSITE_TYPE = "storehouse"
REWARD_DISTANCE_DELTA = "distance_delta"
REWARD_STOCK_DELTA = "stock_delta"
REWARD_MODES = frozenset({REWARD_DISTANCE_DELTA, REWARD_STOCK_DELTA})
RECOVERABLE_BACKEND_ERRORS = (ConnectionError, OSError, RuntimeError, TimeoutError)


def player_stock_expression(player_id: int) -> str:
    return (
        "(() => { "
        "const cmpPlayerManager = Engine.QueryInterface(SYSTEM_ENTITY, "
        "IID_PlayerManager); "
        f"const playerEntity = cmpPlayerManager.GetPlayerByID({int(player_id)}); "
        "const cmpPlayer = Engine.QueryInterface(playerEntity, IID_Player); "
        "return cmpPlayer.GetResourceCounts(); "
        "})()"
    )


def unit_carried_resource_expression(entity_id: int, resource: str) -> str:
    resource_literal = json.dumps(resource)
    return (
        "(() => { "
        f"const cmpGatherer = Engine.QueryInterface({int(entity_id)}, "
        "IID_ResourceGatherer); "
        "if (!cmpGatherer) return 0; "
        "let total = 0; "
        "for (const item of cmpGatherer.GetCarryingStatus()) { "
        "const itemType = item.type || item.generic || item.resource || ''; "
        f"if (itemType === {resource_literal} || "
        f"itemType.split('.')[0] === {resource_literal}) "
        "total += +(item.amount || 0); "
        "} "
        "return total; "
        "})()"
    )


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


def _default_backend_factory(uri: str) -> tuple[Any, Any]:
    zero_ad = _load_zero_ad()
    return zero_ad.ZeroAD(uri), zero_ad.actions


def _resolve_backend(
    uri: str,
    game: Any,
    actions: Any,
    backend_factory: Callable[[str], tuple[Any, Any]] | None,
) -> tuple[Any, Any]:
    if game is not None and actions is not None:
        return game, actions

    if backend_factory is None:
        backend_factory = _default_backend_factory
    resolved_game, resolved_actions = backend_factory(uri)
    if game is not None:
        resolved_game = game
    if actions is not None:
        resolved_actions = actions
    return resolved_game, resolved_actions


def _extract_stock(value: object, resource: str) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, Mapping):
        raise TypeError("stock evaluation must return a number or mapping")
    if resource in value:
        return _extract_stock(value[resource], resource)
    for key in ("resourceCounts", "resources", "stock"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and resource in nested:
            return _extract_stock(nested[resource], resource)
    raise KeyError(f"stock evaluation did not include resource '{resource}'")


def _extract_carried_resource(value: object, resource: str) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list):
        total = 0.0
        for item in value:
            if not isinstance(item, Mapping):
                continue
            item_resource = item.get("type", item.get("generic", item.get("resource")))
            specific_resource = (
                item_resource.split(".")[0]
                if isinstance(item_resource, str)
                else item_resource
            )
            if item_resource == resource or specific_resource == resource:
                total += _extract_carried_resource(item.get("amount", 0.0), resource)
        return total
    if isinstance(value, Mapping):
        if "amount" in value and value.get("type", value.get("generic")) == resource:
            return _extract_carried_resource(value["amount"], resource)
        if resource in value:
            return _extract_carried_resource(value[resource], resource)
    raise TypeError("carried resource evaluation must return a number, list, or mapping")


class ZeroADGatherEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_labels = GATHER_OBSERVATION_LABELS

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
        engine_observer=None,
        reward_mode=REWARD_DISTANCE_DELTA,
        stock_resource="wood",
        stock_player=1,
        stock_success_threshold=1.0,
        gather_command_distance=None,
        agent_controls_gather=False,
        gather_action_threshold=0.0,
        agent_controls_click=None,
        click_action_threshold=None,
        resource_state_observation=False,
        carried_resource_observation_scale=20.0,
        stock_observation_scale=1000.0,
        distance_shaping_scale=0.0,
        gather_ready_reward=0.0,
        carried_resource_delta_reward_scale=0.0,
        gather_cycle_no_click_reward=0.0,
        carrying_no_click_reward=0.0,
        click_gather_cycle_penalty=0.0,
        backend_retries=0,
        backend_retry_delay=1.0,
        server_command=None,
        server_startup_delay=2.0,
        backend_factory=None,
    ):
        # reach_threshold=12: el aldeano no puede pisar el arbol (obstaculo solido);
        # se frena a ~9.5m del centro, asi que "llegar" se cuenta a <12m.
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"unknown reward_mode '{reward_mode}'")
        if not isinstance(agent_controls_gather, bool):
            raise ValueError("agent_controls_gather must be a boolean")
        if agent_controls_click is None:
            resolved_agent_controls_click = agent_controls_gather
        elif not isinstance(agent_controls_click, bool):
            raise ValueError("agent_controls_click must be a boolean")
        else:
            resolved_agent_controls_click = agent_controls_click
        resolved_click_action_threshold = (
            gather_action_threshold
            if click_action_threshold is None
            else click_action_threshold
        )
        if (
            isinstance(resolved_click_action_threshold, bool)
            or not isinstance(resolved_click_action_threshold, Real)
            or not np.isfinite(float(resolved_click_action_threshold))
            or float(resolved_click_action_threshold) < -1.0
            or float(resolved_click_action_threshold) > 1.0
        ):
            raise ValueError("click_action_threshold must be finite in [-1, 1]")
        if not isinstance(resource_state_observation, bool):
            raise ValueError("resource_state_observation must be a boolean")
        if (
            isinstance(carried_resource_observation_scale, bool)
            or not isinstance(carried_resource_observation_scale, Real)
            or not np.isfinite(float(carried_resource_observation_scale))
            or float(carried_resource_observation_scale) <= 0.0
        ):
            raise ValueError(
                "carried_resource_observation_scale must be finite and positive"
            )
        if (
            isinstance(stock_observation_scale, bool)
            or not isinstance(stock_observation_scale, Real)
            or not np.isfinite(float(stock_observation_scale))
            or float(stock_observation_scale) <= 0.0
        ):
            raise ValueError("stock_observation_scale must be finite and positive")
        if (
            isinstance(distance_shaping_scale, bool)
            or not isinstance(distance_shaping_scale, Real)
            or not np.isfinite(float(distance_shaping_scale))
            or float(distance_shaping_scale) < 0.0
        ):
            raise ValueError("distance_shaping_scale must be finite and non-negative")
        if (
            isinstance(gather_ready_reward, bool)
            or not isinstance(gather_ready_reward, Real)
            or not np.isfinite(float(gather_ready_reward))
            or float(gather_ready_reward) < 0.0
        ):
            raise ValueError("gather_ready_reward must be finite and non-negative")
        if (
            isinstance(carried_resource_delta_reward_scale, bool)
            or not isinstance(carried_resource_delta_reward_scale, Real)
            or not np.isfinite(float(carried_resource_delta_reward_scale))
            or float(carried_resource_delta_reward_scale) < 0.0
        ):
            raise ValueError(
                "carried_resource_delta_reward_scale must be finite and non-negative"
            )
        if (
            isinstance(gather_cycle_no_click_reward, bool)
            or not isinstance(gather_cycle_no_click_reward, Real)
            or not np.isfinite(float(gather_cycle_no_click_reward))
            or float(gather_cycle_no_click_reward) < 0.0
        ):
            raise ValueError(
                "gather_cycle_no_click_reward must be finite and non-negative"
            )
        if (
            isinstance(carrying_no_click_reward, bool)
            or not isinstance(carrying_no_click_reward, Real)
            or not np.isfinite(float(carrying_no_click_reward))
            or float(carrying_no_click_reward) < 0.0
        ):
            raise ValueError("carrying_no_click_reward must be finite and non-negative")
        if (
            isinstance(click_gather_cycle_penalty, bool)
            or not isinstance(click_gather_cycle_penalty, Real)
            or not np.isfinite(float(click_gather_cycle_penalty))
            or float(click_gather_cycle_penalty) < 0.0
        ):
            raise ValueError("click_gather_cycle_penalty must be finite and non-negative")
        self.uri = uri
        self._backend_factory = backend_factory
        self._uses_injected_backend = game is not None and actions is not None
        self.server_command = (
            None
            if server_command is None
            else tuple(str(part) for part in server_command)
        )
        self.server_startup_delay = server_startup_delay
        self._server_process = None
        if self.server_command is not None:
            self._start_server()
        self.game, self.actions = _resolve_backend(
            uri,
            game,
            actions,
            backend_factory,
        )
        self.engine_observer = (
            engine_observer
            if engine_observer is not None
            else EngineObserverClient(uri)
        )
        self.scenario_config = scenario_config
        self.save_replay = save_replay
        self.map_size_m = map_size_m
        self.horizon = horizon
        self.reach_threshold = reach_threshold
        self.gather_command_distance = (
            reach_threshold
            if gather_command_distance is None
            else gather_command_distance
        )
        self.sim_steps_per_action = sim_steps_per_action
        self.reward_mode = reward_mode
        self.stock_resource = stock_resource
        self.stock_player = stock_player
        self.stock_success_threshold = stock_success_threshold
        self.agent_controls_gather = agent_controls_gather
        self.agent_controls_click = resolved_agent_controls_click
        self.click_action_threshold = float(resolved_click_action_threshold)
        self.resource_state_observation = resource_state_observation
        self.carried_resource_observation_scale = float(
            carried_resource_observation_scale
        )
        self.stock_observation_scale = float(stock_observation_scale)
        self.distance_shaping_scale = float(distance_shaping_scale)
        self.gather_ready_reward = float(gather_ready_reward)
        self.carried_resource_delta_reward_scale = float(
            carried_resource_delta_reward_scale
        )
        self.gather_cycle_no_click_reward = float(gather_cycle_no_click_reward)
        self.carrying_no_click_reward = float(carrying_no_click_reward)
        self.click_gather_cycle_penalty = float(click_gather_cycle_penalty)
        self.backend_retries = backend_retries
        self.backend_retry_delay = backend_retry_delay
        self.observation_labels = (
            GATHER_RESOURCE_OBSERVATION_LABELS
            if self.resource_state_observation
            else GATHER_OBSERVATION_LABELS
        )
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(len(self.observation_labels),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(3 if self.agent_controls_click else 2,),
            dtype=np.float32,
        )
        self._step_count = 0
        self._prev_dist = None
        self._prev_stock = 0.0
        self._initial_stock = 0.0
        self._prev_carried_resource = 0.0
        self._gather_cycle_active = False
        self._gather_cycle_elapsed = 0
        self._closed = False

    def _positions(self, state):
        v = xz(state.units(owner=1, entity_type=VILLAGER_TYPE)[0].position())
        r = xz(state.units(owner=0, entity_type=RESOURCE_TYPE)[0].position())
        return v, r

    def _entities(self, state):
        villager = state.units(owner=1, entity_type=VILLAGER_TYPE)[0]
        resource = state.units(owner=0, entity_type=RESOURCE_TYPE)[0]
        return villager, resource

    def _dropsite(self, state):
        dropsites = state.units(owner=1, entity_type=DROPSITE_TYPE)
        return dropsites[0] if dropsites else None

    def _resource_stock(self) -> float:
        evaluate = getattr(self.game, "evaluate", None)
        if not callable(evaluate):
            raise RuntimeError("stock_delta reward requires game.evaluate(js)")
        return _extract_stock(
            evaluate(player_stock_expression(self.stock_player)),
            self.stock_resource,
        )

    def _carried_resource(self, villager) -> float:
        evaluate = getattr(self.game, "evaluate", None)
        entity_id = getattr(villager, "id", None)
        if not callable(evaluate) or not callable(entity_id):
            raise RuntimeError(
                "resource state observation requires game.evaluate(js) and unit.id()"
            )
        return _extract_carried_resource(
            evaluate(
                unit_carried_resource_expression(
                    entity_id(),
                    self.stock_resource,
                )
            ),
            self.stock_resource,
        )

    def _tracks_carried_resource(self) -> bool:
        return bool(
            self.resource_state_observation
            or self.carried_resource_delta_reward_scale
            or self.carrying_no_click_reward
        )

    def _observation(
        self,
        villager_xz,
        resource_xz,
        *,
        villager=None,
        carried_resource=None,
        resource_stock=None,
    ):
        if not self.resource_state_observation:
            return build_observation(villager_xz, resource_xz, self.map_size_m)
        if villager is None:
            villager = self._entities(self.game.current_state)[0]
        if carried_resource is None:
            carried_resource = self._carried_resource(villager)
        if resource_stock is None:
            resource_stock = self._resource_stock()
        return build_observation(
            villager_xz,
            resource_xz,
            self.map_size_m,
            carried_resource=carried_resource,
            carried_resource_scale=self.carried_resource_observation_scale,
            resource_stock=resource_stock,
            resource_stock_scale=self.stock_observation_scale,
        )

    def _start_gather_cycle(self) -> None:
        self._gather_cycle_active = True
        self._gather_cycle_elapsed = 0

    def _finish_gather_cycle(self) -> None:
        self._gather_cycle_active = False
        self._gather_cycle_elapsed = 0

    def _start_server(self) -> None:
        if self.server_command is None:
            return
        if self._server_process is not None and self._server_process.poll() is None:
            return
        self._server_process = subprocess.Popen(self.server_command)
        if self.server_startup_delay:
            time.sleep(self.server_startup_delay)

    def _stop_server(self) -> None:
        if self._server_process is None:
            return
        if self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait(timeout=5)
        self._server_process = None

    def _close_backend(self) -> None:
        close_backend = getattr(self.game, "close", None)
        if callable(close_backend):
            close_backend()

    def _reconnect_backend(self) -> None:
        if self._uses_injected_backend and self._backend_factory is None:
            return
        self._close_backend()
        self.game, self.actions = _resolve_backend(
            self.uri,
            None,
            None,
            self._backend_factory,
        )

    def _recover_backend(self) -> None:
        self._close_backend()
        if self.server_command is not None:
            self._stop_server()
            self._start_server()
        self._reconnect_backend()

    def _reset_once(self, *, seed=None):
        super().reset(seed=seed)
        self.game.reset(self.scenario_config, save_replay=self.save_replay)
        state = self.game.step()  # un tick para que las entidades existan
        v, r = self._positions(state)
        villager = self._entities(state)[0]
        self._prev_dist = distance(v, r)
        self._step_count = 0
        self._finish_gather_cycle()
        self._prev_carried_resource = 0.0
        carried_resource = None
        if self._tracks_carried_resource():
            carried_resource = self._carried_resource(villager)
            self._prev_carried_resource = carried_resource
        if self.reward_mode == REWARD_STOCK_DELTA:
            self._initial_stock = self._resource_stock()
            self._prev_stock = self._initial_stock
            info = {
                "distance": self._prev_dist,
                "resource_stock": self._prev_stock,
                "episode_resource_stock_delta": 0.0,
                "reward_mode": self.reward_mode,
            }
        else:
            info = {}
        return (
            self._observation(
                v,
                r,
                villager=villager,
                carried_resource=carried_resource,
                resource_stock=(
                    self._prev_stock
                    if self.reward_mode == REWARD_STOCK_DELTA
                    else None
                ),
            ),
            info,
        )

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

    def _command_for_action(self, action):
        action_values = np.asarray(action).reshape(-1)
        x, z = denormalize_action(action_values, self.map_size_m)
        state = self.game.current_state
        villager, resource = self._entities(state)
        _v, r = self._positions(state)
        dropsite = self._dropsite(state)
        dropsite_xz = None if dropsite is None else xz(dropsite.position())
        click_signal = (
            float(action_values[2])
            if self.agent_controls_click
            else None
        )
        click_requested = (
            not self.agent_controls_click
            or click_signal > self.click_action_threshold
        )
        target_hits_resource = is_reached(
            distance((x, z), r),
            self.gather_command_distance,
        )
        target_hits_dropsite = (
            dropsite_xz is not None
            and is_reached(
                distance((x, z), dropsite_xz),
                self.gather_command_distance,
            )
        )
        gather_requested = click_requested and target_hits_resource
        return_resource_requested = click_requested and target_hits_dropsite
        command_info = {
            "click_requested": click_requested,
            "gather_requested": gather_requested,
            "return_resource_requested": return_resource_requested,
            "target_hits_resource": target_hits_resource,
            "target_hits_dropsite": target_hits_dropsite,
        }
        if click_signal is not None:
            command_info["click_signal"] = click_signal
        if self._gather_cycle_active:
            command_info["gather_cycle_active"] = True
            command_info["gather_cycle_elapsed"] = self._gather_cycle_elapsed
        if not click_requested:
            return None, "no_click", command_info
        if (
            self.reward_mode == REWARD_STOCK_DELTA
            and return_resource_requested
            and dropsite is not None
        ):
            command_info["gather_cycle_active"] = self._gather_cycle_active
            command_info["gather_cycle_elapsed"] = self._gather_cycle_elapsed
            return (
                self._return_resource_command(villager, dropsite),
                "return_resource",
                command_info,
            )
        if self._gather_cycle_active:
            command_info["click_while_gather_cycle_active"] = True
            self._finish_gather_cycle()
            command_info["gather_cycle_active"] = False
            command_info["gather_cycle_finished"] = "interrupted_by_click"
        if (
            self.reward_mode == REWARD_STOCK_DELTA
            and gather_requested
        ):
            self._start_gather_cycle()
            command_info["gather_cycle_active"] = True
            command_info["gather_cycle_elapsed"] = self._gather_cycle_elapsed
            return (
                self._gather_command(villager, resource),
                "gather",
                command_info,
            )
        return (
            self.actions.walk([villager], x, z),
            "walk",
            command_info,
        )

    def _gather_command(self, villager, resource):
        gather = getattr(self.actions, "gather")
        try:
            return gather([villager], resource)
        except TypeError as error:
            try:
                return gather([villager], [resource])
            except TypeError:
                raise error

    def _return_resource_command(self, villager, dropsite):
        return_resource = getattr(self.actions, "returnresource", None)
        if callable(return_resource):
            return return_resource([villager], dropsite)
        return {
            "type": "returnresource",
            "entities": [villager.id()],
            "target": dropsite.id(),
            "queued": False,
        }

    def _step_once(self, action):
        cmd, command_name, command_info = self._command_for_action(action)
        state = self.game.step() if cmd is None else self.game.step([cmd])
        for _ in range(self.sim_steps_per_action - 1):
            state = self.game.step()
        v, r = self._positions(state)
        villager = self._entities(state)[0]
        cur_dist = distance(v, r)
        info = {
            "distance": cur_dist,
            "command": command_name,
            "reward_mode": self.reward_mode,
        }
        if self.reward_mode == REWARD_STOCK_DELTA:
            cur_stock = self._resource_stock()
            stock_reward = stock_delta_reward(self._prev_stock, cur_stock)
            self._prev_stock = cur_stock
            cur_carried_resource = None
            carried_resource_delta = 0.0
            if self._tracks_carried_resource():
                cur_carried_resource = self._carried_resource(villager)
                carried_resource_delta = max(
                    0.0,
                    cur_carried_resource - self._prev_carried_resource,
                )
                self._prev_carried_resource = cur_carried_resource
            episode_stock_delta = cur_stock - self._initial_stock
            terminated = episode_stock_delta >= self.stock_success_threshold
            if self._gather_cycle_active and stock_reward > 0.0:
                self._finish_gather_cycle()
                command_info["gather_cycle_active"] = False
                command_info.pop("gather_cycle_elapsed", None)
                command_info["gather_cycle_finished"] = "stock_delta"
            elif self._gather_cycle_active:
                self._gather_cycle_elapsed += 1
                command_info["gather_cycle_active"] = True
                command_info["gather_cycle_elapsed"] = self._gather_cycle_elapsed
            distance_shaping_reward = (
                self.distance_shaping_scale
                * gather_reward(self._prev_dist, cur_dist)
            )
            ready_reward = (
                self.gather_ready_reward
                if is_reached(cur_dist, self.gather_command_distance)
                else 0.0
            )
            carried_resource_reward = (
                self.carried_resource_delta_reward_scale
                * carried_resource_delta
            )
            no_click_cycle_reward = (
                self.gather_cycle_no_click_reward
                if (
                    command_name == "no_click"
                    and command_info.get("gather_cycle_active", False)
                )
                else 0.0
            )
            no_click_carrying_reward = (
                self.carrying_no_click_reward
                if (
                    command_name == "no_click"
                    and cur_carried_resource is not None
                    and cur_carried_resource > 0.0
                )
                else 0.0
            )
            click_cycle_penalty = (
                self.click_gather_cycle_penalty
                if command_info.get("click_while_gather_cycle_active", False)
                else 0.0
            )
            reward = (
                stock_reward
                + distance_shaping_reward
                + ready_reward
                + carried_resource_reward
                + no_click_cycle_reward
                + no_click_carrying_reward
                - click_cycle_penalty
            )
            info.update(
                {
                    "resource_stock": cur_stock,
                    "resource_stock_delta": stock_reward,
                    "episode_resource_stock_delta": episode_stock_delta,
                    **command_info,
                }
            )
            if self.distance_shaping_scale:
                info["distance_shaping_reward"] = distance_shaping_reward
            if self.gather_ready_reward:
                info["gather_ready_reward"] = ready_reward
            if cur_carried_resource is not None:
                info["carried_resource"] = cur_carried_resource
                info["carried_resource_delta"] = carried_resource_delta
            if self.carried_resource_delta_reward_scale:
                info["carried_resource_delta_reward"] = carried_resource_reward
            if self.gather_cycle_no_click_reward:
                info["gather_cycle_no_click_reward"] = no_click_cycle_reward
            if self.carrying_no_click_reward:
                info["carrying_no_click_reward"] = no_click_carrying_reward
            if self.click_gather_cycle_penalty:
                info["click_gather_cycle_penalty"] = click_cycle_penalty
        else:
            reward = gather_reward(self._prev_dist, cur_dist)
            terminated = is_reached(cur_dist, self.reach_threshold)
        self._prev_dist = cur_dist
        self._step_count += 1
        truncated = self._step_count >= self.horizon
        obs = self._observation(
            v,
            r,
            villager=villager,
            carried_resource=(
                cur_carried_resource
                if self.reward_mode == REWARD_STOCK_DELTA
                and self._tracks_carried_resource()
                else None
            ),
            resource_stock=(
                self._prev_stock if self.reward_mode == REWARD_STOCK_DELTA else None
            ),
        )
        return obs, reward, terminated, truncated, info

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
            info.update(
                {
                    "interrupted": True,
                    "reward_mode": self.reward_mode,
                }
            )
            return observation, 0.0, False, True, info

    def capture_agent_frame(self):
        """Capture the current pre-action Polites view from the patched engine."""

        state = getattr(self.game, "current_state", None)
        if state is None:
            raise EngineObserverUnavailable(
                "the engine observer needs a current game state before capture"
            )
        villagers = state.units(owner=1, entity_type=VILLAGER_TYPE)
        if len(villagers) != 1:
            raise EngineObserverUnavailable(
                "the gather observer requires exactly one current Polites"
            )
        entity_id = getattr(villagers[0], "id", None)
        if not callable(entity_id):
            raise EngineObserverUnavailable(
                "the zero_ad entity does not expose an engine entity ID"
            )
        return self.engine_observer.capture(entity_id())

    def close(self):
        """Release the optional live/injected backend exactly once."""

        if self._closed:
            return
        self._closed = True
        self._close_backend()
        self._stop_server()
        super().close()
