"""Gymnasium environment for a team of villagers gathering one resource."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .assignment_actions import (
    ACTION_MODES,
    ASSIGNMENT_CLICK_MODE,
    JOINT_ASSIGNMENT_CLICK_MODE,
    RAW_CLICK_MODE,
    assignment_action_space,
    assignment_to_raw_click,
    joint_assignment_action_space,
    joint_assignment_from_index,
)
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
from .roster import Roster, build_roster, refresh_roster
from .team_render import TeamRenderState, render_team_observer_frame


class ZeroADTeamGatherEnv(gym.Env):
    """Drive ``villager_count`` villagers with one action vector."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_config,
        *,
        villager_count: int = 4,
        resource_count: int = 4,
        action_mode: str = RAW_CLICK_MODE,
        randomize_layout: bool = False,
        uri: str = "http://localhost:6000",
        map_size_m: float = 512.0,
        horizon: int = 120,
        sim_steps_per_action: int = 80,
        gather_command_distance: float = 12.0,
        click_action_threshold: float = 0.0,
        stock_resource: str = "wood",
        stock_player: int = 1,
        stock_success_threshold: float = 80.0,
        carried_resource_observation_scale: float = 20.0,
        stock_observation_scale: float = 1000.0,
        resource_amount_scale: float = 200.0,
        distance_shaping_scale: float = 0.02,
        carried_resource_delta_reward_scale: float = 0.2,
        click_gather_cycle_penalty: float = 1.0,
        carrying_no_click_reward: float = 0.0,
        backend_retries: int = 0,
        backend_retry_delay: float = 1.0,
        save_replay: bool = False,
        observer_view: str = "team",
        observer_view_margin_m: float = 12.0,
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
        if action_mode not in ACTION_MODES:
            raise ValueError(
                "action_mode must be 'raw_click', 'assignment_click', or "
                "'joint_assignment_click'"
            )
        if not isinstance(randomize_layout, bool):
            raise ValueError("randomize_layout must be a boolean")
        if not 0 <= observer_villager_slot < villager_count:
            raise ValueError("observer_villager_slot must address a configured villager")
        if observer_view not in {"team", "villager"}:
            raise ValueError("observer_view must be 'team' or 'villager'")
        if not observer_view_margin_m >= 0.0:
            raise ValueError("observer_view_margin_m must be non-negative")
        if sim_frame_observer is not None and not callable(sim_frame_observer):
            raise ValueError("sim_frame_observer must be callable or None")
        self.observer_view = observer_view
        self.observer_view_margin_m = observer_view_margin_m
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
        self.randomize_layout = randomize_layout
        self._scenario_template: dict[str, Any] | None = None
        if randomize_layout:
            try:
                parsed_scenario = json.loads(scenario_config)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "randomize_layout requires a JSON scenario configuration"
                ) from error
            if (
                not isinstance(parsed_scenario, dict)
                or not isinstance(parsed_scenario.get("settings"), dict)
            ):
                raise ValueError(
                    "randomize_layout requires scenario settings as an object"
                )
            self._scenario_template = parsed_scenario
        self._layout_seed: int | None = None
        self.save_replay = save_replay
        self.villager_count = villager_count
        self.resource_count = resource_count
        self.action_mode = action_mode
        self.map_size_m = map_size_m
        self.horizon = horizon
        self.sim_steps_per_action = sim_steps_per_action
        self.gather_command_distance = gather_command_distance
        self.click_action_threshold = click_action_threshold
        self.stock_resource = stock_resource
        self.stock_player = stock_player
        self.stock_success_threshold = stock_success_threshold
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
            carrying_no_click_reward=carrying_no_click_reward,
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
        if action_mode == ASSIGNMENT_CLICK_MODE:
            self.action_space = assignment_action_space(
                villager_count=villager_count,
                resource_count=resource_count,
            )
        elif action_mode == JOINT_ASSIGNMENT_CLICK_MODE:
            self.action_space = joint_assignment_action_space(
                villager_count=villager_count,
                resource_count=resource_count,
            )
        else:
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
        self._previous_villager_xz: tuple[tuple[float, float], ...] = ()
        self._target_index: tuple[int | None, ...] = ()
        self._cycle_active: tuple[bool, ...] = ()
        self._delivered: tuple[float, ...] = ()
        self._last_remaining: tuple[float, ...] = ()
        self._closed = False

    # ------------------------------------------------------------------ engine

    def _roster_for(self, state: Any) -> Roster:
        if self._roster is not None:
            return refresh_roster(
                state,
                previous=self._roster,
                villager_count=self.villager_count,
                resource_count=self.resource_count,
                villager_type=VILLAGER_TYPE,
                resource_type=RESOURCE_TYPE,
                dropsite_type=DROPSITE_TYPE,
            )
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

    def _scenario_for_reset(self) -> str:
        """Return an immutable per-episode scenario with a fresh map seed."""

        if not self.randomize_layout:
            self._layout_seed = None
            return self.scenario_config
        if self._scenario_template is None:
            raise RuntimeError("missing randomized scenario template")
        layout_seed = int(self.np_random.integers(0, 2**31))
        settings = self._scenario_template["settings"]
        scenario = {
            **self._scenario_template,
            "settings": {
                **settings,
                "Seed": layout_seed,
                "AISeed": layout_seed,
            },
        }
        self._layout_seed = layout_seed
        return json.dumps(scenario, separators=(",", ":"), sort_keys=True)

    def _reset_once(self, *, seed=None):
        super().reset(seed=seed)
        # Resource ids belong to one generated map. Never carry a depleted-slot
        # cache into the next reset, even if the engine happens to reuse ids.
        self._roster = None
        state = self.game.reset(
            self._scenario_for_reset(),
            save_replay=self.save_replay,
        )
        if state is None:
            state = self.game.step()
        roster = self._roster_for(state)
        self._roster = roster
        villager_xz = tuple(xz(unit.position()) for unit in roster.villagers)
        self._target_index = tuple(None for _ in roster.villagers)
        self._cycle_active = tuple(False for _ in roster.villagers)
        self._delivered = tuple(0.0 for _ in roster.villagers)
        stock, carried, remaining = self._read_engine(roster)
        self._initial_stock = stock
        self._previous_stock = stock
        self._last_remaining = remaining
        self._previous_carried = carried
        self._previous_villager_xz = villager_xz
        self._step_count = 0
        observation = build_team_observation(
            self._snapshot(roster, stock, carried, remaining),
            self.observation_scales,
        )
        return observation, {
            "resource_stock": stock,
            "episode_resource_stock_delta": 0.0,
            "layout_seed": self._layout_seed,
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
    ) -> tuple[list[Any], list[bool], list[bool]]:
        commands: list[Any] = []
        disrupted: list[bool] = []
        commanded: list[bool] = []
        resource_xz = [xz(unit.position()) for unit in roster.resources]
        dropsite_xz = xz(roster.dropsite.position())
        targets = list(self._target_index)
        cycles = list(self._cycle_active)
        threshold = (
            0.0
            if self.action_mode
            in {ASSIGNMENT_CLICK_MODE, JOINT_ASSIGNMENT_CLICK_MODE}
            else self.click_action_threshold
        )
        for villager in range(self.villager_count):
            target = targets[villager]
            # A destroyed tree remains a fixed action/observation slot, but it
            # is no longer an active command target. Clearing it makes an idle
            # worker available for another live tree on the next decision.
            if target is not None and self._last_remaining[target] <= 0.0:
                targets[villager] = None
                cycles[villager] = False
            base = 3 * villager
            x, z = denormalize_action(action[base : base + 2], self.map_size_m)
            clicked = float(action[base + 2]) > threshold
            # Busy means the engine is already doing productive work for this
            # villager: chopping, or hauling a load back on its own. Tracking
            # only the gather cycle let the first stray walk clear the flag and
            # made every later command free, so a villager could hold a full
            # load for the whole episode at no cost.
            holding = self._previous_carried[villager] > 0.0
            busy = cycles[villager] or holding
            disrupted.append(False)
            commanded.append(clicked)
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
                # Ordering the deposit a loaded villager is already making
                # repeats its own work; ordering it while chopping throws the
                # unfinished load away.
                if busy and not holding:
                    disrupted[villager] = True
                targets[villager] = None
                cycles[villager] = False
                commands.append(self._return_resource_command(unit, roster.dropsite))
                continue
            if hits_resource:
                if self._last_remaining[hit_resource] <= 0.0:
                    # A categorical action can still name a dead fixed slot.
                    # Do not turn that stale coordinate into a walk or send a
                    # gather order for an entity 0 A.D. has already destroyed.
                    commanded[-1] = False
                    continue
                # Re-issuing the same gather order is not a disruption: the
                # villager keeps working the tree it already has. Only pointing
                # it at a different tree throws away work in progress.
                repeat = (
                    cycles[villager]
                    and not holding
                    and targets[villager] == hit_resource
                )
                if busy and not repeat:
                    disrupted[villager] = True
                targets[villager] = hit_resource
                cycles[villager] = True
                commands.append(
                    self._gather_command(unit, roster.resources[hit_resource])
                )
                continue
            if busy:
                disrupted[villager] = True
            targets[villager] = None
            cycles[villager] = False
            commands.append(self.actions.walk([unit], x, z))
        self._target_index = tuple(targets)
        self._cycle_active = tuple(cycles)
        return commands, disrupted, commanded

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

    def _decode_action(self, action: object, roster: Roster) -> np.ndarray:
        if self.action_mode == ASSIGNMENT_CLICK_MODE:
            assignments = np.asarray(action)
            if assignments.shape != (self.villager_count,):
                raise ValueError(
                    "assignment action must contain one category per villager"
                )
        elif self.action_mode == JOINT_ASSIGNMENT_CLICK_MODE:
            assignments = joint_assignment_from_index(
                action,
                villager_count=self.villager_count,
                resource_count=self.resource_count,
            )
            assignments = self._joint_assignments_for_execution(assignments)
        else:
            values = np.asarray(action, dtype=np.float32).reshape(-1)
            if values.size != 3 * self.villager_count:
                raise ValueError("action must hold three values per villager")
            return values

        return assignment_to_raw_click(
            assignments,
            tuple(xz(unit.position()) for unit in roster.resources),
            map_size_m=self.map_size_m,
        )

    def _joint_assignments_for_execution(
        self,
        assignments: np.ndarray,
    ) -> np.ndarray:
        """Remove state-invalid high-level commands before map-click decoding.

        The categorical table guarantees a fresh command bundle has no duplicate
        tree. This second, observed-state check keeps chopping targets reserved
        for their current owner and turns vanished fixed slots into no-ops. It
        mirrors the policy mask so a caller cannot bypass the M2 capacity
        contract; choosing to interrupt a worker remains a learned action.
        """

        executable = assignments.copy()
        reserved_owners: dict[int, int] = {}
        for villager, target in enumerate(self._target_index):
            carrying = self._previous_carried[villager] > 0.0
            active = self._cycle_active[villager]
            if (
                active
                and not carrying
                and target is not None
                and self._last_remaining[target] > 0.0
            ):
                reserved_owners[target + 1] = villager
        for villager, category in enumerate(executable):
            if category == 0:
                continue
            resource = int(category) - 1
            if (
                (
                    int(category) in reserved_owners
                    and reserved_owners[int(category)] != villager
                )
                or self._last_remaining[resource] <= 0.0
            ):
                executable[villager] = 0
        return executable

    def _step_once(self, action):
        if self._roster is None:
            raise RuntimeError("the environment must be reset before stepping")
        values = self._decode_action(action, self._roster)
        previous_villager_xz = self._previous_villager_xz
        previous_carried = self._previous_carried
        commands, disrupted, commanded = self._commands(values, self._roster)
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
        for index in range(self.villager_count):
            carrying = previous_carried[index] > 0.0
            target_index = self._target_index[index]
            target_xz = dropsite_xz if carrying else (
                None if target_index is None else resource_xz[target_index]
            )
            # Both ends of the step are measured against the same point. Storing
            # last step's distance instead compared it against last step's
            # target, so switching targets teleported the reference point and
            # paid the move as if it were progress.
            closed = 0.0 if target_xz is None else (
                distance(previous_villager_xz[index], target_xz)
                - distance(villager_xz[index], target_xz)
            )
            villagers.append(
                VillagerRewardInputs(
                    distance_closed_m=closed,
                    carried_resource_delta=(carried[index] - previous_carried[index]),
                    disrupted_while_busy=disrupted[index],
                    carrying_without_command=carrying and not commanded[index],
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

        # A deposit ends that villager's cycle, mirroring M1's rule that the
        # cycle finishes when wood lands in the player's stock. Only the villager
        # whose load actually dropped is finished: clearing the whole team's
        # flags let one deposit erase three other villagers' work in progress
        # from both the reward and the observation.
        self._cycle_active = tuple(
            False
            if (
                previous_carried[index] > carried[index]
                or (
                    target is not None
                    and remaining[target] <= 0.0
                )
            )
            else active
            for index, (active, target) in enumerate(
                zip(self._cycle_active, self._target_index, strict=True)
            )
        )
        self._target_index = tuple(
            None
            if (
                previous_carried[index] > carried[index]
                or (target is not None and remaining[target] <= 0.0)
            )
            else target
            for index, target in enumerate(self._target_index)
        )
        self._previous_stock = stock
        self._previous_carried = carried
        self._last_remaining = remaining
        self._previous_villager_xz = villager_xz
        self._step_count += 1

        episode_delta = stock - self._initial_stock
        terminated = episode_delta >= self.stock_success_threshold
        truncated = not terminated and self._step_count >= self.horizon
        observation = build_team_observation(
            self._snapshot(roster, stock, carried, remaining),
            self.observation_scales,
        )
        info = {
            "resource_stock": stock,
            "resource_stock_delta": terms.stock_delta,
            "episode_resource_stock_delta": episode_delta,
            "layout_seed": self._layout_seed,
            "distance_shaping_reward": terms.distance_shaping,
            "carried_resource_delta_reward": terms.carried_delta,
            "click_gather_cycle_penalty": terms.click_penalty,
            "carrying_no_click_reward": terms.carrying_no_click,
            "commands": len(commands),
            "disrupted_villagers": sum(1 for value in disrupted if value),
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
        """Capture the engine-rendered view of the team or of one member.

        The team view uses the dropsite to select the owning player's line of
        sight, then centres and widens the camera around every villager,
        resource, and the dropsite itself.
        """

        state = getattr(self.game, "current_state", None)
        if state is None:
            raise EngineObserverUnavailable(
                "the engine observer needs a current game state before capture"
            )
        roster = self._roster if self._roster is not None else self._roster_for(state)
        if self.observer_view == "villager":
            unit = roster.villagers[self.observer_villager_slot]
            entity_id = getattr(unit, "id", None)
            if not callable(entity_id):
                raise EngineObserverUnavailable(
                    "the zero_ad entity does not expose an engine entity ID"
                )
            return self.engine_observer.capture(entity_id())
        if roster.dropsite is None:
            raise EngineObserverUnavailable("the team view needs a storehouse")
        entity_id = getattr(roster.dropsite, "id", None)
        if not callable(entity_id):
            raise EngineObserverUnavailable(
                "the zero_ad entity does not expose an engine entity ID"
            )
        positions = [
            xz(unit.position())
            for unit in (*roster.villagers, *roster.resources, roster.dropsite)
        ]
        min_x = min(x for x, _z in positions)
        max_x = max(x for x, _z in positions)
        min_z = min(z for _x, z in positions)
        max_z = max(z for _x, z in positions)
        centre = ((min_x + max_x) / 2.0, (min_z + max_z) / 2.0)
        span = max(max_x - min_x, max_z - min_z) / 2.0
        return self.engine_observer.capture(
            entity_id(),
            view_range_m=span + self.observer_view_margin_m,
            focus_xz=centre,
        )

    def capture_schematic_frame(self, action: object):
        """Render the whole current team when the engine frame is unavailable."""

        if self._roster is None or self._roster.dropsite is None:
            raise RuntimeError("the environment must be reset before rendering")
        raw_action = self._decode_action(action, self._roster)
        threshold = (
            0.0
            if self.action_mode
            in {ASSIGNMENT_CLICK_MODE, JOINT_ASSIGNMENT_CLICK_MODE}
            else self.click_action_threshold
        )
        targets = tuple(
            denormalize_action(raw_action[base : base + 2], self.map_size_m)
            for base in range(0, raw_action.size, 3)
            if float(raw_action[base + 2]) > threshold
        )
        return render_team_observer_frame(
            TeamRenderState(
                villager_xz=tuple(
                    xz(unit.position()) for unit in self._roster.villagers
                ),
                resource_xz=tuple(
                    xz(unit.position()) for unit in self._roster.resources
                ),
                resource_remaining=self._last_remaining,
                carried=self._previous_carried,
                dropsite_xz=xz(self._roster.dropsite.position()),
                targets_xz=targets,
            )
        )

    def close(self):
        """Release the backend exactly once."""

        if self._closed:
            return
        self._closed = True
        self._close_backend()
        super().close()
