"""Tk window for engine-rendered Polites LOS and the M0 policy input."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from rl.experiments.evaluation import DecisionObserver, DecisionRecord

from .core import (
    GATHER_OBSERVATION_LABELS,
    POLITES_VISION_RADIUS_M,
    denormalize_action,
)
from .engine_observer import EngineObserverError, EngineObserverFrame


class AgentViewUnavailable(RuntimeError):
    """Raised when the requested observation window cannot be created."""


class AgentView(Protocol):
    """Runtime contract used by the evaluation CLI."""

    def update(self, record: DecisionRecord) -> None: ...

    def pause(self, delay: float) -> None: ...

    def close(self) -> None: ...


def make_agent_view_observer(
    agent_view: AgentView,
    *,
    delay: float,
) -> DecisionObserver:
    """Update the local view before advancing the environment."""

    def observe(record: DecisionRecord) -> None:
        agent_view.update(record)
        if delay:
            agent_view.pause(delay)

    return observe


@dataclass(frozen=True, slots=True)
class LocalObservationScene:
    """Physical vision plus the omniscient M0 policy input used to derive it."""

    polites_world_m: tuple[float, float]
    resource_world_m: tuple[float, float]
    resource_local_m: tuple[float, float]
    observed_distance_m: float
    physical_distance_m: float
    vision_radius_m: float
    visible_resource_local_m: tuple[float, float] | None
    normalized_observation: tuple[float, ...]


def _validated_map_size(map_size_m: float) -> float:
    parsed = float(map_size_m)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("map_size_m must be finite and positive")
    return parsed


def _validated_vision_radius(vision_radius_m: float) -> float:
    parsed = float(vision_radius_m)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("vision_radius_m must be finite and positive")
    return parsed


def project_local_observation(
    observation: object,
    map_size_m: float,
    vision_radius_m: float = POLITES_VISION_RADIUS_M,
) -> LocalObservationScene:
    """Separate physical Polites vision from the omniscient M0 observation."""

    values = np.asarray(observation).reshape(-1)
    if len(values) != len(GATHER_OBSERVATION_LABELS):
        raise ValueError("gather agent view requires exactly five values")
    if not np.isfinite(values).all():
        raise ValueError("gather agent view values must be finite")

    size = _validated_map_size(map_size_m)
    vision_radius = _validated_vision_radius(vision_radius_m)
    normalized = tuple(float(value) for value in values)
    polites_world = denormalize_action(normalized[:2], size)
    resource_world = denormalize_action(normalized[2:4], size)
    resource_local = (
        resource_world[0] - polites_world[0],
        resource_world[1] - polites_world[1],
    )
    physical_distance = math.hypot(*resource_local)
    coordinate_tolerance = float(np.finfo(np.float32).eps) * size
    return LocalObservationScene(
        polites_world_m=polites_world,
        resource_world_m=resource_world,
        resource_local_m=resource_local,
        observed_distance_m=normalized[4] * size,
        physical_distance_m=physical_distance,
        vision_radius_m=vision_radius,
        visible_resource_local_m=(
            resource_local
            if physical_distance <= vision_radius + coordinate_tolerance
            else None
        ),
        normalized_observation=normalized,
    )


def format_physical_status(scene: LocalObservationScene) -> str:
    """Describe only information available inside the physical vision panel."""

    if scene.visible_resource_local_m is None:
        return f"tree NOT VISIBLE · outside {scene.vision_radius_m:g} m physical vision"
    return (
        f"tree VISIBLE · local x={scene.resource_local_m[0]:+.1f} m  "
        f"z={scene.resource_local_m[1]:+.1f} m"
    )


def format_policy_readout(scene: LocalObservationScene) -> str:
    """Explain the deliberate mismatch between physical vision and policy input."""

    raw = ", ".join(
        f"{label}={value:.9g}"
        for label, value in zip(
            GATHER_OBSERVATION_LABELS,
            scene.normalized_observation,
            strict=True,
        )
    )
    disclosure = (
        "OMNISCIENT POLICY INPUT — tree coordinates are supplied regardless of "
        "LOS; the renderer frame above is the visibility authority."
    )
    return (
        f"{disclosure}\n"
        f"policy world target=({scene.resource_world_m[0]:.1f}, "
        f"{scene.resource_world_m[1]:.1f}) · observed distance="
        f"{scene.observed_distance_m:.1f} m\n"
        f"raw policy input: [{raw}]"
    )


class _TkGatherAgentView:
    """Engine-rendered LOS canvas driven synchronously by policy decisions."""

    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 790
    CANVAS_SIZE = 520

    BACKGROUND = "#171a16"
    GRID = "#343c31"
    TEXT = "#e8eadf"
    MUTED = "#a8ae9d"
    WARNING = "#e3b35b"

    def __init__(
        self,
        tk: Any,
        root: Any,
        map_size_m: float,
        vision_radius_m: float,
        frame_provider: Callable[[], EngineObserverFrame],
    ):
        self._tk = tk
        self._map_size_m = _validated_map_size(map_size_m)
        self._radius_m = _validated_vision_radius(vision_radius_m)
        self._frame_provider = frame_provider
        self._engine_photo = None
        self._closed = False

        self._root = root
        self._root.title("Polites physical vision")
        self._root.configure(background=self.BACKGROUND)
        self._root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}+24+48")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        try:
            self._root.attributes("-topmost", True)
        except tk.TclError:
            pass

        tk.Label(
            self._root,
            text="POLITES / ENGINE LOS VIEW",
            background=self.BACKGROUND,
            foreground=self.TEXT,
            font=("DejaVu Sans", 16, "bold"),
            anchor="w",
        ).pack(fill="x", padx=30, pady=(22, 2))
        tk.Label(
            self._root,
            text=(
                "Actual 0 A.D. scene · live engine Vision range · "
                "omniscient policy input disclosed below"
            ),
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=("DejaVu Sans", 9),
            anchor="w",
        ).pack(fill="x", padx=30, pady=(0, 14))

        self._canvas = tk.Canvas(
            self._root,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            background=self.BACKGROUND,
            highlightbackground=self.GRID,
            highlightthickness=1,
        )
        self._canvas.pack(padx=30)

        self._metrics = tk.StringVar(value="Waiting for the first policy step…")
        tk.Label(
            self._root,
            textvariable=self._metrics,
            background=self.BACKGROUND,
            foreground=self.TEXT,
            font=("DejaVu Sans", 10),
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=30, pady=(14, 4))

        self._raw_values = tk.StringVar(value="")
        tk.Label(
            self._root,
            textvariable=self._raw_values,
            background=self.BACKGROUND,
            foreground=self.WARNING,
            font=("DejaVu Sans Mono", 8),
            justify="left",
            anchor="w",
            wraplength=self.CANVAS_SIZE,
        ).pack(fill="x", padx=30)

        self._draw_waiting_frame()
        self._pump_events()

    def _draw_waiting_frame(self) -> None:
        self._canvas.delete("all")
        center = self.CANVAS_SIZE / 2.0
        self._canvas.create_text(
            center,
            center,
            text="Waiting for the first engine frame…",
            fill=self.MUTED,
            anchor="center",
            font=("DejaVu Sans", 10),
        )

    def _draw_scene(self, scene: LocalObservationScene, record: DecisionRecord) -> None:
        self._draw_engine_frame(self._frame_provider())

        self._metrics.set(
            f"episode {record.episode + 1}  ·  step {record.step}\n"
            "engine view: actual Player 1 LOS · renderer is visibility authority"
        )
        self._raw_values.set(format_policy_readout(scene))

    def _draw_engine_frame(self, frame: EngineObserverFrame) -> None:
        """Display the exact scene/LOS frame produced by the 0 A.D. renderer."""

        self._canvas.delete("all")
        self._engine_photo = self._tk.PhotoImage(data=frame.ppm, format="PPM")
        center = self.CANVAS_SIZE / 2.0
        self._canvas.create_image(center, center, image=self._engine_photo)
        self._canvas.create_text(
            12,
            12,
            text="ENGINE RENDER · PLAYER 1 LOS",
            fill=self.TEXT,
            anchor="nw",
            font=("DejaVu Sans", 8, "bold"),
        )

    def _pump_events(self) -> bool:
        if self._closed:
            return False
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
        return not self._closed

    def update(self, record: DecisionRecord) -> None:
        if self._closed:
            return
        scene = project_local_observation(
            record.observation,
            self._map_size_m,
            vision_radius_m=self._radius_m,
        )
        self._draw_scene(scene, record)
        self._pump_events()

    def pause(self, delay: float) -> None:
        """Keep Tk responsive while holding the pre-action frame on screen."""

        deadline = time.monotonic() + float(delay)
        while not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not self._pump_events():
                break
            time.sleep(min(0.02, remaining))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


def open_agent_view(env: object) -> AgentView:
    """Create the opt-in Tk window after validating gather-env compatibility."""

    labels = tuple(getattr(env, "observation_labels", ()))
    if labels != GATHER_OBSERVATION_LABELS:
        raise AgentViewUnavailable(
            "--agent-view currently supports only the five-value M0 gather observation"
        )
    map_size_m = getattr(env, "map_size_m", None)
    try:
        parsed_map_size = _validated_map_size(map_size_m)
    except (TypeError, ValueError) as error:
        raise AgentViewUnavailable(
            "the selected environment does not expose a valid map_size_m"
        ) from error

    engine_observer = getattr(env, "engine_observer", None)
    check_available = getattr(engine_observer, "check_available", None)
    frame_provider = getattr(env, "capture_agent_frame", None)
    if not callable(check_available) or not callable(frame_provider):
        raise AgentViewUnavailable(
            "the selected environment does not expose the engine observer"
        )
    try:
        check_available()
    except EngineObserverError as error:
        raise AgentViewUnavailable(str(error)) from error

    try:
        import tkinter as tk
    except ImportError as error:
        raise AgentViewUnavailable(
            "Tkinter is unavailable in this Python runtime"
        ) from error

    root = None
    try:
        root = tk.Tk()
        return _TkGatherAgentView(
            tk,
            root,
            parsed_map_size,
            POLITES_VISION_RADIUS_M,
            frame_provider,
        )
    except tk.TclError as error:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
        raise AgentViewUnavailable(
            "could not open a Tk window; run evaluation from a graphical session"
        ) from error
