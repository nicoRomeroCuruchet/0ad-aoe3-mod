"""Tk debug window for the M0 gather policy observation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rl.experiments.evaluation import DecisionRecord

from .core import GATHER_OBSERVATION_LABELS, denormalize_action


class AgentViewUnavailable(RuntimeError):
    """Raised when the requested observation window cannot be created."""


class AgentView(Protocol):
    """Runtime contract used by the evaluation CLI."""

    def update(self, record: DecisionRecord) -> None: ...

    def pause(self, delay: float) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalObservationScene:
    """Exact M0 policy input expressed in world and Polites-local coordinates."""

    polites_world_m: tuple[float, float]
    resource_world_m: tuple[float, float]
    resource_local_m: tuple[float, float]
    observed_distance_m: float
    normalized_observation: tuple[float, ...]


def _validated_map_size(map_size_m: float) -> float:
    parsed = float(map_size_m)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("map_size_m must be finite and positive")
    return parsed


def project_local_observation(
    observation: object,
    map_size_m: float,
) -> LocalObservationScene:
    """Recenter the exact five-value M0 observation on the Polites."""

    values = np.asarray(observation).reshape(-1)
    if len(values) != len(GATHER_OBSERVATION_LABELS):
        raise ValueError("gather agent view requires exactly five values")
    if not np.isfinite(values).all():
        raise ValueError("gather agent view values must be finite")

    size = _validated_map_size(map_size_m)
    normalized = tuple(float(value) for value in values)
    polites_world = denormalize_action(normalized[:2], size)
    resource_world = denormalize_action(normalized[2:4], size)
    resource_local = (
        resource_world[0] - polites_world[0],
        resource_world[1] - polites_world[1],
    )
    return LocalObservationScene(
        polites_world_m=polites_world,
        resource_world_m=resource_world,
        resource_local_m=resource_local,
        observed_distance_m=normalized[4] * size,
        normalized_observation=normalized,
    )


class _TkGatherAgentView:
    """Small live canvas driven synchronously by evaluation steps."""

    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 760
    CANVAS_SIZE = 520
    CANVAS_MARGIN = 34

    BACKGROUND = "#171a16"
    PANEL = "#20251e"
    GRID = "#343c31"
    MAP_EDGE = "#77836d"
    TEXT = "#e8eadf"
    MUTED = "#a8ae9d"
    POLITES = "#57a6d9"
    RESOURCE = "#8fbd63"
    VECTOR = "#d7a64d"

    def __init__(self, tk: Any, root: Any, map_size_m: float):
        self._tk = tk
        self._map_size_m = _validated_map_size(map_size_m)
        self._radius_m = self._map_size_m
        self._closed = False

        self._root = root
        self._root.title("Polites local observation")
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
            text="POLITES / LOCAL OBSERVATION",
            background=self.BACKGROUND,
            foreground=self.TEXT,
            font=("DejaVu Sans", 16, "bold"),
            anchor="w",
        ).pack(fill="x", padx=30, pady=(22, 2))
        tk.Label(
            self._root,
            text="Centered projection of policy input — not camera pixels",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=("DejaVu Sans", 9),
            anchor="w",
        ).pack(fill="x", padx=30, pady=(0, 14))

        self._canvas = tk.Canvas(
            self._root,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            background=self.PANEL,
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
            foreground=self.MUTED,
            font=("DejaVu Sans Mono", 8),
            justify="left",
            anchor="w",
            wraplength=self.CANVAS_SIZE,
        ).pack(fill="x", padx=30)

        self._draw_grid()
        self._pump_events()

    def _canvas_point(self, local_x_m: float, local_z_m: float) -> tuple[float, float]:
        center = self.CANVAS_SIZE / 2.0
        extent = center - self.CANVAS_MARGIN
        scale = extent / self._radius_m
        return center + local_x_m * scale, center + local_z_m * scale

    def _clip_local_point(self, local_x_m: float, local_z_m: float) -> tuple[float, float, bool]:
        limit = self._radius_m * 0.9
        largest_component = max(abs(local_x_m), abs(local_z_m))
        if largest_component <= limit:
            return local_x_m, local_z_m, True
        factor = limit / largest_component
        return local_x_m * factor, local_z_m * factor, False

    def _draw_grid(self) -> None:
        self._canvas.delete("all")
        center = self.CANVAS_SIZE / 2.0
        extent = center - self.CANVAS_MARGIN
        for index in range(-4, 5):
            position = center + extent * index / 4
            self._canvas.create_line(
                self.CANVAS_MARGIN,
                position,
                self.CANVAS_SIZE - self.CANVAS_MARGIN,
                position,
                fill=self.GRID,
            )
            self._canvas.create_line(
                position,
                self.CANVAS_MARGIN,
                position,
                self.CANVAS_SIZE - self.CANVAS_MARGIN,
                fill=self.GRID,
            )
        self._canvas.create_text(
            self.CANVAS_SIZE - self.CANVAS_MARGIN,
            center - 9,
            text="+x",
            fill=self.MUTED,
            anchor="e",
            font=("DejaVu Sans", 8),
        )
        self._canvas.create_text(
            center + 8,
            self.CANVAS_SIZE - self.CANVAS_MARGIN,
            text="+z",
            fill=self.MUTED,
            anchor="sw",
            font=("DejaVu Sans", 8),
        )
        self._canvas.create_text(
            self.CANVAS_MARGIN,
            self.CANVAS_MARGIN - 10,
            text=f"relative plot ±{self._radius_m:.0f} m · no sensing cutoff",
            fill=self.MUTED,
            anchor="w",
            font=("DejaVu Sans", 8),
        )

    def _draw_map_boundary(self, scene: LocalObservationScene) -> None:
        left = -scene.polites_world_m[0]
        top = -scene.polites_world_m[1]
        right = self._map_size_m - scene.polites_world_m[0]
        bottom = self._map_size_m - scene.polites_world_m[1]
        x1, y1 = self._canvas_point(left, top)
        x2, y2 = self._canvas_point(right, bottom)
        self._canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=self.MAP_EDGE,
            width=2,
        )

    def _draw_scene(self, scene: LocalObservationScene, record: DecisionRecord) -> None:
        self._draw_grid()
        self._draw_map_boundary(scene)

        center = self.CANVAS_SIZE / 2.0
        resource_x, resource_z, visible = self._clip_local_point(
            *scene.resource_local_m
        )
        resource_canvas_x, resource_canvas_y = self._canvas_point(
            resource_x,
            resource_z,
        )
        self._canvas.create_line(
            center,
            center,
            resource_canvas_x,
            resource_canvas_y,
            fill=self.VECTOR,
            width=2,
            dash=(6, 5),
        )

        self._canvas.create_rectangle(
            resource_canvas_x - 3,
            resource_canvas_y + 4,
            resource_canvas_x + 3,
            resource_canvas_y + 14,
            fill="#725035",
            outline="",
        )
        self._canvas.create_oval(
            resource_canvas_x - 10,
            resource_canvas_y - 10,
            resource_canvas_x + 10,
            resource_canvas_y + 9,
            fill=self.RESOURCE if visible else self.PANEL,
            outline=self.RESOURCE,
            width=3 if not visible else 1,
        )
        self._canvas.create_text(
            resource_canvas_x,
            resource_canvas_y - 17,
            text=(
                "RESOURCE"
                if visible
                else "RESOURCE / CLIPPED"
            ),
            fill=self.RESOURCE,
            anchor="s",
            font=("DejaVu Sans", 8, "bold"),
        )

        self._canvas.create_oval(
            center - 10,
            center - 10,
            center + 10,
            center + 10,
            fill=self.POLITES,
            outline=self.TEXT,
            width=2,
        )
        self._canvas.create_text(
            center,
            center + 17,
            text="POLITES",
            fill=self.POLITES,
            anchor="n",
            font=("DejaVu Sans", 8, "bold"),
        )

        clipping_note = (
            ""
            if visible
            else "\nplot marker clipped; coordinates still observed — no sensing cutoff"
        )
        self._metrics.set(
            f"episode {record.episode + 1}  ·  step {record.step}\n"
            f"resource local: x={scene.resource_local_m[0]:+.1f} m  "
            f"z={scene.resource_local_m[1]:+.1f} m  ·  "
            f"observed distance={scene.observed_distance_m:.1f} m\n"
            f"world: Polites=({scene.polites_world_m[0]:.1f}, "
            f"{scene.polites_world_m[1]:.1f})  resource=("
            f"{scene.resource_world_m[0]:.1f}, {scene.resource_world_m[1]:.1f})"
            f"{clipping_note}"
        )
        raw = ", ".join(
            f"{label}={value:.9g}"
            for label, value in zip(
                GATHER_OBSERVATION_LABELS,
                scene.normalized_observation,
                strict=True,
            )
        )
        self._raw_values.set(f"raw policy input: [{raw}]")

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
        scene = project_local_observation(record.observation, self._map_size_m)
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

    try:
        import tkinter as tk
    except ImportError as error:
        raise AgentViewUnavailable(
            "Tkinter is unavailable in this Python runtime"
        ) from error

    root = None
    try:
        root = tk.Tk()
        return _TkGatherAgentView(tk, root, parsed_map_size)
    except tk.TclError as error:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
        raise AgentViewUnavailable(
            "could not open a Tk window; run evaluation from a graphical session"
        ) from error
