"""Top-down schematic of a team gather step, for watching training live.

The engine observer renders one villager's 32 m keyhole, which cannot show four
villagers spread across the map. This draws the whole work area from the
observation instead: every villager, every tree, the dropsite, and who is
carrying a load. It is pure Python, so it costs nothing on the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rollout_recording import ppm_to_png
from .engine_observer import EngineObserverFrame


BACKGROUND = (26, 32, 28)
GRID = (44, 54, 48)
TREE = (74, 152, 92)
TREE_EMPTY = (60, 70, 62)
DROPSITE = (196, 142, 74)
VILLAGER = (86, 158, 230)
VILLAGER_LOADED = (245, 196, 96)
TARGET = (232, 108, 60)


@dataclass(frozen=True, slots=True)
class TeamRenderState:
    """Everything the schematic draws, in metres."""

    villager_xz: tuple[tuple[float, float], ...]
    resource_xz: tuple[tuple[float, float], ...]
    resource_remaining: tuple[float, ...]
    carried: tuple[float, ...]
    dropsite_xz: tuple[float, float]
    targets_xz: tuple[tuple[float, float], ...] = ()


def _to_pixel(value: float, low: float, span: float, size: int) -> int:
    scaled = (value - low) / span if span > 0 else 0.5
    return max(0, min(size - 1, int(round(scaled * (size - 1)))))


def _paint(pixels: bytearray, size: int, x: int, y: int, radius: int, color) -> None:
    for row in range(max(0, y - radius), min(size, y + radius + 1)):
        for column in range(max(0, x - radius), min(size, x + radius + 1)):
            offset = (row * size + column) * 3
            pixels[offset : offset + 3] = bytes(color)


def _paint_cross(pixels: bytearray, size: int, x: int, y: int, radius: int, color) -> None:
    for delta in range(-radius, radius + 1):
        for point in ((x + delta, y), (x, y + delta)):
            column, row = point
            if 0 <= column < size and 0 <= row < size:
                offset = (row * size + column) * 3
                pixels[offset : offset + 3] = bytes(color)


def render_team_frame(state: TeamRenderState, *, size: int = 512, margin_m: float = 24.0) -> bytes:
    """Draw one top-down PNG framing every entity in the scene."""

    points = [
        *state.villager_xz,
        *state.resource_xz,
        state.dropsite_xz,
    ]
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    low_x, high_x = min(xs) - margin_m, max(xs) + margin_m
    low_z, high_z = min(zs) - margin_m, max(zs) + margin_m
    # One scale for both axes keeps the geometry undistorted.
    span = max(high_x - low_x, high_z - low_z)
    low_x -= (span - (high_x - low_x)) / 2.0
    low_z -= (span - (high_z - low_z)) / 2.0

    pixels = bytearray(bytes(BACKGROUND) * size * size)
    for line in range(0, size, 64):
        for index in range(size):
            for offset in ((line * size + index) * 3, (index * size + line) * 3):
                pixels[offset : offset + 3] = bytes(GRID)

    def place(point):
        return (
            _to_pixel(point[0], low_x, span, size),
            _to_pixel(point[1], low_z, span, size),
        )

    dropsite = place(state.dropsite_xz)
    _paint(pixels, size, dropsite[0], dropsite[1], 9, DROPSITE)

    for index, resource in enumerate(state.resource_xz):
        remaining = (
            state.resource_remaining[index]
            if index < len(state.resource_remaining)
            else 0.0
        )
        column, row = place(resource)
        _paint(pixels, size, column, row, 7, TREE if remaining > 0 else TREE_EMPTY)

    for index, target in enumerate(state.targets_xz):
        column, row = place(target)
        _paint_cross(pixels, size, column, row, 6, TARGET)
        del index

    for index, villager in enumerate(state.villager_xz):
        carrying = index < len(state.carried) and state.carried[index] > 0.0
        column, row = place(villager)
        _paint(
            pixels,
            size,
            column,
            row,
            6,
            VILLAGER_LOADED if carrying else VILLAGER,
        )

    return ppm_to_png(
        EngineObserverFrame(size, size, b"P6\n%d %d\n255\n" % (size, size) + bytes(pixels))
    )
