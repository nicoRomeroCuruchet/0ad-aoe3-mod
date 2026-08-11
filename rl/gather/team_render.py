"""Pure-Python top-down schematic that frames an entire gather team."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .engine_observer import EngineObserverFrame


BACKGROUND = (26, 32, 28)
GRID = (44, 54, 48)
TREE = (74, 152, 92)
TREE_EMPTY = (60, 70, 62)
DROPSITE = (196, 142, 74)
VILLAGER = (86, 158, 230)
VILLAGER_LOADED = (245, 196, 96)
TARGET = (232, 108, 60)


def _valid_point(point: object) -> bool:
    if not isinstance(point, tuple) or len(point) != 2:
        return False
    try:
        return all(math.isfinite(float(value)) for value in point)
    except (TypeError, ValueError):
        return False


def _valid_amount(value: object) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed >= 0.0


@dataclass(frozen=True, slots=True)
class TeamRenderState:
    """Everything the schematic draws, expressed in map metres."""

    villager_xz: tuple[tuple[float, float], ...]
    resource_xz: tuple[tuple[float, float], ...]
    resource_remaining: tuple[float, ...]
    carried: tuple[float, ...]
    dropsite_xz: tuple[float, float]
    targets_xz: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.villager_xz or not self.resource_xz:
            raise ValueError("team render state needs villagers and resources")
        if len(self.resource_remaining) != len(self.resource_xz):
            raise ValueError("resource_remaining must match resource_xz")
        if len(self.carried) != len(self.villager_xz):
            raise ValueError("carried must match villager_xz")
        points = (
            *self.villager_xz,
            *self.resource_xz,
            self.dropsite_xz,
            *self.targets_xz,
        )
        if not all(_valid_point(point) for point in points):
            raise ValueError("team render coordinates must be finite x/z pairs")
        amounts = (*self.resource_remaining, *self.carried)
        if not all(_valid_amount(value) for value in amounts):
            raise ValueError("team render amounts must be finite and non-negative")


def _to_pixel(value: float, low: float, span: float, size: int) -> int:
    scaled = (value - low) / span if span > 0.0 else 0.5
    return max(0, min(size - 1, int(round(scaled * (size - 1)))))


def _paint(
    pixels: bytearray,
    size: int,
    x: int,
    y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for row in range(max(0, y - radius), min(size, y + radius + 1)):
        for column in range(max(0, x - radius), min(size, x + radius + 1)):
            offset = (row * size + column) * 3
            pixels[offset : offset + 3] = bytes(color)


def _paint_cross(
    pixels: bytearray,
    size: int,
    x: int,
    y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for delta in range(-radius, radius + 1):
        for column, row in ((x + delta, y), (x, y + delta)):
            if 0 <= column < size and 0 <= row < size:
                offset = (row * size + column) * 3
                pixels[offset : offset + 3] = bytes(color)


def render_team_observer_frame(
    state: TeamRenderState,
    *,
    size: int = 512,
    margin_m: float = 24.0,
) -> EngineObserverFrame:
    """Draw a square PPM frame containing every team entity without distortion."""

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    try:
        margin = float(margin_m)
    except (TypeError, ValueError) as error:
        raise ValueError("margin_m must be finite and non-negative") from error
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin_m must be finite and non-negative")

    points = [*state.villager_xz, *state.resource_xz, state.dropsite_xz]
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    low_x, high_x = min(xs) - margin, max(xs) + margin
    low_z, high_z = min(zs) - margin, max(zs) + margin
    span = max(high_x - low_x, high_z - low_z)
    low_x -= (span - (high_x - low_x)) / 2.0
    low_z -= (span - (high_z - low_z)) / 2.0

    pixels = bytearray(bytes(BACKGROUND) * size * size)
    for line in range(0, size, 64):
        for index in range(size):
            for offset in ((line * size + index) * 3, (index * size + line) * 3):
                pixels[offset : offset + 3] = bytes(GRID)

    def place(point: tuple[float, float]) -> tuple[int, int]:
        return (
            _to_pixel(point[0], low_x, span, size),
            _to_pixel(point[1], low_z, span, size),
        )

    drop_x, drop_z = place(state.dropsite_xz)
    _paint(pixels, size, drop_x, drop_z, 9, DROPSITE)
    for resource, remaining in zip(
        state.resource_xz,
        state.resource_remaining,
        strict=True,
    ):
        column, row = place(resource)
        _paint(pixels, size, column, row, 7, TREE if remaining > 0.0 else TREE_EMPTY)
    for target in state.targets_xz:
        column, row = place(target)
        _paint_cross(pixels, size, column, row, 6, TARGET)
    for villager, carried in zip(state.villager_xz, state.carried, strict=True):
        column, row = place(villager)
        color = VILLAGER_LOADED if carried > 0.0 else VILLAGER
        _paint(pixels, size, column, row, 6, color)

    ppm = b"P6\n%d %d\n255\n" % (size, size) + bytes(pixels)
    return EngineObserverFrame(size, size, ppm)
