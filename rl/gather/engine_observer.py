"""Client and strict frame parser for the patched 0 A.D. observer endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


ENGINE_OBSERVER_PROTOCOL = "0ad-rl-observer-v3"
MAX_VIEW_RANGE_M = 512.0
MAX_FRAME_DIMENSION = 512
MAX_FRAME_BYTES = MAX_FRAME_DIMENSION * MAX_FRAME_DIMENSION * 3 + 64


class EngineObserverError(RuntimeError):
    """Base error for engine-rendered observation failures."""


class EngineObserverUnavailable(EngineObserverError):
    """Raised when the running 0 A.D. binary lacks the observer endpoint."""


class EngineObserverProtocolError(EngineObserverError):
    """Raised when the engine returns a malformed or incompatible frame."""


@dataclass(frozen=True, slots=True)
class EngineObserverFrame:
    """Validated binary PPM image returned by the engine."""

    width: int
    height: int
    ppm: bytes


def parse_ppm(payload: bytes) -> EngineObserverFrame:
    """Validate the deliberately small P6 PPM contract used by the endpoint."""

    if not isinstance(payload, bytes):
        raise EngineObserverProtocolError("observer frame must be bytes")
    if len(payload) > MAX_FRAME_BYTES:
        raise EngineObserverProtocolError("observer frame exceeds the size limit")

    try:
        magic, dimensions, maximum, pixels = payload.split(b"\n", 3)
    except ValueError as error:
        raise EngineObserverProtocolError(
            "observer frame has an incomplete P6 header"
        ) from error

    if magic != b"P6":
        raise EngineObserverProtocolError("observer frame must use binary P6 PPM")
    try:
        width_text, height_text = dimensions.split()
        width = int(width_text)
        height = int(height_text)
    except (ValueError, TypeError) as error:
        raise EngineObserverProtocolError(
            "observer frame has invalid dimensions"
        ) from error
    if not (1 <= width <= MAX_FRAME_DIMENSION and 1 <= height <= MAX_FRAME_DIMENSION):
        raise EngineObserverProtocolError("observer frame dimensions are out of range")
    if maximum != b"255":
        raise EngineObserverProtocolError("observer frame color maximum must be 255")
    if len(pixels) != width * height * 3:
        raise EngineObserverProtocolError("observer frame has incomplete pixel data")
    return EngineObserverFrame(width=width, height=height, ppm=payload)


def _validated_base_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "observer URI must be an HTTP(S) server URL without credentials"
        )
    return uri.rstrip("/")


class EngineObserverClient:
    """Synchronous client used immediately before each policy action."""

    def __init__(
        self,
        uri: str,
        *,
        timeout: float = 2.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self._base_uri = _validated_base_uri(uri)
        self._timeout = float(timeout)
        if not 0.0 < self._timeout <= 30.0:
            raise ValueError("observer timeout must be between 0 and 30 seconds")
        self._opener = opener

    def _read(self, route: str, *, accept: str, limit: int) -> tuple[bytes, str]:
        request = Request(
            f"{self._base_uri}/{route}",
            headers={"Accept": accept},
            method="GET",
        )
        try:
            # The URL is validated above and intentionally targets the same server
            # already selected for the zero_ad environment.
            with self._opener(request, timeout=self._timeout) as response:  # noqa: S310
                payload = response.read(limit + 1)
                content_type = response.headers.get("Content-Type", "")
        except HTTPError as error:
            if error.code == 404:
                raise EngineObserverUnavailable(
                    "the running 0 A.D. binary has no rendered observer; "
                    "run `make engine-observer`, then restart `make server`"
                ) from error
            raise EngineObserverUnavailable(
                f"engine observer request failed with HTTP {error.code}"
            ) from error
        except (OSError, URLError) as error:
            raise EngineObserverUnavailable(
                f"could not reach the engine observer at {self._base_uri}"
            ) from error
        if len(payload) > limit:
            raise EngineObserverProtocolError("engine observer response is too large")
        return payload, content_type.partition(";")[0].strip().lower()

    def check_available(self) -> None:
        """Fail before training if the server is not the patched engine."""

        payload, _content_type = self._read(
            "observer/status",
            accept="text/plain",
            limit=128,
        )
        if payload.decode("ascii", errors="replace") != ENGINE_OBSERVER_PROTOCOL:
            raise EngineObserverProtocolError(
                "the engine observer protocol does not match this RL client"
            )

    def capture(
        self,
        entity_id: int,
        view_range_m: float | None = None,
        focus_xz: tuple[float, float] | None = None,
    ) -> EngineObserverFrame:
        """Request the Player-1 LOS render centered on one owned entity.

        ``view_range_m`` overrides the view half-width in metres; ``None``
        keeps the entity's own vision range, which is the per-agent view.
        ``focus_xz`` recentres the camera on a map point while the entity
        keeps selecting whose line of sight is rendered.
        """

        if (
            isinstance(entity_id, bool)
            or not isinstance(entity_id, int)
            or not 0 < entity_id < 2**32
        ):
            raise ValueError("observer entity ID must be a positive integer")
        parameters: dict[str, Any] = {"entity": entity_id}
        if view_range_m is not None:
            range_value = float(view_range_m)
            if not 0.0 < range_value <= MAX_VIEW_RANGE_M:
                raise ValueError(
                    f"observer view range must be in (0, {MAX_VIEW_RANGE_M}] metres"
                )
            parameters["range"] = f"{range_value:.3f}"
        if focus_xz is not None:
            x, z = (float(focus_xz[0]), float(focus_xz[1]))
            for name, value in (("x", x), ("z", z)):
                if not 0.0 <= value <= 4096.0:
                    raise ValueError(
                        f"observer focus {name} must be in [0, 4096] metres"
                    )
                parameters[name] = f"{value:.3f}"
        query = urlencode(parameters)
        payload, content_type = self._read(
            f"observe?{query}",
            accept="image/x-portable-pixmap",
            limit=MAX_FRAME_BYTES,
        )
        if content_type != "image/x-portable-pixmap":
            raise EngineObserverProtocolError(
                "engine observer returned an unexpected content type"
            )
        return parse_ppm(payload)
