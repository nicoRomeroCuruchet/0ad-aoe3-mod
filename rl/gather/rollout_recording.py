"""No-GUI rollout recording for engine-rendered gather observations."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import time
import zlib
from pathlib import Path
from typing import Any

from rl.experiments.evaluation import DecisionRecord
from rl.gather.engine_observer import EngineObserverFrame, EngineObserverUnavailable


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum & 0xFFFFFFFF)
    )


def _ppm_pixels(frame: EngineObserverFrame) -> bytes:
    _magic, _dimensions, _maximum, pixels = frame.ppm.split(b"\n", 3)
    if len(pixels) != frame.width * frame.height * 3:
        raise ValueError("PPM frame size does not match its dimensions")
    return pixels


def ppm_to_png(frame: EngineObserverFrame) -> bytes:
    """Encode the observer's P6 PPM frame as a dependency-free RGB PNG."""

    pixels = _ppm_pixels(frame)
    scanlines = b"".join(
        b"\x00" + pixels[row : row + frame.width * 3]
        for row in range(0, len(pixels), frame.width * 3)
    )
    header = struct.pack(">IIBBBBB", frame.width, frame.height, 8, 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _coord_to_pixel(value: float, size: int) -> int:
    normalized = max(-1.0, min(1.0, float(value)))
    return int(round((normalized + 1.0) * 0.5 * (size - 1)))


def _paint_square(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for py in range(max(0, y - radius), min(height, y + radius + 1)):
        for px in range(max(0, x - radius), min(width, x + radius + 1)):
            offset = (py * width + px) * 3
            pixels[offset : offset + 3] = bytes(color)


def _schematic_frame(record: DecisionRecord, *, size: int = 512) -> EngineObserverFrame:
    observation = record.observation.astype(float).reshape(-1)
    action = record.action.astype(float).reshape(-1)
    pixels = bytearray([24, 28, 32] * size * size)

    for line in range(0, size, 64):
        for i in range(size):
            row_offset = (line * size + i) * 3
            col_offset = (i * size + line) * 3
            pixels[row_offset : row_offset + 3] = b"\x2d\x35\x3c"
            pixels[col_offset : col_offset + 3] = b"\x2d\x35\x3c"

    if observation.size >= 4:
        villager = (
            _coord_to_pixel(observation[0], size),
            _coord_to_pixel(observation[1], size),
        )
        resource = (
            _coord_to_pixel(observation[2], size),
            _coord_to_pixel(observation[3], size),
        )
        _paint_square(pixels, size, size, resource[0], resource[1], 8, (75, 178, 101))
        _paint_square(pixels, size, size, villager[0], villager[1], 7, (88, 166, 255))
    if action.size >= 2:
        target = (_coord_to_pixel(action[0], size), _coord_to_pixel(action[1], size))
        target_color = (245, 203, 92)
        if action.size >= 3 and action[2] > 0:
            target_color = (255, 126, 80)
        _paint_square(pixels, size, size, target[0], target[1], 5, target_color)

    ppm = b"P6\n%d %d\n255\n" % (size, size) + bytes(pixels)
    return EngineObserverFrame(size, size, ppm)


class AgentViewRolloutRecorder:
    """Save engine observer frames and policy metadata for later inspection."""

    def __init__(
        self,
        root: Path,
        env: object,
        *,
        frame_seconds: float = 0.25,
        capture_attempts: int = 10,
        capture_retry_delay: float = 0.25,
        fallback_to_schematic: bool = False,
    ) -> None:
        self.root = root
        self.env = env
        self.frame_seconds = frame_seconds
        self.capture_attempts = capture_attempts
        self.capture_retry_delay = capture_retry_delay
        self.fallback_to_schematic = fallback_to_schematic
        self.frames_dir = root / "frames"
        self.metadata_path = root / "metadata.jsonl"
        self.index_path = root / "index.html"
        self._records: list[dict[str, Any]] = []

    def _capture_frame(self, record: DecisionRecord) -> tuple[EngineObserverFrame, str]:
        capture = getattr(self.env, "capture_agent_frame", None)
        if not callable(capture):
            raise RuntimeError("selected environment cannot capture agent-view frames")
        last_error: EngineObserverUnavailable | None = None
        for attempt in range(self.capture_attempts):
            try:
                return capture(), "engine"
            except EngineObserverUnavailable as error:
                last_error = error
                if attempt + 1 >= self.capture_attempts:
                    break
                if self.capture_retry_delay:
                    time.sleep(self.capture_retry_delay)
        if self.fallback_to_schematic:
            print(
                "warning: engine observer frame unavailable; "
                "recording schematic rollout frame instead",
                flush=True,
            )
            return _schematic_frame(record), "schematic"
        raise EngineObserverUnavailable(
            "engine observer did not produce a frame after several attempts; "
            "restart `make server` from a graphical session or with Xvfb, and make "
            "sure no training client is still connected to the same RL server"
        ) from last_error

    def observe(self, record: DecisionRecord) -> None:
        first_record = not self._records
        frame, frame_source = self._capture_frame(record)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        frame_name = f"ep{record.episode:03d}-step{record.step:04d}.png"
        frame_path = self.frames_dir / frame_name
        frame_path.write_bytes(ppm_to_png(frame))
        entry = {
            "episode": record.episode,
            "step": record.step,
            "frame": f"frames/{frame_name}",
            "frame_source": frame_source,
            "observation": record.observation.astype(float).tolist(),
            "action": record.action.astype(float).tolist(),
        }
        self._records.append(entry)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open(
            "w" if first_record else "a",
            encoding="utf-8",
        ) as metadata:
            metadata.write(json.dumps(entry, sort_keys=True))
            metadata.write("\n")

    def write_index(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        frame_data = json.dumps(self._records)
        delay_ms = max(1, int(self.frame_seconds * 1000))
        self.index_path.write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    '<html lang="en">',
                    "<head>",
                    '<meta charset="utf-8">',
                    "<title>0 A.D. rollout</title>",
                    "<style>",
                    "body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }",
                    "img { image-rendering: pixelated; width: min(512px, 100%); border: 1px solid #444; }",
                    "button { margin-right: 8px; }",
                    "pre { white-space: pre-wrap; }",
                    "</style>",
                    "</head>",
                    "<body>",
                    "<h1>0 A.D. rollout</h1>",
                    '<p><button id="play">Play</button><button id="pause">Pause</button></p>',
                    '<img id="frame" alt="Recorded agent view frame">',
                    '<pre id="meta"></pre>',
                    "<script>",
                    f"const frames = {frame_data};",
                    f"const delayMs = {delay_ms};",
                    "let i = 0;",
                    "let timer = null;",
                    "const img = document.getElementById('frame');",
                    "const meta = document.getElementById('meta');",
                    "function render() {",
                    "  if (!frames.length) { meta.textContent = 'No frames recorded.'; return; }",
                    "  const frame = frames[i % frames.length];",
                    "  img.src = frame.frame;",
                    "  meta.textContent = JSON.stringify(frame, null, 2);",
                    "}",
                    "function play() { if (timer === null) timer = setInterval(() => { i++; render(); }, delayMs); }",
                    "function pause() { clearInterval(timer); timer = null; }",
                    "document.getElementById('play').onclick = play;",
                    "document.getElementById('pause').onclick = pause;",
                    "render();",
                    "</script>",
                    "</body>",
                    "</html>",
                ]
            ),
            encoding="utf-8",
        )
        return self.index_path

    def write_video(self, path: Path) -> Path:
        """Encode recorded PNG frames to MP4 with ffmpeg."""

        if not self._records:
            raise RuntimeError("cannot write a rollout video without recorded frames")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required to write MP4 rollout videos; "
                "install it with: sudo apt install ffmpeg"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, round(1.0 / self.frame_seconds))
        command = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(self.frames_dir / "*.png"),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        subprocess.run(command, check=True)
        return path


def mode_recording_dir(root: Path, *, deterministic: bool) -> Path:
    return root / ("deterministic" if deterministic else "stochastic")
