"""Write a self-refreshing page showing the newest evaluated episode."""

from __future__ import annotations

from pathlib import Path

from .team_render import TeamRenderState, render_team_frame


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>M2 live &mdash; step {steps}</title>
<style>
 body {{ background:#11150f; color:#e6ebe3; font-family:ui-monospace,Menlo,monospace;
         margin:0; padding:24px; display:flex; flex-direction:column; gap:14px; align-items:flex-start; }}
 h1 {{ font-size:15px; font-weight:600; margin:0; letter-spacing:.02em; }}
 .meta {{ font-size:12px; color:#9aa79b; }}
 .strip {{ display:flex; gap:8px; flex-wrap:wrap; }}
 img {{ image-rendering:pixelated; width:260px; border:1px solid #2c352d; }}
 .legend {{ font-size:11.5px; color:#9aa79b; display:flex; gap:16px; flex-wrap:wrap; }}
 b {{ color:#e6ebe3; font-weight:600; }}
</style>
</head>
<body>
<h1>M2 team gather &mdash; training step {steps}</h1>
<div class="meta">{summary}</div>
<div class="strip">{images}</div>
<div class="legend">
  <span><b style="color:#569ee6">&#9632;</b> villager</span>
  <span><b style="color:#f5c460">&#9632;</b> carrying wood</span>
  <span><b style="color:#4a985c">&#9632;</b> tree</span>
  <span><b style="color:#c48e4a">&#9632;</b> storehouse</span>
  <span><b style="color:#e86c3c">+</b> where the policy pointed</span>
  <span>page refreshes every {refresh}s</span>
</div>
</body>
</html>
"""


class LiveEpisodeView:
    """Keep `root/index.html` showing frames from the most recent episode."""

    def __init__(self, root: Path, *, refresh_seconds: int = 5, max_frames: int = 12):
        self.root = Path(root)
        self.refresh_seconds = refresh_seconds
        self.max_frames = max_frames
        self._frames: list[TeamRenderState] = []

    def observe(self, state: TeamRenderState) -> None:
        """Collect one decision's worth of scene state."""

        self._frames.append(state)

    def publish(self, *, steps: int, summary: str) -> Path:
        """Write the collected frames and the page, then start a new episode."""

        frames = self._frames[: self.max_frames]
        self._frames = []
        if not frames:
            return self.root / "index.html"
        self.root.mkdir(parents=True, exist_ok=True)
        for index, state in enumerate(frames):
            (self.root / f"frame{index:03d}.png").write_bytes(render_team_frame(state))
        for stale in range(len(frames), self.max_frames):
            (self.root / f"frame{stale:03d}.png").unlink(missing_ok=True)
        images = "".join(
            f'<img src="frame{index:03d}.png?v={steps}" alt="decision {index}">'
            for index in range(len(frames))
        )
        page = self.root / "index.html"
        page.write_text(
            PAGE.format(
                refresh=self.refresh_seconds,
                steps=steps,
                summary=summary,
                images=images,
            ),
            encoding="utf-8",
        )
        return page
