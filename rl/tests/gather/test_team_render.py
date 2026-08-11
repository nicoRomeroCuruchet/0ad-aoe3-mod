from pathlib import Path

from rl.gather.live_view import LiveEpisodeView
from rl.gather.team_render import TeamRenderState, render_team_frame


def _state(carried=(0.0, 20.0)):
    return TeamRenderState(
        villager_xz=((100.0, 100.0), (200.0, 140.0)),
        resource_xz=((300.0, 100.0), (320.0, 200.0)),
        resource_remaining=(200.0, 0.0),
        carried=carried,
        dropsite_xz=(60.0, 160.0),
        targets_xz=((300.0, 100.0), (60.0, 160.0)),
    )


def test_render_returns_a_png_of_the_requested_size():
    png = render_team_frame(_state(), size=128)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 100


def test_frames_differ_when_a_villager_picks_up_wood():
    empty = render_team_frame(_state(carried=(0.0, 0.0)), size=128)
    loaded = render_team_frame(_state(carried=(20.0, 20.0)), size=128)

    assert empty != loaded


def test_live_view_writes_a_page_and_frames(tmp_path: Path):
    view = LiveEpisodeView(tmp_path / "live", refresh_seconds=3, max_frames=4)
    for _ in range(6):
        view.observe(_state())

    page = view.publish(steps=3000, summary="success 0%")

    assert page.exists()
    text = page.read_text()
    assert "training step 3000" in text
    assert "success 0%" in text
    assert 'content="3"' in text
    # Capped at max_frames, and the buffer resets for the next episode.
    assert len(list((tmp_path / "live").glob("frame*.png"))) == 4
    assert view._frames == []


def test_live_view_without_frames_does_not_write_a_page(tmp_path: Path):
    view = LiveEpisodeView(tmp_path / "live")

    page = view.publish(steps=10, summary="none")

    assert not page.exists()
