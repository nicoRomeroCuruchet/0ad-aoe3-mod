import math
import sys
from types import ModuleType

import numpy as np
import pytest

from rl.gather.agent_view import (
    _TkGatherAgentView,
    AgentViewUnavailable,
    format_physical_status,
    format_policy_readout,
    open_agent_view,
    project_local_observation,
)
from rl.gather.engine_observer import EngineObserverFrame
from rl.gather.core import (
    GATHER_OBSERVATION_LABELS,
    GATHER_RESOURCE_OBSERVATION_LABELS,
    POLITES_VISION_RADIUS_M,
    build_observation,
)


def test_project_local_observation_recenters_exact_policy_input_on_polites():
    observation = np.array([0.0, 0.0, 0.5, 0.0, 0.25], dtype=np.float32)

    scene = project_local_observation(observation, map_size_m=512.0)

    assert scene.polites_world_m == pytest.approx((256.0, 256.0))
    assert scene.resource_world_m == pytest.approx((384.0, 256.0))
    assert scene.resource_local_m == pytest.approx((128.0, 0.0))
    assert scene.observed_distance_m == pytest.approx(128.0)
    assert scene.physical_distance_m == pytest.approx(128.0)
    assert scene.vision_radius_m == POLITES_VISION_RADIUS_M
    assert scene.visible_resource_local_m is None
    assert scene.normalized_observation == pytest.approx(tuple(observation))
    assert scene.observation_labels == GATHER_OBSERVATION_LABELS


@pytest.mark.parametrize(
    ("distance_m", "is_visible"),
    [(31.999, True), (32.0, True), (32.001, False)],
)
def test_project_local_observation_applies_the_physical_vision_boundary(
    distance_m,
    is_visible,
):
    observation = build_observation(
        (256.0, 256.0),
        (256.0 + distance_m, 256.0),
        512.0,
    )

    scene = project_local_observation(observation, map_size_m=512.0)

    assert scene.physical_distance_m == pytest.approx(distance_m, abs=1e-4)
    assert scene.resource_local_m == pytest.approx((distance_m, 0.0), abs=1e-4)
    assert (scene.visible_resource_local_m is not None) is is_visible


@pytest.mark.parametrize("angle_degrees", [15.0, 45.0, 73.0, 135.0, 225.0])
def test_exact_vision_boundary_is_stable_across_float32_directions(angle_degrees):
    angle = math.radians(angle_degrees)
    resource = (
        256.0 + POLITES_VISION_RADIUS_M * math.cos(angle),
        256.0 + POLITES_VISION_RADIUS_M * math.sin(angle),
    )
    observation = build_observation((256.0, 256.0), resource, 512.0)

    scene = project_local_observation(observation, map_size_m=512.0)

    assert scene.visible_resource_local_m is not None


def test_physical_visibility_uses_geometry_not_the_policy_distance_scalar():
    observation = build_observation(
        (256.0, 256.0),
        (320.0, 256.0),
        512.0,
    )
    observation[4] = 0.0

    scene = project_local_observation(observation, map_size_m=512.0)

    assert scene.observed_distance_m == 0.0
    assert scene.physical_distance_m == pytest.approx(64.0)
    assert scene.visible_resource_local_m is None


def test_policy_readout_discloses_omniscient_coordinates_outside_vision():
    observation = build_observation(
        (256.0, 256.0),
        (320.0, 256.0),
        512.0,
    )
    scene = project_local_observation(observation, map_size_m=512.0)

    readout = format_policy_readout(scene)

    assert "OMNISCIENT" in readout
    assert "renderer frame above is the visibility authority" in readout
    assert "outside 32 m physical vision" not in readout
    for label in GATHER_OBSERVATION_LABELS:
        assert label in readout


def test_project_local_observation_accepts_resource_state_features():
    observation = build_observation(
        (256.0, 256.0),
        (320.0, 256.0),
        512.0,
        carried_resource=10.0,
        carried_resource_scale=20.0,
        resource_stock=300.0,
        resource_stock_scale=1000.0,
    )

    scene = project_local_observation(observation, map_size_m=512.0)
    readout = format_policy_readout(scene)

    assert scene.observation_labels == GATHER_RESOURCE_OBSERVATION_LABELS
    assert scene.normalized_observation[-2:] == pytest.approx((0.5, 0.3))
    assert "carried_wood_norm=0.5" in readout
    assert "stock_wood_norm=0.300000012" in readout


def test_physical_status_does_not_leak_out_of_range_distance():
    observation = build_observation(
        (256.0, 256.0),
        (320.0, 256.0),
        512.0,
    )
    scene = project_local_observation(observation, map_size_m=512.0)

    status = format_physical_status(scene)

    assert status == "tree NOT VISIBLE · outside 32 m physical vision"
    assert "64" not in status


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (np.zeros(4, dtype=np.float32), "at least five values"),
        (np.array([0.0, 0.0, np.nan, 0.0, 0.0]), "finite"),
    ],
)
def test_project_local_observation_rejects_invalid_policy_input(
    observation,
    message,
):
    with pytest.raises(ValueError, match=message):
        project_local_observation(observation, map_size_m=512.0)


@pytest.mark.parametrize("map_size_m", [0.0, -1.0, np.inf])
def test_project_local_observation_requires_a_finite_positive_map(map_size_m):
    with pytest.raises(ValueError, match="map_size_m must be finite and positive"):
        project_local_observation(np.zeros(5, dtype=np.float32), map_size_m)


@pytest.mark.parametrize("vision_radius_m", [0.0, -1.0, np.inf])
def test_project_local_observation_requires_a_finite_positive_vision_radius(
    vision_radius_m,
):
    with pytest.raises(ValueError, match="vision_radius_m must be finite and positive"):
        project_local_observation(
            np.zeros(5, dtype=np.float32),
            map_size_m=512.0,
            vision_radius_m=vision_radius_m,
        )


def test_open_agent_view_rejects_non_gather_observations_before_opening_tk():
    class UnsupportedEnv:
        observation_labels = ("other",)
        map_size_m = 512.0

    with pytest.raises(AgentViewUnavailable, match="starting with the five geometry"):
        open_agent_view(UnsupportedEnv())


@pytest.mark.parametrize("map_size_m", [None, 0.0, np.inf])
def test_open_agent_view_rejects_invalid_map_size_before_opening_tk(map_size_m):
    class InvalidMapEnv:
        observation_labels = GATHER_OBSERVATION_LABELS

    env = InvalidMapEnv()
    env.map_size_m = map_size_m

    with pytest.raises(AgentViewUnavailable, match="valid map_size_m"):
        open_agent_view(env)


def test_open_agent_view_preflights_engine_and_wires_frame_capture(monkeypatch):
    events = []

    class Observer:
        def check_available(self):
            events.append("preflight")

    class GatherEnv:
        observation_labels = GATHER_OBSERVATION_LABELS
        map_size_m = 512.0
        engine_observer = Observer()

        def capture_agent_frame(self):
            events.append("capture")

    fake_tk = ModuleType("tkinter")
    fake_tk.TclError = RuntimeError
    fake_tk.Tk = lambda: events.append("tk") or object()
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    def make_view(_tk, _root, _map_size, _radius, frame_provider):
        events.append("view")
        frame_provider()
        return object()

    monkeypatch.setattr("rl.gather.agent_view._TkGatherAgentView", make_view)

    open_agent_view(GatherEnv())

    assert events == ["preflight", "tk", "view", "capture"]


def test_tk_view_draws_only_the_engine_ppm():
    calls = []

    class Tk:
        @staticmethod
        def PhotoImage(*, data, format):
            calls.append(("photo", data, format))
            return "engine-photo"

    class Canvas:
        def delete(self, tag):
            calls.append(("delete", tag))

        def create_image(self, x, y, **kwargs):
            calls.append(("image", x, y, kwargs))

        def create_text(self, x, y, **kwargs):
            calls.append(("text", x, y, kwargs))

    view = _TkGatherAgentView.__new__(_TkGatherAgentView)
    view._tk = Tk()
    view._canvas = Canvas()
    frame = EngineObserverFrame(1, 1, b"P6\n1 1\n255\n\x00\x00\x00")

    view._draw_engine_frame(frame)

    assert view.WINDOW_WIDTH == view.CANVAS_SIZE == 512
    assert view.WINDOW_HEIGHT == view.CANVAS_SIZE
    assert view._engine_photo == "engine-photo"
    assert calls == [
        ("delete", "all"),
        ("photo", frame.ppm, "PPM"),
        (
            "image",
            view.CANVAS_SIZE / 2,
            view.CANVAS_SIZE / 2,
            {"image": "engine-photo"},
        ),
    ]


def test_tk_view_waiting_canvas_has_no_text():
    calls = []

    class Canvas:
        def delete(self, tag):
            calls.append(("delete", tag))

        def create_text(self, *_args, **_kwargs):
            calls.append(("text",))

    view = _TkGatherAgentView.__new__(_TkGatherAgentView)
    view._canvas = Canvas()

    view._draw_waiting_frame()

    assert calls == [("delete", "all")]
