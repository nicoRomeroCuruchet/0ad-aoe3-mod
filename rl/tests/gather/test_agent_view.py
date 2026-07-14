import math

import numpy as np
import pytest

from rl.gather.agent_view import (
    AgentViewUnavailable,
    format_physical_status,
    format_policy_readout,
    open_agent_view,
    project_local_observation,
)
from rl.gather.core import (
    GATHER_OBSERVATION_LABELS,
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
    assert "outside 32 m physical vision" in readout
    for label in GATHER_OBSERVATION_LABELS:
        assert label in readout


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
        (np.zeros(4, dtype=np.float32), "exactly five values"),
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

    with pytest.raises(AgentViewUnavailable, match="only the five-value M0"):
        open_agent_view(UnsupportedEnv())


@pytest.mark.parametrize("map_size_m", [None, 0.0, np.inf])
def test_open_agent_view_rejects_invalid_map_size_before_opening_tk(map_size_m):
    class InvalidMapEnv:
        observation_labels = GATHER_OBSERVATION_LABELS

    env = InvalidMapEnv()
    env.map_size_m = map_size_m

    with pytest.raises(AgentViewUnavailable, match="valid map_size_m"):
        open_agent_view(env)
