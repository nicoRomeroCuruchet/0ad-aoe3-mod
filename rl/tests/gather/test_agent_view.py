import numpy as np
import pytest

from rl.gather.agent_view import (
    AgentViewUnavailable,
    open_agent_view,
    project_local_observation,
)
from rl.gather.core import GATHER_OBSERVATION_LABELS


def test_project_local_observation_recenters_exact_policy_input_on_polites():
    observation = np.array([0.0, 0.0, 0.5, 0.0, 0.25], dtype=np.float32)

    scene = project_local_observation(observation, map_size_m=512.0)

    assert scene.polites_world_m == pytest.approx((256.0, 256.0))
    assert scene.resource_world_m == pytest.approx((384.0, 256.0))
    assert scene.resource_local_m == pytest.approx((128.0, 0.0))
    assert scene.observed_distance_m == pytest.approx(128.0)
    assert scene.normalized_observation == pytest.approx(tuple(observation))


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
