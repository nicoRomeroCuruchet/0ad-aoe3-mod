import numpy as np
from rl.gather.core import xz, distance, normalize_coord, denormalize_action
from rl.gather.core import build_observation, gather_reward, is_reached
from rl.gather.core import normalize_non_negative
from rl.gather.core import stock_delta_reward

def test_xz_passes_through_ground_plane():
    # position() de 0 A.D. ya devuelve [x, z] en metros (2 elementos).
    assert xz([176, 256]) == (176.0, 256.0)

def test_distance_is_euclidean_on_xz():
    assert distance((0.0, 0.0), (3.0, 4.0)) == 5.0

def test_normalize_coord_maps_to_minus_one_one():
    assert normalize_coord(0.0, 512.0) == -1.0
    assert normalize_coord(512.0, 512.0) == 1.0
    assert normalize_coord(256.0, 512.0) == 0.0

def test_denormalize_action_inverts_normalization():
    assert denormalize_action([-1.0, 1.0], 512.0) == (0.0, 512.0)
    assert denormalize_action([0.0, 0.0], 512.0) == (256.0, 256.0)

def test_build_observation_shape_and_values():
    obs = build_observation((256.0, 256.0), (256.0, 256.0), 512.0)
    assert obs.dtype == np.float32
    assert obs.shape == (5,)
    assert np.allclose(obs, [0.0, 0.0, 0.0, 0.0, 0.0])

def test_build_observation_can_include_resource_state():
    obs = build_observation(
        (256.0, 256.0),
        (256.0, 256.0),
        512.0,
        carried_resource=10.0,
        carried_resource_scale=20.0,
        resource_stock=300.0,
        resource_stock_scale=1000.0,
    )
    assert obs.dtype == np.float32
    assert obs.shape == (7,)
    assert np.allclose(obs, [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3])

def test_build_observation_can_include_return_lifecycle_state():
    obs = build_observation(
        (256.0, 256.0),
        (384.0, 256.0),
        512.0,
        carried_resource=20.0,
        carried_resource_scale=20.0,
        resource_stock=300.0,
        resource_stock_scale=1000.0,
        dropsite_xz=(128.0, 384.0),
        gather_cycle_active=True,
    )

    assert obs.dtype == np.float32
    assert obs.shape == (10,)
    assert np.allclose(
        obs,
        [0.0, 0.0, 0.5, 0.0, 0.25, 1.0, 0.3, -0.5, 0.5, 1.0],
    )

def test_normalize_non_negative_clips_to_unit_interval():
    assert normalize_non_negative(0.0, 20.0) == 0.0
    assert normalize_non_negative(10.0, 20.0) == 0.5
    assert normalize_non_negative(30.0, 20.0) == 1.0

def test_gather_reward_is_positive_when_getting_closer():
    assert gather_reward(10.0, 4.0) == 6.0
    assert gather_reward(4.0, 10.0) == -6.0

def test_stock_delta_reward_uses_real_stock_delta():
    assert stock_delta_reward(10.0, 14.5) == 4.5
    assert stock_delta_reward(14.5, 14.0) == -0.5

def test_is_reached_uses_threshold():
    assert is_reached(3.0, 4.0) is True
    assert is_reached(5.0, 4.0) is False


def test_nearest_index_picks_the_closest_candidate():
    from rl.gather.core import nearest_index

    candidates = [(10.0, 0.0), (1.0, 1.0), (5.0, 5.0)]

    assert nearest_index((0.0, 0.0), candidates) == 1


def test_nearest_index_breaks_ties_by_the_lowest_index():
    from rl.gather.core import nearest_index

    candidates = [(3.0, 0.0), (0.0, 3.0)]

    assert nearest_index((0.0, 0.0), candidates) == 0


def test_nearest_index_rejects_an_empty_candidate_list():
    import pytest

    from rl.gather.core import nearest_index

    with pytest.raises(ValueError, match="at least one candidate"):
        nearest_index((0.0, 0.0), [])
