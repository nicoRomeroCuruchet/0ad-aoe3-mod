import numpy as np
import pytest
from gymnasium import spaces

from rl.gather.assignment_actions import (
    assignment_action_space,
    assignment_to_raw_click,
)


def test_assignment_action_space_has_one_category_per_resource_plus_no_click():
    action_space = assignment_action_space(villager_count=3, resource_count=4)

    assert isinstance(action_space, spaces.MultiDiscrete)
    assert action_space.shape == (3,)
    np.testing.assert_array_equal(action_space.nvec, np.array([5, 5, 5]))
    assert action_space.dtype == np.dtype(np.int64)


@pytest.mark.parametrize(
    ("villager_count", "resource_count", "message"),
    [
        (0, 2, "villager_count must be a positive integer"),
        (-1, 2, "villager_count must be a positive integer"),
        (True, 2, "villager_count must be a positive integer"),
        (2, 0, "resource_count must be a positive integer"),
        (2, -1, "resource_count must be a positive integer"),
        (2, 1.5, "resource_count must be a positive integer"),
    ],
)
def test_assignment_action_space_rejects_invalid_counts(
    villager_count, resource_count, message
):
    with pytest.raises(ValueError, match=message):
        assignment_action_space(
            villager_count=villager_count,
            resource_count=resource_count,
        )


def test_assignment_decoder_maps_slots_to_exact_raw_click_coordinates():
    assignments = np.array([0, 2, 1], dtype=np.int64)
    resource_xz = np.array(
        [
            [10.0, 25.0],
            [75.0, 100.0],
        ],
        dtype=np.float64,
    )

    decoded = assignment_to_raw_click(
        assignments,
        resource_xz,
        map_size_m=100.0,
    )

    assert decoded.dtype == np.float32
    assert decoded.shape == (9,)
    np.testing.assert_allclose(
        decoded,
        np.array(
            [
                0.0,
                0.0,
                -1.0,
                0.5,
                1.0,
                1.0,
                -0.8,
                -0.5,
                1.0,
            ],
            dtype=np.float32,
        ),
    )


def test_assignment_decoder_does_not_mutate_its_inputs():
    assignments = np.array([2, 0], dtype=np.int32)
    resource_xz = np.array([[5.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    assignments_before = assignments.copy()
    resource_xz_before = resource_xz.copy()

    assignment_to_raw_click(assignments, resource_xz, map_size_m=40.0)

    np.testing.assert_array_equal(assignments, assignments_before)
    np.testing.assert_array_equal(resource_xz, resource_xz_before)


@pytest.mark.parametrize(
    "assignments",
    [
        np.array(1, dtype=np.int64),
        np.array([[1], [0]], dtype=np.int64),
    ],
    ids=["scalar", "matrix"],
)
def test_assignment_decoder_rejects_non_vector_actions(assignments):
    with pytest.raises(ValueError, match="assignments must be a one-dimensional array"):
        assignment_to_raw_click(
            assignments,
            np.array([[5.0, 10.0]], dtype=np.float32),
            map_size_m=20.0,
        )


def test_assignment_decoder_rejects_non_integer_actions():
    with pytest.raises(ValueError, match="assignments must have an integer dtype"):
        assignment_to_raw_click(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([[5.0, 10.0]], dtype=np.float32),
            map_size_m=20.0,
        )


@pytest.mark.parametrize(
    "assignments",
    [
        np.array([-1, 0], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
    ],
    ids=["negative", "past-last-resource"],
)
def test_assignment_decoder_rejects_categories_outside_available_slots(assignments):
    with pytest.raises(
        ValueError, match="assignment categories must be between 0 and 2"
    ):
        assignment_to_raw_click(
            assignments,
            np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
            map_size_m=20.0,
        )


@pytest.mark.parametrize(
    ("resource_xz", "message"),
    [
        (np.empty((0, 2), dtype=np.float32), "at least one resource"),
        (np.array([1.0, 2.0]), "resource_xz must have shape"),
        (np.ones((2, 3)), "resource_xz must have shape"),
        (np.array([[np.nan, 2.0]]), "resource coordinates must be finite"),
        (np.array([[-0.1, 2.0]]), "resource coordinates must lie within the map"),
        (np.array([[2.0, 20.1]]), "resource coordinates must lie within the map"),
    ],
    ids=[
        "empty",
        "flat",
        "wrong-width",
        "non-finite",
        "negative",
        "past-map-edge",
    ],
)
def test_assignment_decoder_rejects_invalid_resource_coordinates(resource_xz, message):
    with pytest.raises(ValueError, match=message):
        assignment_to_raw_click(
            np.array([0], dtype=np.int64),
            resource_xz,
            map_size_m=20.0,
        )


@pytest.mark.parametrize("map_size_m", [0.0, -1.0, np.nan, np.inf])
def test_assignment_decoder_rejects_invalid_map_size(map_size_m):
    with pytest.raises(ValueError, match="map_size_m must be finite and positive"):
        assignment_to_raw_click(
            np.array([1], dtype=np.int64),
            np.array([[5.0, 10.0]], dtype=np.float32),
            map_size_m=map_size_m,
        )
