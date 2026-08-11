import numpy as np
import pytest

from rl.gather.core import GATHER_LIFECYCLE_OBSERVATION_LABELS
from rl.gather.observation import (
    TEAM_CORE_LABELS,
    TeamObservationScales,
    TeamSnapshot,
    build_team_observation,
    team_observation_labels,
)


SCALES = TeamObservationScales(
    map_size_m=512.0,
    carried_resource_scale=20.0,
    stock_scale=1000.0,
    resource_amount_scale=200.0,
)


def _snapshot():
    return TeamSnapshot(
        villager_xz=((128.0, 256.0), (384.0, 256.0)),
        resource_xz=((256.0, 256.0), (400.0, 256.0)),
        resource_remaining=(200.0, 100.0),
        carried=(0.0, 20.0),
        target_index=(0, 1),
        gather_cycle_active=(False, True),
        dropsite_xz=(64.0, 320.0),
        stock=300.0,
    )


def test_core_prefix_matches_the_m1_observation_layout():
    assert TEAM_CORE_LABELS == GATHER_LIFECYCLE_OBSERVATION_LABELS


def test_slice_shape_and_labels_follow_the_configured_counts():
    observation = build_team_observation(_snapshot(), SCALES)
    labels = team_observation_labels(2, 2)

    assert observation.shape == (2, 21)
    assert observation.dtype == np.float32
    assert len(labels) == 21
    assert labels[:10] == TEAM_CORE_LABELS
    assert labels[10] == "agent_id_norm"
    assert labels[11:16] == (
        "tree0_dx_norm",
        "tree0_dz_norm",
        "tree0_dist_norm",
        "tree0_is_current_target",
        "tree0_remaining_norm",
    )


def test_core_values_reproduce_the_m1_fields():
    observation = build_team_observation(_snapshot(), SCALES)

    # villager 0 at (128, 256), its target tree 0 at (256, 256), map 512 m.
    assert observation[0][0] == pytest.approx(-0.5)
    assert observation[0][1] == pytest.approx(0.0)
    assert observation[0][2] == pytest.approx(0.0)
    assert observation[0][3] == pytest.approx(0.0)
    assert observation[0][4] == pytest.approx(128.0 / (np.sqrt(2.0) * 512.0))
    assert observation[0][5] == pytest.approx(0.0)
    assert observation[0][6] == pytest.approx(0.3)
    assert observation[0][7] == pytest.approx(-0.75)
    assert observation[0][8] == pytest.approx(0.25)
    assert observation[0][9] == pytest.approx(0.0)
    # villager 1 carries a full load and its cycle is running.
    assert observation[1][5] == pytest.approx(1.0)
    assert observation[1][9] == pytest.approx(1.0)


def test_agent_id_is_normalized_over_the_team():
    observation = build_team_observation(_snapshot(), SCALES)

    assert observation[0][10] == pytest.approx(0.0)
    assert observation[1][10] == pytest.approx(1.0)


def test_single_villager_gets_agent_id_zero():
    snapshot = TeamSnapshot(
        villager_xz=((128.0, 256.0),),
        resource_xz=((256.0, 256.0),),
        resource_remaining=(200.0,),
        carried=(0.0,),
        target_index=(0,),
        gather_cycle_active=(False,),
        dropsite_xz=(64.0, 320.0),
        stock=0.0,
    )

    observation = build_team_observation(snapshot, SCALES)

    assert observation.shape == (1, 16)
    assert observation[0][10] == pytest.approx(0.0)


def test_relational_block_flags_only_the_villagers_current_target():
    observation = build_team_observation(_snapshot(), SCALES)

    assert observation[0][14] == pytest.approx(1.0)
    assert observation[1][14] == pytest.approx(0.0)
    assert observation[0][19] == pytest.approx(0.0)
    assert observation[1][19] == pytest.approx(1.0)


def test_villager_without_an_assignment_has_neutral_target_fields():
    snapshot = TeamSnapshot(
        villager_xz=((128.0, 256.0),),
        resource_xz=((256.0, 256.0),),
        resource_remaining=(200.0,),
        carried=(0.0,),
        target_index=(None,),
        gather_cycle_active=(False,),
        dropsite_xz=(64.0, 320.0),
        stock=0.0,
    )

    observation = build_team_observation(snapshot, SCALES)

    np.testing.assert_array_equal(observation[0, 2:5], np.zeros(3, dtype=np.float32))
    assert observation[0, 14] == pytest.approx(0.0)


def test_relational_block_is_ordered_by_roster_not_by_distance():
    observation = build_team_observation(_snapshot(), SCALES)

    # Villager 1 is nearest tree 1, but tree 0 still occupies the first block.
    assert observation[1][13] == pytest.approx(128.0 / (np.sqrt(2.0) * 512.0))
    assert observation[1][18] == pytest.approx(16.0 / (np.sqrt(2.0) * 512.0))


def test_remaining_is_normalized_and_clipped():
    observation = build_team_observation(_snapshot(), SCALES)

    assert observation[0][15] == pytest.approx(1.0)
    assert observation[0][20] == pytest.approx(0.5)


def test_corner_to_corner_distances_fit_the_declared_unit_interval():
    snapshot = TeamSnapshot(
        villager_xz=((0.0, 0.0),),
        resource_xz=((512.0, 512.0),),
        resource_remaining=(200.0,),
        carried=(0.0,),
        target_index=(0,),
        gather_cycle_active=(True,),
        dropsite_xz=(0.0, 0.0),
        stock=0.0,
    )

    observation = build_team_observation(snapshot, SCALES)

    assert observation[0, 4] == pytest.approx(1.0)
    assert observation[0, 13] == pytest.approx(1.0)
    assert np.all(observation <= 1.0)


def test_snapshot_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="per-villager fields must have equal length"):
        TeamSnapshot(
            villager_xz=((0.0, 0.0), (1.0, 1.0)),
            resource_xz=((2.0, 2.0),),
            resource_remaining=(10.0,),
            carried=(0.0,),
            target_index=(0, 0),
            gather_cycle_active=(False, False),
            dropsite_xz=(0.0, 0.0),
            stock=0.0,
        )


def test_snapshot_rejects_a_target_index_outside_the_roster():
    with pytest.raises(ValueError, match="target_index out of range"):
        TeamSnapshot(
            villager_xz=((0.0, 0.0),),
            resource_xz=((2.0, 2.0),),
            resource_remaining=(10.0,),
            carried=(0.0,),
            target_index=(3,),
            gather_cycle_active=(False,),
            dropsite_xz=(0.0, 0.0),
            stock=0.0,
        )
