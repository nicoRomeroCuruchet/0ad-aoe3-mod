import numpy as np
from rl.gather.core import xz, distance, normalize_coord, denormalize_action

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
