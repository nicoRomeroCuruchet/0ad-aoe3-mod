import numpy as np


def xz(pos):
    """0 A.D. position() ya devuelve [x, z] en metros (2 elementos)."""
    return (float(pos[0]), float(pos[1]))


def distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def normalize_coord(value, map_size_m):
    """metros [0, map_size_m] -> [-1, 1]."""
    return 2.0 * value / map_size_m - 1.0


def denormalize_action(action, map_size_m):
    """accion [-1, 1] -> metros [0, map_size_m]; devuelve (x, z)."""
    x = (float(action[0]) + 1.0) * 0.5 * map_size_m
    z = (float(action[1]) + 1.0) * 0.5 * map_size_m
    return (x, z)
