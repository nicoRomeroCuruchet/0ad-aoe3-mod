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


def build_observation(villager_xz, resource_xz, map_size_m):
    d = distance(villager_xz, resource_xz)
    return np.array([
        normalize_coord(villager_xz[0], map_size_m),
        normalize_coord(villager_xz[1], map_size_m),
        normalize_coord(resource_xz[0], map_size_m),
        normalize_coord(resource_xz[1], map_size_m),
        d / map_size_m,
    ], dtype=np.float32)


def gather_reward(prev_dist, cur_dist):
    return float(prev_dist - cur_dist)


def is_reached(cur_dist, threshold):
    return bool(cur_dist < threshold)
