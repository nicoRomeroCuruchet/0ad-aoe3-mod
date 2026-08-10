import numpy as np


GATHER_OBSERVATION_LABELS = (
    "villager_x_norm",
    "villager_z_norm",
    "resource_x_norm",
    "resource_z_norm",
    "distance_norm",
)

GATHER_RESOURCE_OBSERVATION_LABELS = (
    *GATHER_OBSERVATION_LABELS,
    "carried_wood_norm",
    "stock_wood_norm",
)

# Mirrors Vision/Range inherited by units/athenai/polites from the 0 A.D.
# public template template_unit_support_female_citizen.
POLITES_VISION_RADIUS_M = 32.0


def xz(pos):
    """0 A.D. position() ya devuelve [x, z] en metros (2 elementos)."""
    return (float(pos[0]), float(pos[1]))


def distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def normalize_coord(value, map_size_m):
    """metros [0, map_size_m] -> [-1, 1]."""
    return 2.0 * value / map_size_m - 1.0


def normalize_non_negative(value, scale):
    """Non-negative scalar -> [0, 1], clipped at the configured scale."""
    parsed_scale = float(scale)
    if parsed_scale <= 0.0:
        raise ValueError("scale must be positive")
    return float(np.clip(float(value) / parsed_scale, 0.0, 1.0))


def denormalize_action(action, map_size_m):
    """accion [-1, 1] -> metros [0, map_size_m]; devuelve (x, z)."""
    x = (float(action[0]) + 1.0) * 0.5 * map_size_m
    z = (float(action[1]) + 1.0) * 0.5 * map_size_m
    return (x, z)


def build_observation(
    villager_xz,
    resource_xz,
    map_size_m,
    *,
    carried_resource=None,
    carried_resource_scale=1.0,
    resource_stock=None,
    resource_stock_scale=1.0,
):
    d = distance(villager_xz, resource_xz)
    values = [
        normalize_coord(villager_xz[0], map_size_m),
        normalize_coord(villager_xz[1], map_size_m),
        normalize_coord(resource_xz[0], map_size_m),
        normalize_coord(resource_xz[1], map_size_m),
        d / map_size_m,
    ]
    if carried_resource is not None or resource_stock is not None:
        if carried_resource is None or resource_stock is None:
            raise ValueError("carried_resource and resource_stock must be provided together")
        values.extend(
            [
                normalize_non_negative(carried_resource, carried_resource_scale),
                normalize_non_negative(resource_stock, resource_stock_scale),
            ]
        )
    return np.array(values, dtype=np.float32)


def gather_reward(prev_dist, cur_dist):
    return float(prev_dist - cur_dist)


def stock_delta_reward(prev_stock, cur_stock):
    return float(cur_stock - prev_stock)


def is_reached(cur_dist, threshold):
    return bool(cur_dist < threshold)
