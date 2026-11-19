"""Sparse sampling of a Cartesian lattice in the input surface band."""
from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np

from .deadline import checkpoint
from .schemas import require


@dataclass
class GridSamples:
    blocks: np.ndarray
    keys: np.ndarray
    points: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    block_indices: np.ndarray
    report: dict


def query(provider, points, batch_size):
    """Keep validity per channel. A V failure must not discard valid E."""
    parts = []
    masks = []
    for start in range(0, len(points), batch_size):
        checkpoint('edge_first_query')
        result = provider.evaluate(points[start:start + batch_size])
        parts.append(result.values)
        masks.append(result.valid & ~result.saturated & np.isfinite(result.values))
    if not parts:
        return np.empty((0, 3)), np.empty((0, 3), bool)
    return np.vstack(parts), np.vstack(masks)


def candidate_blocks(mesh, width, spacing, budget):
    """Conservatively cover the band without a dense full-volume buffer."""
    from .mesh_densification import coarse_refine_surface

    # Split long surface triangles before rasterizing padded bounding boxes.
    # This avoids the enormous AABB of a large, obliquely oriented triangle.
    proxy, _, _ = coarse_refine_surface(mesh, 3 * spacing)
    triangles = proxy.vertices[proxy.triangles]
    lower = np.floor((triangles.min(1) - width) / spacing).astype(int)
    upper = np.floor((triangles.max(1) + width) / spacing).astype(int)
    legal_lo = int(np.floor(-1 / spacing))
    legal_hi = int(np.floor(1 / spacing))
    lower = np.maximum(lower, legal_lo)
    upper = np.minimum(upper, legal_hi)
    blocks = set()
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if i % 1024 == 0:
            checkpoint('edge_first_blocks')
        blocks.update(product(*(range(a, b + 1) for a, b in zip(lo, hi))))
        require(len(blocks) <= budget, 'scout_block_budget', 'Too many candidate blocks',
                blocks=len(blocks), budget=budget)
    return np.asarray(sorted(blocks), dtype=np.int32).reshape(-1, 3)


def sample_grid(mesh, provider, cfg):
    started = perf_counter()
    H, h = cfg.coarse_spacing, cfg.fine_spacing
    factor = round(H / h)
    if provider.presence()[0] == 'absent':
        return GridSamples(np.empty((0, 3), int), np.empty((0, 3), int), np.empty((0, 3)),
                           np.empty((0, 3)), np.empty((0, 3), bool),
                           np.empty((0, factor + 1, factor + 1, factor + 1), int),
                           dict(edge_field_absent=True, fine_points=0, selected_blocks=0))
    offsets = np.array(list(product(range(factor + 1), repeat=3)), np.int32)
    blocks = candidate_blocks(mesh, provider.band_width, H, cfg.scout_point_budget)
    centers = (blocks + .5) * H
    radius = np.sqrt(3) * H / 2
    surface_distance = np.abs(provider.sdf(centers))
    near = surface_distance <= provider.band_width + radius
    blocks, centers = blocks[near], centers[near]
    coarse_start = perf_counter()
    inner = surface_distance[near] <= min(.015, provider.band_width)
    values = np.zeros((len(centers), 3))
    valid = np.zeros((len(centers), 3), bool)
    for selection in (inner, ~inner):
        values[selection], valid[selection] = query(provider, centers[selection], cfg.query_batch_size)
    threshold = cfg.shell_radius + radius + cfg.coarse_error_margin
    keep = (valid[:, 1] & (values[:, 1] <= threshold))
    if provider.presence()[1] == 'present':
        keep |= valid[:, 2] & (values[:, 2] <= threshold)
    unknown = np.flatnonzero(~valid[:, 1])
    # Scout the outer band with corners when the center is outside it.
    corners = np.array(list(product((0, factor), repeat=3)), np.int32)
    outer_keys, outer_inverse = np.unique(
        (blocks[unknown, None, :] * factor + corners).reshape(-1, 3),
        axis=0, return_inverse=True,
    )
    outer_values, outer_valid = query(provider, outer_keys * h, cfg.query_batch_size)
    ii = outer_inverse.reshape(-1, 8)
    # A valid corner's cover radius is the full coarse diagonal.
    corner_keep = outer_valid[ii, 1] & (outer_values[ii, 1] <= threshold + radius)
    if provider.presence()[1] == 'present':
        corner_keep |= outer_valid[ii, 2] & (outer_values[ii, 2] <= threshold + radius)
    keep[unknown] |= corner_keep.any(axis=1) | ~outer_valid[ii, 1].any(axis=1)
    fine_start = perf_counter()
    selected = blocks[keep]
    total_corners = len(selected) * len(offsets)
    require(total_corners <= cfg.scout_point_budget, 'scout_corner_budget',
            'Fine grid allocation exceeds the configured budget', corners=total_corners,
            budget=cfg.scout_point_budget, selected_blocks=len(selected))
    keys, inverse = np.unique(
        (selected[:, None, :] * factor + offsets[None, :, :]).reshape(-1, 3),
        axis=0, return_inverse=True,
    )
    points = keys * h
    fine_values = np.zeros((len(points), 3))
    fine_valid = np.zeros((len(points), 3), bool)
    index = {tuple(key): i for i, key in enumerate(outer_keys)}
    if factor % 2 == 0:
        for i, key in enumerate(blocks * factor + factor // 2, len(outer_keys)):
            index[tuple(key)] = i
    reuse = np.array([index.get(tuple(key), -1) for key in keys], dtype=int)
    values_pool = np.vstack((outer_values, values))
    valid_pool = np.vstack((outer_valid, valid))
    known = reuse >= 0
    fine_values[known] = values_pool[reuse[known]]
    fine_valid[known] = valid_pool[reuse[known]]
    fine_values[~known], fine_valid[~known] = query(provider, points[~known], cfg.query_batch_size)
    report = dict(coarse_blocks=len(blocks), coarse_inner=int(inner.sum()), coarse_valid_e=int(valid[:, 1].sum()),
                  coarse_unknown=len(unknown), outer_corners=len(outer_keys),
                  unknown_after_corner_scout=int((~outer_valid[ii, 1].any(axis=1)).sum()),
                  selected_blocks=len(selected),
                  fine_points=len(points), fine_new_points=int((~known).sum()),
                  fine_valid_e=int(fine_valid[:, 1].sum()), reused_points=int(known.sum()),
                  coarse_e_threshold=threshold,
                  timings=dict(blocks=coarse_start-started, coarse=fine_start-coarse_start,
                               fine=perf_counter()-fine_start))
    return GridSamples(selected, keys, points, fine_values, fine_valid,
                       inverse.reshape(len(selected), factor + 1, factor + 1, factor + 1), report)
