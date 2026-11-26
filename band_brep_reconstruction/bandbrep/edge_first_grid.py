"""Sparse dyadic sampling of a Cartesian lattice in the input surface band (8D joint field)."""
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
    values: np.ndarray      # [N, 3]: [sdf, dE, dV]
    offsets: np.ndarray     # [N, 2, 3]: [uE, uV]
    valid: np.ndarray       # [N, 3]
    report: dict
    origin: np.ndarray | None = None


def query(provider, points, batch_size):
    """Keep validity per channel. A V failure must not discard valid E."""
    parts = []
    offset_parts = []
    masks = []
    for start in range(0, len(points), batch_size):
        checkpoint('edge_first_query')
        result = provider.evaluate(points[start:start + batch_size])
        parts.append(result.values)
        offset_parts.append(result.offsets)
        # Clipped far-field distances (saturated & values > 0, e.g. d >= udf_clip=0.05) are valid
        # lower bounds for block pruning; only negative/nonfinite values are invalid.
        masks.append(
            result.valid & (~result.saturated | (result.values > 0)) & np.isfinite(result.values)
        )
    if not parts:
        return np.empty((0, 3)), np.empty((0, 2, 3)), np.empty((0, 3), bool)
    return np.vstack(parts), np.vstack(offset_parts), np.vstack(masks)


def candidate_blocks(mesh, width, spacing, budget):
    """Conservatively cover the band without a dense full-volume buffer."""
    from .mesh_densification import coarse_refine_surface

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
    """Query nested dyadic grids (e.g. 0.024 -> 0.012 -> 0.006 -> 0.003) for the 8D field."""
    started = perf_counter()
    spacings = cfg.scout_spacings()
    h = cfg.fine_spacing
    H = spacings[0]
    last_coarse = spacings[-2] if len(spacings) > 1 else h
    factor = round(last_coarse / h)
    empty = GridSamples(
        np.empty((0, 3), int), np.empty((0, 3), int), np.empty((0, 3)),
        np.empty((0, 3)), np.empty((0, 2, 3)), np.empty((0, 3), bool),
        dict(fine_points=0, selected_blocks=0),
    )
    if provider.presence()[0] == 'absent':
        return empty

    bound = int(np.ceil(2 / h)) + 2 * round(H / h) + 2
    base = 2 * bound + 1
    require(base**3 < 2**63, 'grid_key_range', 'Fine grid exceeds integer key range')

    def encode(keys):
        x, y, z = (keys.astype(np.int64) + bound).T
        return (x * base + y) * base + z

    def decode(codes):
        return np.column_stack((codes // base**2, codes // base % base, codes % base)) - bound

    codes = np.empty(0, np.int64)
    values = np.empty((0, 3))
    offsets = np.empty((0, 2, 3))
    valid = np.empty((0, 3), bool)

    def observe(keys):
        nonlocal codes, values, offsets, valid
        request, inverse = np.unique(encode(keys), return_inverse=True)
        lookup = np.searchsorted(codes, request)
        missing = (lookup == len(codes))
        missing[~missing] = codes[lookup[~missing]] != request[~missing]
        if missing.any():
            new = request[missing]
            require(len(codes) + len(new) <= cfg.scout_point_budget, 'scout_point_budget',
                    'Unique query budget exceeded', unique=len(codes) + len(new),
                    budget=cfg.scout_point_budget)
            v, off, ok = query(provider, decode(new) * (h * 0.5), cfg.query_batch_size)
            codes = np.r_[codes, new]
            order = np.argsort(codes)
            codes = codes[order]
            values = np.vstack((values, v))[order]
            offsets = np.vstack((offsets, off))[order]
            valid = np.vstack((valid, ok))[order]
        at = np.searchsorted(codes, request)[inverse]
        return values[at], valid[at]

    blocks = candidate_blocks(mesh, provider.band_width, H, cfg.scout_point_budget)
    levels = []
    has_vertex = provider.presence()[1] == 'present'
    aux_points = 0

    def scout(blocks, spacing):
        nonlocal aux_points
        start = perf_counter()
        stride = round(spacing / h)
        center = (2 * blocks + 1) * stride
        dist = np.abs(provider.sdf(center * (h * 0.5)))
        radius = np.sqrt(3) * spacing * 0.5
        near = dist <= provider.band_width + radius
        blocks, center, dist = blocks[near], center[near], dist[near]
        order = np.argsort(dist > 0.015, kind='stable')
        blocks, center = blocks[order], center[order]
        old_count, old_aux = len(codes), aux_points
        v, ok = observe(center)
        margins = np.array([cfg.scout_margin, cfg.scout_margin]) + cfg.coarse_error_margin
        channels = 2 if has_vertex else 1
        keep = (ok[:, 1:1 + channels] & (v[:, 1:1 + channels] <= margins[:channels] + radius)).any(1)
        unknown = np.flatnonzero(~ok[:, 1:1 + channels].all(1))
        corners = np.array(list(product((0, 2 * stride), repeat=3)), int)
        for start_row in range(0, len(unknown), 32768):
            ii = unknown[start_row:start_row + 32768]
            cv, valid_corner = observe((2 * blocks[ii, None, :] * stride + corners).reshape(-1, 3))
            cv = cv[:, 1:1 + channels].reshape(-1, 8, channels)
            valid_corner = valid_corner[:, 1:1 + channels].reshape(-1, 8, channels)
            possible = ~(valid_corner & (cv > margins[:channels] + 2 * radius)).any(axis=1)
            unsupported = ~valid_corner.any(axis=(1, 2))
            if unsupported.any():
                xyz = center[ii[unsupported]] * (h * 0.5)
                squared, _, closest = provider.oracle.closest(xyz)
                unique, index = np.unique(closest, axis=0, return_inverse=True)
                pv, _, valid_probe = query(provider, unique, cfg.query_batch_size)
                aux_points += len(unique)
                bound_val = margins[:channels] + radius + np.sqrt(squared)[:, None]
                possible[unsupported] &= ~(valid_probe[index, 1:1 + channels]
                                           & (pv[index, 1:1 + channels] > bound_val))
            keep[ii] |= possible.any(axis=1)
        levels.append(dict(spacing=spacing, blocks=len(blocks), kept=int(keep.sum()),
                           unknown_centers=len(unknown), new_points=len(codes) - old_count,
                           surface_probes=aux_points - old_aux,
                           seconds=perf_counter() - start))
        return blocks[keep]

    selected = scout(blocks, H)
    for prev_s, next_s in zip(spacings[:-1], spacings[1:-1]):
        if not len(selected):
            break
        ratio = round(prev_s / next_s)
        ensure = len(selected) * ratio**3
        require(ensure <= cfg.scout_point_budget, 'scout_block_budget',
                'Too many intermediate grid blocks', blocks=ensure)
        offsets_blk = np.array(list(product(range(ratio), repeat=3)))
        selected = scout((selected[:, None, :] * ratio + offsets_blk).reshape(-1, 3), next_s)

    if not len(selected):
        empty.report.update(levels=levels, unique_query_points=len(codes), surface_probes=aux_points)
        return empty

    corner_offsets = np.array(list(product(range(factor + 1), repeat=3)))
    start = perf_counter()
    old_count = len(codes)
    total = len(selected) * len(corner_offsets)
    require(total <= 16 * cfg.scout_point_budget, 'scout_index_budget',
            'Too many fine block index entries', corners=total)
    index_codes = np.empty((len(selected), len(corner_offsets)), dtype=np.int64)
    for i in range(0, len(selected), 4096):
        local = (selected[i:i + 4096, None, :] * factor + corner_offsets).reshape(-1, 3)
        index_codes[i:i + 4096] = encode(2 * local).reshape(-1, len(corner_offsets))
    fine_codes = np.unique(index_codes)
    lookup = np.searchsorted(codes, fine_codes)
    missing = (lookup == len(codes))
    missing[~missing] = codes[lookup[~missing]] != fine_codes[~missing]
    if missing.any():
        new = fine_codes[missing]
        require(len(codes) + len(new) <= cfg.scout_point_budget, 'scout_point_budget',
                'Unique query budget exceeded', unique=len(codes) + len(new),
                budget=cfg.scout_point_budget)
        v, off, ok = query(provider, decode(new) * (h * 0.5), cfg.query_batch_size)
        codes = np.r_[codes, new]
        order = np.argsort(codes)
        codes = codes[order]
        values = np.vstack((values, v))[order]
        offsets = np.vstack((offsets, off))[order]
        valid = np.vstack((valid, ok))[order]
    at = np.searchsorted(codes, fine_codes)
    keys = decode(fine_codes) // 2
    report = dict(
        levels=levels,
        selected_blocks=len(selected),
        fine_points=len(keys),
        fine_new_points=len(codes) - old_count,
        unique_query_points=len(codes),
        surface_probes=aux_points,
        fine_valid_e=int(valid[at, 1].sum()),
        fine_seconds=perf_counter() - start,
        elapsed_seconds=perf_counter() - started,
    )
    return GridSamples(selected, keys, keys * h, values[at], offsets[at], valid[at], report)
