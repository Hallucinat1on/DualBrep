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
    """Query nested grids. Half-fine integer keys also represent block centers."""
    started = perf_counter()
    h = cfg.fine_spacing
    H, M = cfg.coarse_spacing, cfg.middle_spacing
    factor = round(M / h)
    shape = (factor + 1,) * 3
    empty = GridSamples(np.empty((0, 3), int), np.empty((0, 3), int), np.empty((0, 3)),
                        np.empty((0, 3)), np.empty((0, 3), bool),
                        np.empty((0, *shape), int), dict(fine_points=0, selected_blocks=0))
    if provider.presence()[0] == 'absent':
        return empty

    # Pack grid keys to run one fast unique/search operation per batch.
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
    valid = np.empty((0, 3), bool)

    def observe(keys):
        nonlocal codes, values, valid
        request, inverse = np.unique(encode(keys), return_inverse=True)
        lookup = np.searchsorted(codes, request)
        missing = (lookup == len(codes))
        missing[~missing] = codes[lookup[~missing]] != request[~missing]
        if missing.any():
            new = request[missing]
            require(len(codes) + len(new) <= cfg.scout_point_budget, 'scout_point_budget',
                    'Unique query budget exceeded', unique=len(codes)+len(new),
                    budget=cfg.scout_point_budget)
            v, ok = query(provider, decode(new) * (h * .5), cfg.query_batch_size)
            codes = np.r_[codes, new]
            order = np.argsort(codes)
            codes, values = codes[order], np.vstack((values, v))[order]
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
        dist = np.abs(provider.sdf(center * (h * .5)))
        radius = np.sqrt(3) * spacing * .5
        near = dist <= provider.band_width + radius
        blocks, center, dist = blocks[near], center[near], dist[near]
        order = np.argsort(dist > .015, kind='stable')
        blocks, center = blocks[order], center[order]
        old_count, old_aux = len(codes), aux_points
        v, ok = observe(center)
        margins = np.array([cfg.shell_radius, cfg.vertex_level]) + cfg.coarse_error_margin
        channels = 2 if has_vertex else 1
        keep = (ok[:, 1:1+channels] & (v[:, 1:1+channels] <= margins[:channels] + radius)).any(1)
        unknown = np.flatnonzero(~ok[:, 1:1+channels].all(1))
        corners = np.array(list(product((0, 2*stride), repeat=3)), int)
        for start_row in range(0, len(unknown), 32768):
            ii = unknown[start_row:start_row+32768]
            cv, valid_corner = observe((2*blocks[ii, None, :]*stride + corners).reshape(-1, 3))
            cv = cv[:, 1:1+channels].reshape(-1, 8, channels)
            valid_corner = valid_corner[:, 1:1+channels].reshape(-1, 8, channels)
            possible = ~(valid_corner & (cv > margins[:channels] + 2*radius)).any(axis=1)
            # A block touching the valid band can have all eight corner probes
            # outside it. A nearest-surface observation still provides a
            # conservative exclusion bound, including its distance to center.
            unsupported = ~valid_corner.any(axis=(1,2))
            if unsupported.any():
                xyz = center[ii[unsupported]]*(h*.5)
                squared, _, closest = provider.oracle.closest(xyz)
                unique, index = np.unique(closest,axis=0,return_inverse=True)
                pv, valid_probe = query(provider,unique,cfg.query_batch_size)
                aux_points += len(unique)
                bound = margins[:channels]+radius+np.sqrt(squared)[:,None]
                possible[unsupported] &= ~(valid_probe[index,1:1+channels]
                                           & (pv[index,1:1+channels]>bound))
            keep[ii] |= possible.any(axis=1)
        levels.append(dict(spacing=spacing, blocks=len(blocks), kept=int(keep.sum()),
                           unknown_centers=len(unknown), new_points=len(codes)-old_count,
                           surface_probes=aux_points-old_aux,
                           seconds=perf_counter()-start))
        return blocks[keep]

    selected = scout(blocks, H)
    ratio = round(H/M)
    if ratio > 1 and len(selected):
        ensure = len(selected)*ratio**3
        require(ensure <= cfg.scout_point_budget, 'scout_block_budget',
                'Too many middle grid blocks', blocks=ensure)
        offsets = np.array(list(product(range(ratio),repeat=3)))
        selected = scout((selected[:,None,:]*ratio+offsets).reshape(-1,3),M)
    if not len(selected):
        empty.report.update(levels=levels, unique_query_points=len(codes),surface_probes=aux_points)
        return empty
    offsets = np.array(list(product(range(factor+1), repeat=3)))
    start = perf_counter()
    old_count = len(codes)
    total = len(selected)*len(offsets)
    require(total <= 16*cfg.scout_point_budget, 'scout_index_budget',
            'Too many fine block index entries', corners=total)
    # Build the index in bounded chunks; compute each coordinate only once.
    index_codes = np.empty((len(selected),len(offsets)), dtype=np.int64)
    for i in range(0,len(selected),2048):
        local = (selected[i:i+2048,None,:]*factor + offsets).reshape(-1,3)
        observe(2*local)
        index_codes[i:i+2048] = encode(2*local).reshape(-1,len(offsets))
    fine_codes, inverse = np.unique(index_codes,return_inverse=True)
    at = np.searchsorted(codes,fine_codes)
    keys = decode(fine_codes)//2
    report = dict(levels=levels, selected_blocks=len(selected), fine_points=len(keys),
                  fine_new_points=len(codes)-old_count, unique_query_points=len(codes), surface_probes=aux_points,
                  fine_valid_e=int(valid[at,1].sum()), fine_seconds=perf_counter()-start,
                  elapsed_seconds=perf_counter()-started)
    return GridSamples(selected,keys,keys*h,values[at],valid[at],
                       inverse.reshape(len(selected),*shape),report)
