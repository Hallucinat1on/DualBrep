"""Distance-ordered thinning on the observed part of a sparse Cartesian grid."""
from functools import lru_cache
from itertools import product
import heapq
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .deadline import checkpoint


OFFSETS = np.array([p for p in product((-1, 0, 1), repeat=3) if any(p)])
RANK = np.abs(OFFSETS).sum(axis=1)
FACE_BITS = sum(1 << i for i in np.flatnonzero(RANK == 1).tolist())
N18_BITS = sum(1 << i for i in np.flatnonzero(RANK <= 2).tolist())
ALL_BITS = (1 << 26) - 1


def adjacency_mask(connectivity):
    diff = np.abs(OFFSETS[:, None] - OFFSETS)
    adjacent = diff.max(axis=2) == 1 if connectivity == 26 else diff.sum(axis=2) == 1
    return [sum(1 << j for j in np.flatnonzero(row).tolist()) for row in adjacent]


ADJ26 = adjacency_mask(26)
ADJ6 = adjacency_mask(6)
# A digital corner needs the diagonal only if a shorter local connection is absent.
SHORTCUTS = [
    [int(j) for j in np.flatnonzero(
        (RANK < rank) & (np.abs(OFFSETS - offset).sum(axis=1) < rank)
        & (np.abs(OFFSETS - offset).max(axis=1) == 1))]
    for offset, rank in zip(OFFSETS, RANK)
]


def bit_indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def flood(seed, mask, adjacency):
    reached = pending = seed
    while pending:
        bit = pending & -pending
        pending ^= bit
        added = adjacency[bit.bit_length() - 1] & mask & ~reached
        reached |= added
        pending |= added
    return reached


@lru_cache(maxsize=65536)
def simple_point(mask):
    """The local 26-foreground and 6-background connectivity numbers must both be one."""
    background = ALL_BITS ^ mask
    faces = background & FACE_BITS
    return bool(mask and faces and flood(mask & -mask, mask, ADJ26) == mask
                and (flood(faces & -faces, background & N18_BITS, ADJ6) & faces) == faces)


def lookup(sorted_codes, codes):
    index = np.searchsorted(sorted_codes, codes)
    present = index < len(sorted_codes)
    present[present] = sorted_codes[index[present]] == codes[present]
    return index, present


def lattice_neighbors(grid, level):
    valid = grid.valid[:, 1]
    inside = valid & (grid.values[:, 1] <= level)
    extent = np.ptp(grid.keys, axis=0).astype(np.int64) + 3
    strides = np.array([extent[1] * extent[2], extent[2], 1], dtype=np.int64)
    codes = grid.keys.astype(np.int64) @ strides
    order = np.argsort(codes[inside])
    foreground = codes[inside][order]
    fg_keys = grid.keys[inside][order]
    exterior_rows = np.flatnonzero(valid & ~inside)
    exterior_rows = exterior_rows[np.argsort(codes[exterior_rows])]
    exterior = codes[exterior_rows]

    neighbors = np.full((len(foreground), 26), -1, dtype=np.int32)
    frozen = np.zeros(len(foreground), dtype=bool)
    boundary = np.zeros(len(exterior_rows), dtype=bool)
    masks = np.zeros(len(foreground), dtype=np.uint32)
    for i, offset in enumerate(OFFSETS @ strides):
        checkpoint('skeleton_neighbors')
        nearby = foreground + offset
        index, present = lookup(foreground, nearby)
        neighbors[present, i] = index[present]
        masks[present] |= 1 << i
        index, known = lookup(exterior, nearby[~present])
        boundary[index[known]] = True
        frozen[~present] |= ~known
    return fg_keys, grid.values[inside, 1][order], neighbors, masks, frozen, grid.keys[exterior_rows[boundary]]


def thin_volume(grid, vertices, cfg):
    """Return skeleton positions and links using only the observed E sublevel set."""
    started = perf_counter()
    if not np.any(grid.valid[:, 1] & (grid.values[:, 1] <= cfg.shell_radius)):
        report = dict(foreground=0, skeleton_nodes=0, skeleton_links=0, unknown_interface=0,
                      protected_vertices=0, seconds=perf_counter()-started)
        return np.empty((0, 3)), np.empty((0, 2), int), report
    keys, values, neighbors, masks, frozen, boundary = lattice_neighbors(grid, cfg.shell_radius)
    fixed = frozen.copy()
    tree = cKDTree(keys)
    protected = set()
    for vertex in vertices:
        distance, idx = tree.query(vertex.position / cfg.fine_spacing)
        if distance * cfg.fine_spacing <= vertex.support_radius:
            protected.add(int(idx))
    fixed[list(protected)] = True

    # A sparse Euclidean distance transform: only observed exterior voxels
    # touching foreground can affect its boundary-voxel deletion order.
    depth = cKDTree(boundary).query(keys)[0] if len(boundary) else np.full(len(keys), np.inf)
    ready = ~fixed & ((masks & FACE_BITS) != FACE_BITS)
    queue = [(depth[i], -values[i], int(i)) for i in np.flatnonzero(ready)]
    heapq.heapify(queue)
    active = np.ones(len(keys), dtype=bool)
    visits = 0
    while queue:
        visits += 1
        if visits % 4096 == 0:
            checkpoint('skeleton_thinning')
        _, _, i = heapq.heappop(queue)
        ready[i] = False
        mask = int(masks[i])
        if not simple_point(mask):
            continue
        active[i] = False
        for j in bit_indices(mask):
            neighbor = neighbors[i, j]
            masks[neighbor] &= ALL_BITS ^ (1 << (25 - j))
            if not fixed[neighbor] and not ready[neighbor]:
                item = int(neighbor)
                heapq.heappush(queue, (depth[item], -values[item], item))
                ready[neighbor] = True

    ids = np.flatnonzero(active)
    remap = np.full(len(keys), -1, dtype=np.int32)
    remap[ids] = np.arange(len(ids))
    links = []
    for count, i in enumerate(ids):
        if count % 4096 == 0:
            checkpoint('skeleton_links')
        mask = int(masks[i])
        for j in bit_indices(mask):
            neighbor = neighbors[i, j]
            if neighbor <= i or any(mask & (1 << k) for k in SHORTCUTS[j]):
                continue
            links.append((remap[i], remap[neighbor]))
    link_array = np.asarray(links, dtype=int).reshape(-1, 2)
    return keys[ids] * cfg.fine_spacing, link_array, dict(
        foreground=len(keys), skeleton_nodes=len(ids), skeleton_links=len(links),
        unknown_interface=int(frozen.sum()), protected_vertices=len(protected), seconds=perf_counter()-started)
