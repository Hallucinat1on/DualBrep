"""Local marching cubes and distance-projection skeleton samples."""
from itertools import product

import numpy as np

from .deadline import checkpoint


def extract_shell(grid, cfg):
    from skimage.measure import marching_cubes

    vertices, faces = [], []
    offset = 0
    factor = grid.block_indices.shape[1] - 1
    h, r = cfg.fine_spacing, cfg.shell_radius
    corner_slices = [tuple(slice(i, factor + i) for i in corner)
                     for corner in product((0, 1), repeat=3)]
    valid_cells = clipped_cells = 0
    for block, indices in zip(grid.blocks, grid.block_indices):
        checkpoint('edge_first_marching_cubes')
        valid = grid.valid[indices, 1]
        values = grid.values[indices, 1]
        all_valid = np.logical_and.reduce([valid[key] for key in corner_slices])
        inside = np.logical_or.reduce([(valid & (values < r))[key] for key in corner_slices])
        outside = np.logical_or.reduce([(valid & (values >= r))[key] for key in corner_slices])
        crossing = all_valid & inside & outside
        valid_cells += int(crossing.sum())
        clipped_cells += int((inside & ~all_valid).sum())
        if not crossing.any():
            continue
        # Lewiner masks a cell at its upper corner. Fill only to make the
        # backing array finite; cells touching unknown data are never emitted.
        mask = np.zeros(indices.shape, bool)
        mask[1:, 1:, 1:] = crossing
        field = np.where(valid, values - r, r).astype(np.float32)
        points, triangles, _, _ = marching_cubes(field, 0, spacing=(h, h, h),
                                                 mask=mask, allow_degenerate=False)
        vertices.append(points + block * factor * h)
        faces.append(triangles + offset)
        offset += len(points)
    if not vertices:
        return np.empty((0, 3)), np.empty((0, 3), int), dict(
            crossing_cells=0, clipped_cells=clipped_cells, triangles=0)
    points = np.vstack(vertices)
    _, unique, inverse = np.unique(np.round(points / (h * 1e-4)).astype(int), axis=0,
                                   return_index=True, return_inverse=True)
    triangles = inverse[np.vstack(faces)]
    _, keep = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    return points[unique], triangles[keep], dict(crossing_cells=valid_cells,
                                                  clipped_cells=clipped_cells,
                                                  triangles=len(keep))
