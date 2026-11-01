"""Native condition grids and production-compatible MC, without CAD imports."""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def grid_value(grid, xyz, gradient=False):
    p = np.asarray(xyz, np.float32).reshape(-1, 3)
    if not np.isfinite(p).all() or (np.abs(p) > 1.000001).any():
        raise ValueError('SDF query outside normalized domain')
    r = grid.shape[0]
    coord = (p + np.float32(1)) * np.float32((r - 1) / 2)
    lower = np.clip(np.floor(coord).astype(np.int64), 0, r - 2)
    frac = coord - lower.astype(np.float32)
    result = np.zeros(len(p), np.float32)
    deriv = np.zeros((len(p), 3), np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                bits = (dx, dy, dz)
                factors = [frac[:, a] if bits[a] else 1 - frac[:, a] for a in range(3)]
                values = grid[lower[:, 0] + dx, lower[:, 1] + dy, lower[:, 2] + dz]
                weight = factors[0] * factors[1] * factors[2]
                result += values * weight
                if gradient:
                    for a in range(3):
                        others = [b for b in range(3) if b != a]
                        deriv[:, a] += values * factors[others[0]] * factors[others[1]] * (1 if bits[a] else -1)
    if gradient:
        return deriv * np.float32((r - 1) / 2)
    return result


def samples(vertices, triangles, count=20000, seed=0):
    rng = np.random.default_rng(seed)
    tri = vertices[triangles]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    ids = rng.choice(len(tri), count, p=area / area.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    return tri[ids, 0] + uv[:, :1] * (tri[ids, 1] - tri[ids, 0]) + uv[:, 1:] * (tri[ids, 2] - tri[ids, 0])


def quantiles(values):
    v = np.asarray(values).reshape(-1)
    return dict(zip(('median', 'p95', 'p99', 'max'), map(float, np.quantile(v, [.5, .95, .99, 1])))) if len(v) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--condition')
    p.add_argument('--grid')
    p.add_argument('--resolution', type=int, default=512)
    a = p.parse_args()
    import igl
    import skimage
    from skimage.measure import marching_cubes
    import trimesh
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with np.load(a.mesh) as f:
        v, t = f['vertices'], f['triangles']
    mesh = trimesh.Trimesh(v, t, process=False)
    mesh.merge_vertices(digits_vertex=7)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Source condition mesh not watertight/oriented')
    if mesh.volume < 0:
        mesh.invert()
    v, t = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32)
    if a.condition:
        with np.load(a.condition) as f:
            grid = f['sdf_grid']
        np.save(out / 'grid.npy', grid)
    elif a.grid:
        grid = np.load(a.grid, mmap_mode='c')
    else:
        r = a.resolution
        axis = np.linspace(-1, 1, r, dtype=np.float32)
        grid = np.lib.format.open_memmap(out / 'grid.npy', mode='w+', dtype=np.float32, shape=(r, r, r))
        flat = grid.reshape(-1)
        for begin in range(0, r ** 3, 65536):
            ids = np.arange(begin, min(begin + 65536, r ** 3))
            xyz = np.column_stack((axis[ids // (r * r)], axis[(ids // r) % r], axis[ids % r])).astype(float)
            sq = igl.point_mesh_squared_distance(xyz, v, t)[0]
            winding = igl.fast_winding_number(v, t, xyz)
            flat[begin:begin + len(ids)] = np.sqrt(np.maximum(sq, 0)) * np.where(winding > .5, -1, 1)
        grid.flush()
    grid_seconds = time.monotonic() - start
    r = grid.shape[0]
    mv, mt, _, _ = marching_cubes(np.asarray(grid), level=0, spacing=(2 / (r - 1),) * 3)
    mv -= 1
    np.savez_compressed(out / 'mesh.npz', vertices=mv, triangles=mt)
    sm = trimesh.Trimesh(mv, mt, process=False)
    forward = np.sqrt(igl.point_mesh_squared_distance(samples(mv, mt), v, t)[0])
    reverse = np.sqrt(igl.point_mesh_squared_distance(samples(v, t, seed=1), mv, mt)[0])
    report = dict(resolution=r, grid_seconds=grid_seconds, total_seconds=time.monotonic() - start,
                  vertices=len(mv), triangles=len(mt), watertight=bool(sm.is_watertight),
                  winding_consistent=bool(sm.is_winding_consistent),
                  mc_backend='skimage', mc_version=skimage.__version__, allow_degenerate=True,
                  condition_to_cad=quantiles(forward), cad_to_condition=quantiles(reverse),
                  distance_method='20k area samples per side, point-to-triangle; not Hausdorff',
                  grid_kind='direct_mesh_fast' if a.grid else 'published_native256' if a.condition else 'native512_from_saved_source_mesh',
                  training_asset=False)
    (out / 'grid_report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
