"""Fast, zero-OCC 3D wireframe exporter (4-sided tubes + octahedron vertices) to OBJ+MTL or binary PLY.

Reads `wireframe.json` and `curves.json` (plus optional `edge_support.npz` / mesh) produced by
`band_brep_reconstruction` and exports lightweight colored 3D meshes viewable in any 3D viewer.
"""
import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np

COLORS = {
    'vertex': (230, 50, 50),
    'line': (45, 115, 220),
    'circle': (40, 180, 90),
    'bezier_spline': (240, 145, 30),
    'raw': (140, 140, 145),
    'mesh': (210, 210, 215),
}

_OCTA_VERTS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.float32)

_OCTA_FACES = np.array([
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
], dtype=np.int32)

_QUAD_RING = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)


def _eval_curve(rec, n_arc=32, n_spline=32):
    kind = rec['kind']
    d = rec['data']
    if kind == 'line':
        return np.asarray([d['p0'], d['p1']], dtype=np.float32)
    if kind == 'circle':
        sweep = float(d['sweep'])
        n = max(12, int(np.ceil(n_arc * abs(sweep) / (2.0 * np.pi))) + 1)
        t = np.linspace(0.0, 1.0, n)
        a = float(d['start']) + t * sweep
        c = np.asarray(d['center'], dtype=np.float64)
        x = np.asarray(d['x'], dtype=np.float64)
        y = np.asarray(d['y'], dtype=np.float64)
        r = float(d['radius'])
        pts = c[None, :] + r * (np.cos(a)[:, None] * x[None, :] + np.sin(a)[:, None] * y[None, :])
        return pts.astype(np.float32)
    # bezier_spline
    breaks = np.asarray(d['breaks'], dtype=np.float64)
    controls = np.asarray(d['controls'], dtype=np.float64)
    k = len(breaks) - 1
    n = max(16, min(96, n_spline * max(1, k // 2)))
    t = np.linspace(0.0, 1.0, n)
    span = np.clip(np.searchsorted(breaks, t, side='right') - 1, 0, k - 1)
    u = (t - breaks[span]) / (breaks[span + 1] - breaks[span])
    om = 1.0 - u
    p = controls[span]
    pts = (
        (om**3)[:, None] * p[:, 0]
        + (3 * om**2 * u)[:, None] * p[:, 1]
        + (3 * om * u**2)[:, None] * p[:, 2]
        + (u**3)[:, None] * p[:, 3]
    )
    return pts.astype(np.float32)


def _octahedra(centers, radius):
    if not len(centers):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    c = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    n = len(c)
    verts = (c[:, None, :] + _OCTA_VERTS[None, :, :] * float(radius)).reshape(-1, 3)
    offsets = (np.arange(n, dtype=np.int32) * 6)[:, None, None]
    faces = (_OCTA_FACES[None, :, :] + offsets).reshape(-1, 3)
    return verts, faces


def _polyline_tube(pts, radius):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if len(pts) < 2:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    # Drop consecutive duplicates
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-8]
    pts = pts[keep]
    m = len(pts)
    if m < 2:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)

    # Tangent per vertex (averaged over adjacent segments)
    diff = np.diff(pts, axis=0)
    diff = diff / np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-12)
    tan = np.empty_like(pts)
    tan[0] = diff[0]
    tan[-1] = diff[-1]
    if m > 2:
        mid = diff[:-1] + diff[1:]
        mid_n = np.linalg.norm(mid, axis=1, keepdims=True)
        tan[1:-1] = np.where(mid_n > 1e-6, mid / np.maximum(mid_n, 1e-12), diff[:-1])

    # Parallel-transported orthonormal frame to avoid twist along curves
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(tan[0, 0]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(tan[0], ref)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    us = np.empty_like(pts)
    vs = np.empty_like(pts)
    us[0] = u
    vs[0] = np.cross(tan[0], u)
    for i in range(1, m):
        ui = us[i - 1] - float(np.dot(us[i - 1], tan[i])) * tan[i]
        un = float(np.linalg.norm(ui))
        if un < 1e-6:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(tan[i, 0]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ui = np.cross(tan[i], ref)
            un = max(float(np.linalg.norm(ui)), 1e-12)
        us[i] = ui / un
        vs[i] = np.cross(tan[i], us[i])

    # Ring vertices: [m, 4, 3]
    r = float(radius)
    rings = pts[:, None, :] + r * (_QUAD_RING[None, :, 0:1] * us[:, None, :] + _QUAD_RING[None, :, 1:2] * vs[:, None, :])
    verts = rings.reshape(-1, 3)

    # Connect ring i to ring i+1 (8 triangles per segment)
    idx = np.arange(m - 1, dtype=np.int32) * 4
    c0 = np.arange(4, dtype=np.int32)
    c1 = (c0 + 1) % 4
    a0 = idx[:, None] + c0[None, :]
    a1 = idx[:, None] + c1[None, :]
    b0 = a0 + 4
    b1 = a1 + 4
    f1 = np.stack((a0, b0, b1), axis=-1).reshape(-1, 3)
    f2 = np.stack((a0, b1, a1), axis=-1).reshape(-1, 3)
    faces = np.vstack((f1, f2))
    return verts, faces


def _collect_groups(run_dir, *, edge_radius, vertex_radius, include_raw=False, include_mesh=False):
    wf = json.loads((run_dir / 'wireframe.json').read_text())
    curves = json.loads((run_dir / 'curves.json').read_text())

    groups = {}

    def add(name, v, f):
        if not len(v) or not len(f):
            return
        if name not in groups:
            groups[name] = ([v], [f], len(v))
        else:
            vs, fs, count = groups[name]
            vs.append(v)
            fs.append(f + count)
            groups[name] = (vs, fs, count + len(v))

    # 1. Vertices
    v_pos = [v['position'] for v in wf.get('vertices', [])]
    if v_pos:
        add('vertex', *_octahedra(v_pos, vertex_radius))

    # 2. Parametric curves grouped by kind
    for rec in curves.values():
        pts = _eval_curve(rec)
        kind = rec.get('kind', 'bezier_spline')
        if kind not in COLORS:
            kind = 'bezier_spline'
        add(kind, *_polyline_tube(pts, edge_radius))

    # 3. Optional raw extracted samples
    if include_raw:
        for edge in wf.get('edges', []):
            pts = np.asarray(edge.get('samples', []), dtype=np.float32)
            if len(pts) >= 2:
                add('raw', *_polyline_tube(pts, edge_radius * 0.6))

    # 4. Optional reference mesh
    if include_mesh:
        for fname in ('split_mesh.npz', 'face_regions.npz'):
            p = run_dir / fname
            if p.is_file():
                with np.load(p) as arc:
                    add('mesh', np.asarray(arc['vertices'], np.float32), np.asarray(arc['triangles'], np.int32))
                break

    merged = {}
    for name, (vs, fs, _) in groups.items():
        merged[name] = (np.vstack(vs), np.vstack(fs))
    return merged


def _write_obj(out_path: Path, groups: dict):
    mtl_path = out_path.with_suffix('.mtl')
    mtl_lines = []
    for name in groups:
        r, g, b = [c / 255.0 for c in COLORS[name]]
        alpha = 0.35 if name == 'mesh' else 1.0
        mtl_lines.append(f'newmtl {name}\nKd {r:.4f} {g:.4f} {b:.4f}\nd {alpha:.2f}\n')
    mtl_path.write_text('\n'.join(mtl_lines))

    lines = [f'mtllib {mtl_path.name}\n']
    v_offset = 1
    for name, (verts, faces) in groups.items():
        lines.append(f'g {name}\nusemtl {name}\n')
        lines.extend(f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in verts)
        f_shifted = faces + v_offset
        lines.extend(f'f {a} {b} {c}\n' for a, b, c in f_shifted)
        v_offset += len(verts)
    out_path.write_text(''.join(lines))


def _write_ply(out_path: Path, groups: dict):
    all_v, all_c, all_f = [], [], []
    offset = 0
    for name, (verts, faces) in groups.items():
        rgb = np.broadcast_to(np.asarray(COLORS[name], dtype=np.uint8), (len(verts), 3))
        all_v.append(verts.astype('<f4'))
        all_c.append(rgb)
        all_f.append((faces + offset).astype('<i4'))
        offset += len(verts)
    verts = np.vstack(all_v) if all_v else np.empty((0, 3), '<f4')
    colors = np.vstack(all_c) if all_c else np.empty((0, 3), np.uint8)
    faces = np.vstack(all_f) if all_f else np.empty((0, 3), '<i4')

    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        f'element vertex {len(verts)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        f'element face {len(faces)}\n'
        'property list uchar int vertex_indices\n'
        'end_header\n'
    ).encode('ascii')

    v_dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    v_buf = np.empty(len(verts), dtype=v_dtype)
    if len(verts):
        v_buf['x'], v_buf['y'], v_buf['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
        v_buf['r'], v_buf['g'], v_buf['b'] = colors[:, 0], colors[:, 1], colors[:, 2]

    f_dtype = np.dtype([('n', 'u1'), ('i0', '<i4'), ('i1', '<i4'), ('i2', '<i4')])
    f_buf = np.empty(len(faces), dtype=f_dtype)
    if len(faces):
        f_buf['n'] = 3
        f_buf['i0'], f_buf['i1'], f_buf['i2'] = faces[:, 0], faces[:, 1], faces[:, 2]

    with out_path.open('wb') as fh:
        fh.write(header)
        fh.write(v_buf.tobytes())
        fh.write(f_buf.tobytes())


def _write_support_ply(run_dir: Path, out_path: Path):
    src = run_dir / 'edge_support.npz'
    if not src.is_file():
        return False
    with np.load(src) as arc:
        pts = np.asarray(arc['points'], dtype='<f4').reshape(-1, 3)
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        f'element vertex {len(pts)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'end_header\n'
    ).encode('ascii')
    with out_path.open('wb') as fh:
        fh.write(header)
        fh.write(pts.tobytes())
    return True


def export_run(run_dir: Path, out_path: Path, *, fmt=None, edge_radius=0.0015, vertex_radius=0.004,
               include_raw=False, include_mesh=False, include_support=False):
    groups = _collect_groups(
        run_dir,
        edge_radius=edge_radius,
        vertex_radius=vertex_radius,
        include_raw=include_raw,
        include_mesh=include_mesh,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = (fmt or out_path.suffix.lstrip('.') or 'obj').lower()
    if ext == 'ply':
        _write_ply(out_path, groups)
    else:
        _write_obj(out_path, groups)
    if include_support:
        _write_support_ply(run_dir, out_path.with_name(out_path.stem + '_support.ply'))
    n_v = sum(len(v) for v, _ in groups.values())
    n_f = sum(len(f) for _, f in groups.values())
    return n_v, n_f


def _discover_runs(root: Path):
    if (root / 'wireframe.json').is_file():
        return [(root.name, root)]
    if (root / 'reconstruction/wireframe.json').is_file():
        return [(root.name, root / 'reconstruction')]
    runs = []
    for wf in sorted(root.rglob('wireframe.json')):
        run_dir = wf.parent
        rel = run_dir.parent.relative_to(root) if run_dir.name == 'reconstruction' else run_dir.relative_to(root)
        label = '_'.join(rel.parts) if rel.parts else run_dir.name
        runs.append((label, run_dir))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Run directory or batch experiment directory')
    parser.add_argument('--out', type=Path, required=True, help='Output file (single) or directory (batch)')
    parser.add_argument('--format', choices=('obj', 'ply'), default=None, help='Output format (default: from suffix or obj)')
    parser.add_argument('--edge-radius', type=float, default=0.0015)
    parser.add_argument('--vertex-radius', type=float, default=0.004)
    parser.add_argument('--raw', action='store_true', help='Overlay pre-fit raw edge samples')
    parser.add_argument('--with-mesh', action='store_true', help='Overlay reference surface mesh')
    parser.add_argument('--support', action='store_true', help='Also export edge_support.npz as *_support.ply')
    args = parser.parse_args()

    t0 = time.monotonic()
    runs = _discover_runs(args.input)
    if not runs:
        parser.exit(1, f'No wireframe.json found under {args.input}\n')

    single = len(runs) == 1 and args.out.suffix.lower() in ('.obj', '.ply')
    ext = args.format or (args.out.suffix.lstrip('.').lower() if single else 'obj')
    for label, run_dir in runs:
        target = args.out if single else args.out / f'{label}.{ext}'
        nv, nf = export_run(
            run_dir, target, fmt=ext,
            edge_radius=args.edge_radius, vertex_radius=args.vertex_radius,
            include_raw=args.raw, include_mesh=args.with_mesh, include_support=args.support,
        )
        kb = target.stat().st_size / 1024.0
        print(f'{label:24s} -> {target.name} ({nv:6d} verts, {nf:6d} tris, {kb:6.1f} KB)')
    print(f'Exported {len(runs)} wireframe(s) in {(time.monotonic() - t0) * 1000:.1f} ms')


if __name__ == '__main__':
    main()
