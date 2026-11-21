"""Export Marching Cubes E/V thin shells (and optional skeleton) to Wavefront OBJ."""
import argparse
import json
from pathlib import Path

import numpy as np

from bandbrep.config import Config
from bandbrep.edge_first_grid import GridSamples
from bandbrep.edge_first_mc import extract_shell


def write_mesh_obj(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Write a triangle mesh to an ASCII Wavefront OBJ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write(f'# Exported by export_mc_shell_obj.py ({len(vertices)} vertices, {len(triangles)} faces)\n')
        for x, y, z in np.asarray(vertices, dtype=float).reshape(-1, 3):
            f.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        for a, b, c in np.asarray(triangles, dtype=np.int64).reshape(-1, 3) + 1:
            f.write(f'f {a} {b} {c}\n')


def write_skeleton_obj(path: Path, points: np.ndarray, links: np.ndarray) -> None:
    """Write a 3D skeleton graph as OBJ vertices and line segments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write(f'# Exported skeleton ({len(points)} nodes, {len(links)} links)\n')
        for x, y, z in np.asarray(points, dtype=float).reshape(-1, 3):
            f.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        for a, b in np.asarray(links, dtype=np.int64).reshape(-1, 2) + 1:
            f.write(f'l {a} {b}\n')


def load_scout_grid(scout_path: Path) -> GridSamples:
    with np.load(scout_path) as data:
        return GridSamples(
            blocks=data['blocks'],
            keys=data['keys'],
            points=data['points'],
            values=data['values'],
            valid=data['valid'],
            block_indices=data['block_indices'],
            report={},
        )


def export_from_run_dir(run_dir: Path, out_dir: Path, *,
                        shell_radius: float | None = None,
                        vertex_level: float | None = None,
                        include_skeleton: bool = True) -> dict:
    """Extract and export E/V MC thin shells from a reconstruction output directory."""
    if (run_dir / 'reconstruction').is_dir() and not (run_dir / 'scout.npz').is_file():
        run_dir = run_dir / 'reconstruction'
    cfg_dict = {}
    run_json = run_dir / 'run.json'
    if run_json.is_file():
        cfg_dict = json.loads(run_json.read_text()).get('config', {})
    h = float(cfg_dict.get('fine_spacing', 0.003))
    r_e = float(cfg_dict.get('shell_radius', 0.005)) if shell_radius is None else float(shell_radius)
    r_v = float(cfg_dict.get('vertex_level', 0.005)) if vertex_level is None else float(vertex_level)
    cfg = Config(fine_spacing=h, shell_radius=r_e, vertex_level=r_v)

    exported = {}
    scout_path = run_dir / 'scout.npz'
    if scout_path.is_file():
        grid = load_scout_grid(scout_path)
        e_verts, e_faces, e_rep = extract_shell(grid, cfg, channel=1, spacing=h, level=r_e)
        e_obj = out_dir / f'edge_shell_r{r_e:.4f}.obj'
        write_mesh_obj(e_obj, e_verts, e_faces)
        exported['edge_shell'] = dict(path=str(e_obj), vertices=len(e_verts), faces=len(e_faces), **e_rep)

        if vertex_level is None and (run_dir / 'vertex_shell.npz').is_file():
            with np.load(run_dir / 'vertex_shell.npz') as vs:
                v_verts, v_faces = vs['vertices'], vs['triangles']
            v_rep = {}
        else:
            v_verts, v_faces, v_rep = extract_shell(grid, cfg, channel=2, spacing=h, level=r_v)
        v_obj = out_dir / f'vertex_shell_r{r_v:.4f}.obj'
        write_mesh_obj(v_obj, v_verts, v_faces)
        exported['vertex_shell'] = dict(path=str(v_obj), vertices=len(v_verts), faces=len(v_faces), **v_rep)
    elif (run_dir / 'vertex_shell.npz').is_file():
        with np.load(run_dir / 'vertex_shell.npz') as vs:
            v_verts, v_faces = vs['vertices'], vs['triangles']
        v_obj = out_dir / 'vertex_shell.obj'
        write_mesh_obj(v_obj, v_verts, v_faces)
        exported['vertex_shell'] = dict(path=str(v_obj), vertices=len(v_verts), faces=len(v_faces))
    else:
        raise FileNotFoundError(f'Neither scout.npz nor vertex_shell.npz found in {run_dir}')

    if include_skeleton and (run_dir / 'edge_skeleton.npz').is_file():
        with np.load(run_dir / 'edge_skeleton.npz') as sk:
            s_pts, s_links = sk['points'], sk['links']
        s_obj = out_dir / 'edge_skeleton.obj'
        write_skeleton_obj(s_obj, s_pts, s_links)
        exported['edge_skeleton'] = dict(path=str(s_obj), nodes=len(s_pts), links=len(s_links))

    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path,
                        help='Reconstruction directory (containing scout.npz) or a shell .npz file')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output directory (for run dir input) or .obj file path (for .npz input)')
    parser.add_argument('--shell-radius', type=float, default=None,
                        help='Override E isosurface radius (default: value from run.json or 0.005)')
    parser.add_argument('--vertex-level', type=float, default=None,
                        help='Override V isosurface radius (default: value from run.json or 0.005)')
    parser.add_argument('--no-skeleton', action='store_true',
                        help='Skip exporting edge_skeleton.obj')
    args = parser.parse_args()

    if args.input.is_file() and args.input.suffix == '.npz':
        out_path = args.output or args.input.with_suffix('.obj')
        with np.load(args.input) as data:
            write_mesh_obj(out_path, data['vertices'], data['triangles'])
        print(json.dumps({'mesh': str(out_path), 'vertices': len(data['vertices']),
                          'faces': len(data['triangles'])}, indent=2))
        return

    out_dir = args.output or (args.input / 'mc_vis')
    result = export_from_run_dir(
        args.input, out_dir,
        shell_radius=args.shell_radius,
        vertex_level=args.vertex_level,
        include_skeleton=not args.no_skeleton,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
