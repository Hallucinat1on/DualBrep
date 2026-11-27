"""Run the 20-dev paired reconstruction experiment (hybrid vs global_exact) and offline STEP scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

DEV_20 = (
    '00000168', '00000602', '00000688', '00002331', '00002720',
    '00002748', '00002978', '00003114', '00003261', '00003291',
    '00004610', '00004627', '00005094', '00007970', '00008133',
    '00008222', '00008280', '00008804', '00009340', '00009843',
)


def _run_one(task):
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'benchmarks'))
    sys.path.insert(0, str(root.parent / 'veudf/src'))

    from bandbrep.cad_source import load_cad_source
    from bandbrep.config import Config
    from bandbrep.deadline import sample_budget
    from bandbrep.edge_first import reconstruct, reconstruct_point_cloud
    from bandbrep.point_cloud_source import build_proxy_mesh
    from bandbrep.surface_mesh import normals as mesh_normals
    from evaluate_step import evaluate
    import numpy as np

    group, mid, step_path, sha, out_dir, hybrid, timeout_s = task
    out_dir = Path(out_dir)
    run_dir = out_dir / 'reconstruction'
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        fast_faces=False,
        hybrid_projection=hybrid,
        allow_partial_wireframe=True,
        scout_point_budget=16_000_000,
        sample_timeout_seconds=timeout_s,
    )
    t0 = time.monotonic()
    status = 'failed'
    error = None
    try:
        with sample_budget(timeout_s):
            mesh, udf = load_cad_source(Path(step_path), sha)
            if group == 'point_cloud':
                rng = np.random.default_rng(0)
                tri = mesh.vertices[mesh.triangles]
                tri_n = mesh_normals(mesh)
                areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                pick = rng.choice(len(tri), size=50000, p=areas / areas.sum())
                r = np.sqrt(rng.random(len(pick)))
                s = rng.random(len(pick))
                bary = np.column_stack((1.0 - r, r * (1.0 - s), r * s))
                raw_pts = np.einsum('ni,nij->nj', bary, tri[pick])
                raw_nrms = tri_n[pick]
                proxy = build_proxy_mesh(raw_pts, raw_nrms, depth=8)
                proxy.report['source_step_sha256'] = sha
                proxy.report['normalization'] = mesh.report.get('normalization', {})
                res = reconstruct_point_cloud(raw_pts, udf, cfg, run_dir, normals=raw_nrms, proxy_mesh=proxy)
            else:
                res = reconstruct(mesh, udf, cfg, run_dir)
            if res['report'].get('status') == 'solid_validated':
                status = 'success'
                sc = evaluate(Path(step_path), run_dir)
                (out_dir / 'score.json').write_text(json.dumps(sc, indent=2) + '\n')
    except Exception as exc:
        error = dict(code=getattr(exc, 'code', type(exc).__name__), message=str(exc))
        if getattr(exc, 'code', None) == 'sample_timeout':
            status = 'timeout'
    elapsed = time.monotonic() - t0
    return dict(group=group, model=mid, status=status, elapsed_seconds=round(elapsed, 2), error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step-root', type=Path, default=Path('/houjunhao/Datasets/abc/step/0000'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=3600.0)
    parser.add_argument('--with-point-cloud', action='store_true')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from bandbrep.cache import file_hash

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for mid in DEV_20:
        matches = sorted((args.step_root / mid).glob('*.step'))
        if not matches:
            raise FileNotFoundError(f'No STEP file found for {mid} under {args.step_root}')
        step_path = matches[0]
        manifest.append(dict(id=mid, step=str(step_path), source_sha256=file_hash(step_path)))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    groups = [('hybrid', True), ('global_exact', False)]
    if args.with_point_cloud:
        groups.append(('point_cloud', False))
    tasks = []
    for grp, hybrid in groups:
        for row in manifest:
            tasks.append((
                grp,
                row['id'],
                row['step'],
                row['source_sha256'],
                str(args.output / grp / row['id']),
                hybrid,
                args.timeout,
            ))

    results = []
    t_start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done = len(results)
            err_tag = f" ({rec['error']['code']})" if rec.get('error') else ''
            print(f"[{done:2d}/{len(tasks)}] {rec['group']:12s} {rec['model']} -> {rec['status']:7s}{err_tag} {rec['elapsed_seconds']:6.1f}s", flush=True)
            (args.output / 'results.json').write_text(json.dumps(sorted(results, key=lambda r: (r['group'], r['model'])), indent=2) + '\n')

    print(f'All {len(results)} tasks finished in {time.monotonic() - t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
