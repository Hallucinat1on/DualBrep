"""Frozen val-only paired CAD reconstruction experiments; no production writes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
GROUPS = {'A': ('A', .003), 'B': ('B', .003), 'C004': ('B', .004),
          'C005': ('B', .005), 'C006': ('B', .006), 'D0': ('D', .003), 'D006': ('D', .006)}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def code_identity():
    paths = sorted(list((ROOT / 'bandbrep').rglob('*.py')) + list(HERE.glob('*.py')))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def check_identity(out):
    identity = load(out / 'identity.json')
    if identity['code'] != code_identity():
        raise RuntimeError('Frozen experiment code changed; use a new output version')
    for path, expected in identity['inputs'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    if digest(out / 'manifest.json') != identity['manifest_sha256']:
        raise RuntimeError('Frozen manifest changed')
    if digest(out / 'selection_policy.json') != identity['policy_sha256']:
        raise RuntimeError('Frozen selection policy changed')


def select(a):
    out, data = Path(a.output), Path(a.data)
    if (out / 'identity.json').exists():
        check_identity(out)
        print('Frozen cohort already exists', flush=True)
        return
    index = load(data / 'prepared/index.json')
    historical = [json.loads(s) for s in (HERE / 'datasets30_20261013/manifest.jsonl').read_text().splitlines() if s]
    old_ids = set()
    old_families = set()
    for r in historical:
        text = json.dumps(r)
        import re
        old_ids.update(re.findall(r'(?<!\d)\d{8}(?=[/_])', text))
        old_families.update(re.findall(r'(?<=_)[0-9a-f]{24}(?=_)', text))
    eligible = []
    for item in index['assets']:
        if item['split'] != 'val' or item['family_id'] in old_families or item['asset_id'][:8] in old_ids:
            continue
        aid = item['asset_id']
        screen = load(data / 'screen_results' / (aid + '.json'))
        if screen['status'] != 'passed':
            raise RuntimeError('Published asset without passed filter')
        report = screen['report']
        n = report['face_count']
        layer = '10-25' if n <= 25 else '26-80' if n <= 80 else '81-100'
        meta = load(data / 'raw' / aid / 'metadata.json')
        eligible.append(dict(item, id=aid[:8], face_count=n, layer=layer,
                             step=meta['source_path'], raw=str(data / 'raw' / aid),
                             prepared=str(data / 'prepared' / aid),
                             min_edge=report['min_target_edge_length']))
    rng = random.Random(a.seed)
    rng.shuffle(eligible)
    chosen, used = [], set()
    # Scarce high-complexity families first; all quotas fixed before reconstruction.
    quotas = {'81-100': 5, '26-80': 12, '10-25': 13}
    for layer, count in quotas.items():
        rows = [r for r in eligible if r['layer'] == layer]
        taken = 0
        for row in rows:
            if row['family_id'] in used:
                continue
            chosen.append(row)
            used.add(row['family_id'])
            taken += 1
            if taken == count:
                break
        if taken != count:
            raise RuntimeError('Insufficient independent families for frozen quota: ' + layer)
    check_counts = {'81-100': 2, '26-80': 4, '10-25': 4}
    seen = Counter()
    for row in chosen:
        layer = row['layer']
        row['cohort'] = 'check' if seen[layer] < check_counts[layer] else 'dev'
        seen[layer] += 1
    chosen.sort(key=lambda r: r['id'])
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'manifest.json', chosen)
    save(out / 'selection_policy.json', dict(seed=a.seed, quotas=quotas, dev=20, check=10,
         selection='seeded shuffle of eligible val assets, one model per family, scarce stratum first',
         historical_S30='exclude known IDs and document families', H100_H100b_exposure='unknown: manifests unavailable',
         new_claim='newly drawn filtered val cohort, not certified unseen historical holdout',
         candidate_feature_tolerance=[.003, .004, .005, .006],
         selection_rule='smallest tolerance with >=1 net graph-and-STEP rescue on dev, no STEP/graph regressions and no new geometry alarm; otherwise .003',
         geometry_alarm=dict(volume_rel_error=.01, sampled_distance=.01),
         common_success_geometry_max_increase=.0002, resolution_upgrade_min_dev_rescues=2,
         resolution_upgrade_requires_check_no_regression=True,
         timeout_seconds=a.timeout, preparation_timeout_seconds=a.prepare_timeout, jobs=a.jobs,
         max_triangles=1500000, band_width=.02, outside_policy='fail_closed',
         scope='30-model CAD oracle A/B/C/D; no training or final test',
         environment_strategy='brdf CAD; hunyuan 3.10 skimage 0.24 MC subprocess; no shared environment modifications'))
    inputs = {str(p): digest(p) for p in [data / 'prepared/index.json', data / 'dataset_filter.json', data / 'reports/deduplication.json']}
    for row in chosen:
        p = Path(row['raw']) / 'metadata.json'
        inputs[str(p)] = digest(p)
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'trimesh', 'torch')}
    save(out / 'environment.json', dict(cad_python=sys.version, cad_packages=versions,
         grid_environment=json.loads(subprocess.check_output([a.grid_python, '-c',
             'import json,sys,numpy,skimage,trimesh; print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,skimage=skimage.__version__,trimesh=trimesh.__version__)))'], text=True))))
    inputs[str(out / 'environment.json')] = digest(out / 'environment.json')
    save(out / 'identity.json', dict(code=code_identity(), inputs=inputs,
         policy_sha256=digest(out / 'selection_policy.json'),
         manifest_sha256=digest(out / 'manifest.json'), created=time.time(),
         python=sys.version, executable=sys.executable, groups=GROUPS))
    print(json.dumps({'selected': len(chosen), 'layers': dict(Counter(r['layer'] for r in chosen)),
                      'cohorts': dict(Counter(r['cohort'] for r in chosen)), 'ids': [r['id'] for r in chosen]}, indent=2), flush=True)


def source_for(row):
    import numpy as np
    from experiment_fields import BenchmarkCAD
    from bandbrep.vendor.oracle.config import Config as OracleConfig
    from bandbrep.vendor.oracle.distance import SDFOracle
    from bandbrep.vendor.oracle.cad import read_normalized
    raw, prepared = Path(row['raw']), Path(row['prepared'])
    meta = load(raw / 'metadata.json')
    if digest(row['step']) != row['source_sha256']:
        raise RuntimeError('STEP source hash mismatch')
    for name, expected in row['hashes'].items():
        if digest(prepared / name) != expected:
            raise RuntimeError('Prepared asset hash mismatch: ' + name)
    for name in ('surface_mesh.npz', 'topology.json'):
        if digest(raw / name) != meta['artifact_sha256'][name]:
            raise RuntimeError('Raw artifact hash mismatch: ' + name)
    source = BenchmarkCAD(row['step'])
    if source.targets.topology != load(raw / 'topology.json'):
        raise RuntimeError('Target topology differs from frozen raw')
    if not np.allclose(source.normalization['world_to_normalized'], meta['normalization']['world_to_normalized'], atol=1e-12, rtol=0):
        raise RuntimeError('Normalization differs from frozen raw')
    source.config = OracleConfig(**meta['config'])
    solid, _ = read_normalized(row['step'], source.config)
    with np.load(raw / 'surface_mesh.npz') as f:
        source.mesh = (f['vertices'], f['triangles'], f['face_ids'])
    source.surface = SDFOracle(solid, source.targets.faces, source.mesh, source.config)
    return source


def cfg_for(mode, tolerance):
    from bandbrep.config import load_config
    return load_config(preset='recommended', overrides=dict(resolution=512 if mode == 'D' else 256,
        max_resolution=512 if mode == 'D' else 256, query_batch_size=65536,
        feature_sdf_tolerance=tolerance, save_queries=False))


def worker(a):
    import numpy as np
    from alignment_fields import AlignmentField
    from alignment_grid import grid_value, quantiles
    from bandbrep.schemas import SurfaceMesh, write_json, require
    from bandbrep.surface_mesh import orient_and_validate
    from experiment_fields import ScalarOnly
    from run_reconstruction_experiments import graph_case, full, save_graph
    out = Path(a.output)
    row = next(r for r in load(out / 'manifest.json') if r['id'] == a.model)
    mode = a.group if a.phase == 'prepare' else GROUPS[a.group][0]
    cache = out / 'cache' / row['id'] / mode
    folder = out / 'runs' / row['id'] / a.group
    target = cache if a.phase == 'prepare' else folder
    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(id=row['id'], group=a.group, status='running', phase=a.phase)
    def progress(stage):
        save(target / 'progress.json', dict(stage=stage, elapsed_seconds=time.monotonic() - started))
    try:
        progress('source_audit')
        source = source_for(row)
        if a.phase == 'prepare':
            progress('native_grid_and_mc')
            cmd = [a.grid_python, str(HERE / 'alignment_grid.py'), '--mesh', str(Path(row['raw']) / 'surface_mesh.npz'), '--output', str(cache)]
            if mode == 'A':
                grid = np.lib.format.open_memmap(cache / 'grid.npy', mode='w+', dtype=np.float32, shape=(256, 256, 256))
                axis = np.linspace(-1, 1, 256, dtype=np.float32)
                for begin in range(0, 256 ** 3, 65536):
                    ids = np.arange(begin, min(begin + 65536, 256 ** 3))
                    p = np.column_stack((axis[ids // 65536], axis[(ids // 256) % 256], axis[ids % 256])).astype(float)
                    grid.reshape(-1)[begin:begin + len(p)] = source.sdf(p)
                grid.flush()
                cmd += ['--grid', str(cache / 'grid.npy')]
            elif mode == 'B':
                cmd += ['--condition', str(Path(row['prepared']) / 'condition.npz')]
            else:
                cmd += ['--resolution', '512']
            subprocess.run(cmd, check=True)
        else:
            prep_process = load(cache / 'process.json')
            if prep_process.get('status') != 'completed' or load(cache / 'prepared.json')['status'] != 'prepared':
                raise RuntimeError('Paired geometry preparation did not succeed')
        grid = np.load(cache / 'grid.npy', mmap_mode='r')
        cfg = cfg_for(mode, .003 if a.phase == 'prepare' else GROUPS[a.group][1])
        provider = AlignmentField(source, grid, mode, row['source_sha256'] + ':' + mode, progress=progress)
        with np.load(cache / 'mesh.npz') as f:
            mesh = SurfaceMesh(f['vertices'], f['triangles'].astype(np.int64))
        require(len(mesh.triangles) <= cfg.max_triangles, 'mesh_resource_limit', 'MC triangle budget exceeded', triangles=len(mesh.triangles))
        progress('surface_validation')
        orient_and_validate(mesh, ScalarOnly(provider), cfg)
        points = mesh.vertices[mesh.triangles].mean(axis=1)
        if a.phase == 'prepare':
            progress('exact_center_fields')
            values = np.empty((len(points), 3))
            for begin in range(0, len(points), 8192):
                values[begin:begin + 8192] = provider.evaluate(points[begin:begin + 8192]).values
            np.savez_compressed(cache / 'centers.npz', points=points, values=values)
            rng = np.random.default_rng(20261018)
            ids = rng.integers(0, grid.size, size=4096)
            r = grid.shape[0]
            axis = np.linspace(-1, 1, r, dtype=np.float32)
            q = np.column_stack((axis[ids // (r * r)], axis[(ids // r) % r], axis[ids % r])).astype(float)
            direct = source.sdf(q)
            saved = grid.reshape(-1)[ids]
            edges = np.vstack([p[np.linspace(0, len(p) - 1, min(len(p), max(128, len(p) // 4))).astype(int)] for p in source.edge.pieces]) if source.edge else np.empty((0, 3))
            stats = dict(node_error=quantiles(abs(direct - saved)),
                         significant_sign_mismatch=int(((direct * saved < 0) & (abs(direct) > 1e-6)).sum()),
                         edge_abs_grid_sdf=quantiles(abs(grid_value(grid, edges))),
                         vertex_abs_grid_sdf=quantiles(abs(grid_value(grid, source.targets.vertices))),
                         edge_probe_count=len(edges), vertex_probe_count=len(source.targets.vertices))
            reference = provider.edge.query(q[:64])['distance']
            from bandbrep.vendor.oracle.distance import EdgeOracle
            cad_ref = EdgeOracle(source.targets.edges, source.config, backend='cad_reference').query(q[:64])['distance']
            stats['edge_reference_max_error'] = float(np.max(abs(reference - cad_ref), initial=0))
            require(stats['edge_reference_max_error'] <= 1e-5, 'oracle_reference_failed', 'Exact E/V backend reference mismatch')
            result.update(status='prepared', surface=mesh.report, diagnostics=stats,
                          artifact_hashes={name: digest(cache / name) for name in ('grid.npy', 'mesh.npz', 'centers.npz')})
        else:
            expected = load(cache / 'prepared.json')['artifact_hashes']
            for name, value in expected.items():
                if digest(cache / name) != value:
                    raise RuntimeError('Prepared cache hash mismatch: ' + name)
            with np.load(cache / 'centers.npz') as f:
                require(np.array_equal(points, f['points']), 'center_cache_mismatch', 'Mesh centers changed')
                values = f['values']
            provider.center_points, provider.center_values = points, values
            progress('topology')
            bands, topo, scoring = graph_case(source, mesh, points, values, provider, cfg)
            save_graph(folder, bands, topo)
            write_json(folder / 'graph_score.json', scoring)
            result['score'] = scoring
            save(folder / 'outcome.json', result)
            reconstruction = full(source, mesh, bands, topo, cfg, folder, field=provider)
            result.update(status=reconstruction['status'], full=reconstruction)
        result['queries'] = provider.report()
    except Exception as exc:
        result.update(status='failed', error=dict(code=getattr(exc, 'code', type(exc).__name__), message=str(exc),
                      details=getattr(exc, 'details', {})), traceback=traceback.format_exc())
        if 'provider' in locals():
            result['queries'] = provider.report()
    result.update(seconds=time.monotonic() - started, maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(target / ('prepared.json' if a.phase == 'prepare' else 'outcome.json'), result)
    print(row['id'], a.group, result['status'], result.get('error', {}).get('code', ''), flush=True)


def launch(a, row, group):
    out = Path(a.output)
    phase = a.phase
    target = out / ('cache' if phase == 'prepare' else 'runs') / row['id'] / group
    result_path = target / ('prepared.json' if phase == 'prepare' else 'outcome.json')
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'process.json').exists():
        return load(target / 'process.json')
    command = [sys.executable, str(Path(__file__).resolve()), '--output', str(out), '--phase', phase,
               '--worker', '--model', row['id'], '--group', group, '--grid-python', a.grid_python]
    started = time.monotonic()
    with open(target / 'worker.log', 'w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=a.prepare_timeout if phase == 'prepare' else a.timeout)
            status = 'completed' if code == 0 and result_path.exists() else 'process_failed'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code, status = proc.returncode, 'timeout'
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            raise
    result = dict(id=row['id'], group=group, status=status, returncode=code, seconds=time.monotonic() - started)
    save(target / 'process.json', result)
    print(json.dumps(result), flush=True)
    return result


def collect(out, row, group):
    folder = out / 'runs' / row['id'] / group
    result = load(folder / 'outcome.json') if (folder / 'outcome.json').exists() else {}
    process = load(folder / 'process.json') if (folder / 'process.json').exists() else {'status': 'not_run'}
    effective = result.get('status', 'missing') if process['status'] == 'completed' else process['status']
    score = result.get('score', {})
    if not score and (folder / 'graph_score.json').exists():
        score = load(folder / 'graph_score.json')
    full = result.get('full', {})
    progress = load(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
    preparation = folder.parents[2] / 'cache' / row['id'] / GROUPS[group][0] / 'prepared.json'
    prep = load(preparation) if preparation.exists() else {}
    return dict(id=row['id'], layer=row['layer'], cohort=row['cohort'], group=group,
                status=effective, step=effective == 'solid_validated',
                graph=bool(score.get('graph_good')), face=bool(score.get('face_good')),
                score=score, fidelity=full.get('fidelity', {}),
                failure=full.get('error', result.get('error', {})),
                failed_stage=full.get('failed_stage', progress.get('stage')), process=process,
                preparation_failure=prep.get('error'), progress=progress,
                queries=result.get('queries', {}), maxrss_kb=result.get('maxrss_kb'))


def summarize(a):
    out = Path(a.output)
    rows = load(out / 'manifest.json')
    groups = a.groups.split(',') if a.groups else list(GROUPS)
    cases = [collect(out, row, group) for group in groups for row in rows]
    summary = {}
    for group in groups:
        summary[group] = {}
        for subset in ('all', 'dev', 'check'):
            c = [r for r in cases if r['group'] == group and (subset == 'all' or r['cohort'] == subset)]
            summary[group][subset] = dict(n=len(c), step=sum(r['step'] for r in c), graph=sum(r['graph'] for r in c),
                face=sum(r['face'] for r in c), joint=sum(r['step'] and r['graph'] for r in c),
                status=dict(Counter(r['status'] for r in c)),
                failures=dict(Counter(r['failure'].get('code', r['status']) for r in c if not r['step'])))
    pairs = {}
    for left, right in [('A', 'B'), ('B', 'C004'), ('B', 'C005'), ('B', 'C006'), ('B', 'D0'), ('C006', 'D006')]:
        if left not in groups or right not in groups:
            continue
        l = {r['id']: r for r in cases if r['group'] == left}
        r = {r['id']: r for r in cases if r['group'] == right}
        pairs[left + '->' + right] = {metric: {'rescues': [i for i in l if not l[i][metric] and r[i][metric]],
                                                   'regressions': [i for i in l if l[i][metric] and not r[i][metric]]}
                                        for metric in ('step', 'graph', 'face')}
    save(out / 'outcomes.json', cases)
    save(out / 'summary.json', dict(groups=summary, pairs=pairs, groups_definition=GROUPS,
                                   scope='filtered val engineering diagnostic; no learned field or final test'))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--data', default='/opt/data/private/yihengxu/Datasets/abc_ve_udf_dualbrep_r256_20261017')
    p.add_argument('--phase', choices=['select', 'prepare', 'run', 'summarize'], required=True)
    p.add_argument('--groups')
    p.add_argument('--ids')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--prepare-timeout', type=int, default=1800)
    p.add_argument('--seed', type=int, default=20261018)
    p.add_argument('--grid-python', default='/miniconda/envs/hunyuan/bin/python')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--model')
    p.add_argument('--group')
    a = p.parse_args()
    if a.phase != 'select':
        check_identity(Path(a.output))
    if a.worker:
        worker(a)
    elif a.phase == 'select':
        select(a)
    elif a.phase == 'summarize':
        summarize(a)
    else:
        out = Path(a.output)
        check_identity(out)
        policy = load(out / 'selection_policy.json')
        if a.timeout != policy['timeout_seconds'] or a.prepare_timeout != policy['preparation_timeout_seconds']:
            raise RuntimeError('Resource budget differs from frozen policy')
        rows = load(out / 'manifest.json')
        if a.ids:
            rows = [r for r in rows if r['id'] in a.ids.split(',')]
        groups = a.groups.split(',') if a.groups else ['A', 'B', 'D'] if a.phase == 'prepare' else list(GROUPS)
        tasks = [(r, g) for r in rows for g in groups]
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            list(pool.map(lambda args: launch(a, *args), tasks))


if __name__ == '__main__':
    main()
