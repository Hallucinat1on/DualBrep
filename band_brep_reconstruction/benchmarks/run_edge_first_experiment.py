"""Frozen 20-dev comparison. Each worker runs both routes on one STEP."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'veudf/src'), str(ROOT / 'band_brep_reconstruction')]

from bandbrep.cache import file_hash
from bandbrep.schemas import write_json

ROUTES = ('trace', 'mc')


def read(path):
    return json.loads(path.read_text())


def initialize(output, workers):
    from bandbrep.config import Config

    history = ROOT / 'band_brep_reconstruction/outputs'
    old_full = history / 'single_pass_t003_20261106_v1/coarse_003'
    old_curve = history / 'curve_distance_coarse003_20261109_v1'
    old_region = history / 'region_repair_coarse003_20261116_v1'
    rows = [dict(id=r['id'], step=r['step'], source_sha256=r['source_sha256'])
            for r in read(old_region / 'manifest.json') if r['cohort'] == 'dev']
    ids = {r['id'] for r in rows}
    if len(rows) != 20 or len(ids) != 20:
        raise ValueError('Expected 20 distinct dev samples')
    for group in (old_full, old_curve):
        other = {r['id']: r['source_sha256'] for r in read(group / 'manifest.json') if r['cohort'] == 'dev'}
        if other != {r['id']: r['source_sha256'] for r in rows}:
            raise ValueError(f'Historical cohort differs: {group}')
    baseline = {name: {} for name in ('11-06 完整流程', '11-09 曲线回放', '11-16 baseline')}
    joint_curve = set(read(old_curve / 'summary.json')['paired']['common'])
    evidence = {str(group / 'manifest.json'): file_hash(group / 'manifest.json')
                for group in (old_full, old_curve, old_region)}
    evidence[str(old_curve / 'summary.json')] = file_hash(old_curve / 'summary.json')
    for row in rows:
        sid = row['id']
        if file_hash(row['step']) != row['source_sha256']:
            raise ValueError(f'Source STEP changed: {sid}')
        for name, path in zip(baseline, (old_full / 'records' / f'{sid}.json',
                                        old_curve / 'records' / f'{sid}.json',
                                        old_region / 'records' / sid / 'baseline.json')):
            item = read(path)
            evidence[str(path)] = file_hash(path)
            timing = item.get('stage_deltas', item.get('timings', {})).copy()
            if name.startswith('11-06'):
                previous = 0.
                for stage in (item.get('run') or {}).get('stages', []):
                    end = stage['elapsed_seconds']
                    timing[stage['stage']] = end - previous
                    previous = end
            if 'curve_recovery' in timing and 'global_coverage' in timing:
                timing['curves'] = timing['curve_recovery'] + timing['global_coverage']
            baseline[name][sid] = dict(status=item['status'], timings=timing,
                                      joint=(sid in joint_curve if name.startswith('11-09') else item['joint_success']),
                                      worker_seconds=item['process']['seconds'])
    output.mkdir(parents=True, exist_ok=False)
    identity = {}
    for root in ('band_brep_reconstruction/bandbrep', 'band_brep_reconstruction/benchmarks', 'veudf/src/veudf'):
        for source in sorted((ROOT / root).rglob('*.py')):
            rel = source.relative_to(ROOT)
            path = output / 'source_snapshot' / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)
            identity[rel.as_posix()] = file_hash(path)
    write_json(output / 'manifest.json', rows)
    write_json(output / 'baseline.json', baseline)
    write_json(output / 'policy.json', dict(workers=workers, threads_per_worker=1,
                                          config=Config(extraction='both').to_dict(),
                                          timeout=10800, order=list(ROUTES)))
    write_json(output / 'identity.json', dict(code=identity, historical=evidence,
               frozen={name: file_hash(output / name) for name in ('manifest.json', 'baseline.json', 'policy.json')}))
    report(output)


def initialize_budget_retry(primary, output, limit):
    """Reuse settled outcomes and rerun only allocation failures with the same code."""
    if read(primary / 'state.json')['status'] != 'completed':
        raise ValueError('The primary batch must be complete before preparing the retry')
    rows = read(primary / 'manifest.json')
    policy = read(primary / 'policy.json')
    old_limit = policy['config']['scout_point_budget']
    if limit <= old_limit:
        raise ValueError('The retry budget must exceed the primary budget')
    records = {r['id']: read(primary / 'records' / (r['id'] + '.json')) for r in rows}
    retry = sorted(sid for sid, record in records.items()
                   if record.get('error', {}).get('code') in ('scout_block_budget', 'scout_corner_budget'))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(primary / 'source_snapshot', output / 'source_snapshot')
    for name in ('manifest.json', 'baseline.json'):
        shutil.copyfile(primary / name, output / name)
    policy['config']['scout_point_budget'] = limit
    write_json(output / 'policy.json', policy)
    carried = sorted(set(records) - set(retry))
    (output / 'records').mkdir()
    for sid in carried:
        write_json(output / 'records' / (sid + '.json'), dict(records[sid], carried_from=str(primary)))
    write_json(output / 'retry.json', dict(primary=str(primary), retried=retry, carried=carried,
                                         old_scout_point_budget=old_limit, new_scout_point_budget=limit))
    identity = read(primary / 'identity.json')
    identity['frozen'] = {name: file_hash(output / name) for name in
                          ('manifest.json', 'baseline.json', 'policy.json', 'retry.json')}
    identity['frozen'].update({f'records/{sid}.json': file_hash(output / 'records' / (sid + '.json'))
                               for sid in carried})
    write_json(output / 'identity.json', identity)
    text = (f'# 角点预算补跑\n\n原批次：`{primary.name}`。原始记录保留。\n\n'
            f'去重前预算从 {old_limit} 增至 {limit}。仅重跑触发 scout 预算的 {len(retry)} 例。'
            '代码、裁剪规则和几何参数一致。其余结果沿用原运行并标记 `carried_from`。\n\n'
            f'补跑 ID：{", ".join(retry)}。\n\n汇总见 [REPORT.md](REPORT.md)。配对 worker 的中位数混合原运行和重跑，'
            '按实际执行代次解释墙钟。\n')
    (output / 'BUDGET_RETRY.md').write_text(text)
    report(output)


def worker(output, sid):
    from bandbrep.cad_source import load_cad_source
    from bandbrep.config import Config
    from bandbrep.deadline import sample_budget
    from bandbrep.edge_first import reconstruct
    from evaluate_step import evaluate

    row = next(r for r in read(output / 'manifest.json') if r['id'] == sid)
    cfg = Config(**read(output / 'policy.json')['config']).validate()
    folder = output / 'runs' / sid
    started = time.monotonic()
    entry = dict(id=sid, routes={})
    try:
        with sample_budget(cfg.sample_timeout_seconds):
            source = Path(row['step'])
            if file_hash(source) != row['source_sha256']:
                raise ValueError('Source STEP changed')
            mesh, udf = load_cad_source(source, row['source_sha256'])
            entry['prepare_seconds'] = time.monotonic() - started
            entry['input_triangles'] = len(mesh.triangles)
            reconstruct(mesh, udf, cfg, folder)
    except Exception as exc:
        entry['error'] = dict(code=getattr(exc, 'code', type(exc).__name__), message=str(exc))
    for route in ROUTES:
        sub = folder / route
        if (sub / 'run.json').exists():
            run = read(sub / 'run.json')
            entry['routes'][route] = run
            if run['status'] == 'solid_validated':
                tick = time.monotonic()
                try:
                    evaluation = evaluate(Path(row['step']), sub)
                    write_json(sub / 'score.json', evaluation)
                    run['score'] = evaluation
                except Exception as exc:
                    run['score_error'] = repr(exc)
                run['scoring_seconds'] = time.monotonic() - tick
    if (folder / 'run.json').exists():
        entry['shared'] = read(folder / 'run.json')
    entry['worker_seconds'] = time.monotonic() - started
    entry['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(output / 'records' / f'{sid}.json', entry)


def launch(output, row, timeout):
    sid = row['id']
    if (output / 'records' / f'{sid}.json').exists():
        return sid
    if (output / 'runs' / sid).exists():
        raise RuntimeError(f'Partial worker exists: {sid}')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', sid, '--output', str(output)]
    started = time.monotonic()
    with (output / 'logs' / f'{sid}.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / 'active' / f'{sid}.json', dict(id=sid, pid=process.pid, unix=time.time()))
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = 'timeout'
    record = output / 'records' / f'{sid}.json'
    if not record.exists():
        folder = output / 'runs' / sid
        partial = {r: read(folder / r / 'run.json') for r in ROUTES if (folder / r / 'run.json').exists()}
        for run in partial.values():
            if run['status'] == 'running':
                run['status'] = 'failed'
        entry = dict(id=sid, routes=partial, worker_seconds=time.monotonic()-started,
                     error=dict(code='worker_timeout' if code == 'timeout' else 'worker_exit', message=str(code)))
        if (folder / 'run.json').exists():
            entry['shared'] = read(folder / 'run.json')
        write_json(record, entry)
    (output / 'active' / f'{sid}.json').unlink()
    return sid


def report(output):
    rows = read(output / 'manifest.json')
    baseline = read(output / 'baseline.json')
    records = {p.stem: read(p) for p in (output / 'records').glob('*.json')}
    total = len(rows)
    summary = dict(total=total, completed=len(records), routes={}, baseline={})
    lines = ['# 边优先配对实验', '', f'已完成 {len(records)}/{total} 例，未结算样本单列。', '',
             '源 STEP 与历史 20 例一致。本轮从 STEP 重建网格和精确 E/V，共享粗细查询后分别执行 trace、局部 MC。',
             '11-06 从头运行，11-09 复用已定位顶点及前缀，11-16 baseline 复用网格及中心场。',
             '历史阶段的计时范围和并发不同。worker 总耗时不能当作同阶段算法加速比。', '',
             '| 方法 | 结算 | STEP | 图+STEP | worker 中位秒 |', '|---|---:|---:|---:|---:|']

    def stats(values):
        nums = [v for v in values if v is not None]
        return dict(n=len(nums), median=statistics.median(nums) if nums else None)

    def fmt(stat):
        return '—' if stat['n'] == 0 else f"{stat['median']:.2f} ({stat['n']})"

    def joint(run):
        return run.get('status') == 'solid_validated' and bool(run.get('score', {}).get('topology', {}).get('graph_good'))

    stage_rows = []
    for name, items in baseline.items():
        values = list(items.values())
        stages = {k: stats(r['timings'].get(k) for r in values) for k in sorted(set().union(*(r['timings'] for r in values)))}
        worker_time = stats(r['worker_seconds'] for r in values)
        s = dict(step=sum(r['status'] == 'solid_validated' for r in values), joint=sum(r['joint'] for r in values),
                 worker=worker_time, stages=stages)
        summary['baseline'][name] = s
        lines.append(f"| {name} | {len(values)}/{total} | {s['step']} | {s['joint']} | {fmt(worker_time)} |")
        stage_rows.extend((name, k, v) for k, v in stages.items())
    failures = []
    for name in ROUTES:
        values = {sid: r.get('routes', {}).get(name, {}) for sid, r in records.items()}
        stages = {k: stats(r.get('stages', {}).get(k, {}).get('seconds') for r in values.values())
                  for k in ('wireframe', 'split_mesh', 'surfaces', 'cad')}
        stages['scoring'] = stats(r.get('scoring_seconds') for r in values.values())
        for k in ('vertices', 'project', 'paths', 'fit', 'coverage'):
            stages['wireframe.' + k] = stats(r.get('stages', {}).get('wireframe', {}).get('timings', {}).get(k)
                                            for r in values.values())
        if name == 'mc':
            stages['wireframe.mc'] = stats(r.get('stages', {}).get('wireframe', {}).get('shell', {}).get('seconds')
                                          for r in values.values())
        old = baseline['11-16 baseline']
        gained = [sid for sid, run in values.items() if joint(run) and not old[sid]['joint']]
        lost = [sid for sid, run in values.items() if old[sid]['joint'] and not joint(run)]
        codes = Counter(r.get('error', records[sid].get('error', {})).get('code', 'unknown')
                        for sid, r in values.items() if r.get('status') != 'solid_validated')
        s = dict(step=sum(r.get('status') == 'solid_validated' for r in values.values()),
                 joint=sum(map(joint, values.values())), stages=stages, gained=gained, lost=lost,
                 failures=dict(codes), score_errors=sum('score_error' in r for r in values.values()))
        summary['routes'][name] = s
        lines.append(f"| {name} | {len(values)}/{total} | {s['step']} | {s['joint']} | 共用配对 worker |")
        failures.append(f"- {name} 相对 11-16 baseline 新增 {gained}，退化 {lost}，失败 {dict(codes)}。")
        stage_rows.extend((name, k, v) for k, v in stages.items())

    shared = dict(prepare=stats(r.get('prepare_seconds') for r in records.values()),
                  scout=stats(r.get('shared', {}).get('stages', {}).get('scout', {}).get('seconds') for r in records.values()),
                  paired_worker=stats(r.get('worker_seconds') for r in records.values()))
    for k in ('blocks', 'coarse', 'fine'):
        shared['scout.' + k] = stats(r.get('shared', {}).get('stages', {}).get('scout', {}).get('timings', {}).get(k)
                                     for r in records.values())
    summary['shared'] = shared
    paired = {}
    for stage in ('wireframe', 'split_mesh', 'surfaces', 'cad'):
        both = [{name: r.get('routes', {}).get(name, {}).get('stages', {}).get(stage, {}).get('seconds') for name in ROUTES}
                for r in records.values()]
        paired[stage] = stats(row['mc']-row['trace'] for row in both if all(v is not None for v in row.values()))
    summary['paired_mc_minus_trace_seconds'] = paired
    lines.extend(['', '## 阶段墙钟秒（括号内为完成该阶段的例数）', '',
                  '| 方法 | 阶段 | 中位秒（n） |', '|---|---|---:|'])
    lines.extend(f'| {name} | {stage} | {fmt(stat)} |' for name, stage, stat in
                 [*(('共用', k, v) for k, v in shared.items()), *stage_rows])
    lines.extend(['', '同一例两个方法均完成该阶段时，MC 减 trace 的中位秒数：', '',
                  *(f'- `{k}` {fmt(v)}' for k, v in paired.items()), '', *failures, '',
                  '失败阶段部分耗时保存在逐例记录中，未作为完成阶段计时。图评分依赖独立 CAD 对照。',
                  '旧 `curves` 含曲线与覆盖，新 `wireframe` 还包含顶点及分支候选，所以两列并非完全同工序。'])
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id', 'method', 'status', 'joint', 'prepare_seconds', 'scout_seconds',
                     'wireframe_seconds', 'split_mesh_seconds', 'surfaces_seconds', 'cad_seconds', 'failed_stage', 'error'])
    for row in rows:
        sid = row['id']
        for name in ROUTES:
            entry = records.get(sid, {})
            run = entry.get('routes', {}).get(name, {})
            writer.writerow([sid, name, run.get('status', 'not_reached' if sid in records else 'pending'),
                             joint(run) if sid in records else '', entry.get('prepare_seconds'),
                             entry.get('shared', {}).get('stages', {}).get('scout', {}).get('seconds'),
                             *(run.get('stages', {}).get(k, {}).get('seconds') for k in ('wireframe', 'split_mesh', 'surfaces', 'cad')),
                             run.get('failed_stage'), run.get('error', entry.get('error', {})).get('code')])
    write_json(output / 'summary.json', summary)
    for name, text in [('REPORT.md', '\n'.join(lines)+'\n'), ('metrics.csv', buf.getvalue())]:
        temp = output / (name + '.tmp')
        temp.write_text(text)
        temp.replace(output / name)
    return summary


def run(output):
    from datetime import datetime, timezone

    identity = read(output / 'identity.json')
    for scope, directory in (('code', ROOT), ('frozen', output)):
        for name, expected in identity[scope].items():
            if file_hash(directory / name) != expected:
                raise RuntimeError(f'Frozen experiment file changed: {name}')
    policy = read(output / 'policy.json')
    output.joinpath('logs').mkdir(exist_ok=True)
    output.joinpath('active').mkdir(exist_ok=True)
    output.joinpath('records').mkdir(exist_ok=True)
    state = dict(status='running', pid=os.getpid(), started=datetime.now(timezone.utc).isoformat())
    write_json(output / 'state.json', state)
    try:
        with ThreadPoolExecutor(policy['workers']) as pool:
            pending = [pool.submit(launch, output, row, policy['timeout']) for row in read(output / 'manifest.json')]
            for future in as_completed(pending):
                print('settled', future.result(), flush=True)
                report(output)
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='controller_failed', error=repr(exc))
        raise
    finally:
        state['ended'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--retry-budget', type=int, default=16000000)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--initialize', action='store_true')
    group.add_argument('--initialize-budget-retry', type=Path)
    group.add_argument('--run', action='store_true')
    group.add_argument('--report', action='store_true')
    group.add_argument('--worker')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.initialize:
        if args.workers < 1:
            parser.error('--workers must be positive')
        initialize(output, args.workers)
    elif args.initialize_budget_retry:
        initialize_budget_retry(args.initialize_budget_retry.resolve(), output, args.retry_budget)
    elif args.run:
        run(output)
    elif args.report:
        report(output)
    else:
        worker(output, args.worker)
