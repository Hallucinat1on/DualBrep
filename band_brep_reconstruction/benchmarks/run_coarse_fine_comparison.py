"""Frozen 0.02/0.03 coarse-target comparison, with two workers across both groups."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import os
import shutil
import statistics
import time

from run_mesh_input_experiment import (MODULE, REPO, read, save, digest, freeze,
                                       verify, launch, report)

GROUPS = {'coarse_002': .02, 'coarse_003': .03}


def initialize(out, preflight):
    if out.exists():raise RuntimeError('Use a new comparison directory')
    audit = read(preflight)
    if not audit['all_twenty_match'] or len(audit['cases']) != 20:
        raise RuntimeError('All 20 production input meshes must match the historical cohort')
    out.mkdir(parents=True)
    shutil.copy2(preflight, out/'preflight.json')
    shutil.copy2(REPO/'veudf/SOURCE.json', out/'production_source.json')
    policy = dict(date=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
                  groups=GROUPS, dev=20, check=10, jobs=2, sample_timeout_seconds=10800.,
                  fine_edge=.005, face_threshold1=.005, face_threshold2=.01,
                  scheduling='interleave the same sample across groups; two workers total',
                  field_source='exact production CAD E/V and oracle presence; no learned model',
                  geometry_source='current production condition mesh; all 20 input fingerprints preflighted',
                  comparison='historical MC and native-four-round results are read only; no old workflow rerun')
    save(out/'campaign.json', policy)
    for group, coarse in GROUPS.items():
        freeze(out/group, MODULE/'outputs/mesh_density_20261028_v2',
               MODULE/'outputs/native_mesh_L5_20261101_v1', 10800., coarse)
    configs = [read(out/g/'policy.json')['config'] for g in GROUPS]
    normalized = [{k:v for k,v in c.items() if k != 'refinement_coarse_edge'} for c in configs]
    if normalized[0] != normalized[1]:raise RuntimeError('Groups differ beyond coarse target')
    identities = [read(out/g/'identity.json') for g in GROUPS]
    if identities[0]['code'] != identities[1]['code']:raise RuntimeError('Group source snapshots differ')
    immutable = ['campaign.json', 'preflight.json', 'production_source.json']
    immutable += [g+'/identity.json' for g in GROUPS]
    save(out/'comparison_identity.json', {'files':{n:digest(out/n) for n in immutable}})
    update_report(out)


def verify_comparison(out):
    for name, expected in read(out/'comparison_identity.json')['files'].items():
        if digest(out/name) != expected:raise RuntimeError('Comparison metadata changed: '+name)
    for group in read(out/'campaign.json')['groups']:verify(out/group)


def median(values):
    return statistics.median(values) if values else None


def update_report(out):
    policy = read(out/'campaign.json')
    groups = list(policy['groups'])
    all_records = {}
    results = {}
    for group in groups:
        folder = out/group
        records = {p.stem:read(p) for p in sorted((folder/'records').glob('*.json'))}
        all_records[group] = records
        finished = (folder/'batch.json').exists() and read(folder/'batch.json')['status'] == 'completed'
        current = read(folder/'summary.json') if finished else report(folder)
        metrics = current['groups'][read(folder/'policy.json')['group']]
        results[group] = dict(coarse_edge=policy['groups'][group], **metrics,
            median_total_seconds=median([r['process']['seconds'] for r in records.values()]),
            median_coarse_triangles=median([r['coarse_refinement']['refined']['triangles'] for r in records.values() if r.get('coarse_refinement')]),
            median_fine_triangles=median([r['fine_refinement']['refined']['triangles'] for r in records.values() if r.get('fine_refinement')]),
            median_evaluation_rows=median([r['queries']['evaluation_rows'] for r in records.values() if r.get('queries')]),
            failure_counts=current['failure_counts'])
    a, b = (all_records[g] for g in groups)
    common = sorted(set(a) & set(b))
    paired = dict(completed_pairs=len(common),
        candidate_only_success=[sid for sid in common if b[sid].get('joint_success') and not a[sid].get('joint_success')],
        baseline_only_success=[sid for sid in common if a[sid].get('joint_success') and not b[sid].get('joint_success')],
        common_success=[sid for sid in common if a[sid].get('joint_success') and b[sid].get('joint_success')],
        baseline_median_seconds=median([a[sid]['process']['seconds'] for sid in common]),
        candidate_median_seconds=median([b[sid]['process']['seconds'] for sid in common]))
    complete = all(len(records) == policy['dev'] for records in all_records.values())
    summary = dict(status='completed' if complete else 'running',
                   completed=sum(len(r) for r in all_records.values()), total=2*policy['dev'],
                   groups=results, paired=paired)
    save(out/'summary.json', summary)
    lines = ['# 粗加密 0.02 / 0.03：两级后处理配对实验', '',
             f"已完成 {summary['completed']}/{summary['total']} 个样本×配置任务；{len(common)}/20 对完成。10 个 check 不运行。", '',
             '细目标固定 0.005，E/V 选区 0.02/0.03，分割阈值 0.005/0.01；仅切超长边。',
             '每例总超时 3 小时，无轮数、三角形或 RSS 硬上限；两组共用两个 CPU worker，不使用 GPU。', '',
             '| 粗目标 | 已完成/20 | 加密达标 | 面正确 | 图正确 | CAD STEP | 图+STEP | 已完成者耗时中位数(秒) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, x in results.items():
        lines.append(f"| {x['coarse_edge']} | {x['completed']} | {x['prepared']} | {x['face_good']} | {x['graph_good']} | {x['cad_step']} | {x['joint_success']} | {x['median_total_seconds']} |")
    lines += ['', '## 逐例对照', '',
              '| ID | 粗0.02状态/错误 | 粗0.03状态/错误 | 粗0.02秒 | 粗0.03秒 |',
              '|---|---|---|---:|---:|']
    rows = [r for r in read(out/groups[0]/'manifest.json') if r['cohort'] == 'dev']
    for row in rows:
        sid = row['id']; left = a.get(sid, {}); right = b.get(sid, {})
        def status(r):return r.get('error', {}).get('code', r.get('status', 'pending'))
        lines.append(f"| {sid} | {status(left)} | {status(right)} | {left.get('process',{}).get('seconds','—')} | {right.get('process',{}).get('seconds','—')} |")
    lines += ['', '## 完成配对的变化', '', '```json', json.dumps(paired, ensure_ascii=False, indent=2), '```', '',
              '运行中统计包含失败和超时样本；pending 不记作失败。耗时中位数基于当前已完成样本，须等待全部完成再选型。',
              '本批使用精确 CAD 场，不能解释为真实网络推理成绩。历史 MC/四轮成绩见各组 REPORT.md，未重跑或覆盖历史结果。', '',
              '- [粗0.02明细](coarse_002/REPORT.md)', '- [粗0.03明细](coarse_003/REPORT.md)',
              '- [输入一致性预检](preflight.json)', '- [汇总 JSON](summary.json)',
              '- active/、logs/ 和逐例产物位于各组目录。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(out):
    verify_comparison(out)
    policy = read(out/'campaign.json'); groups = list(policy['groups'])
    rows = [r for r in read(out/groups[0]/'manifest.json') if r['cohort'] == 'dev']
    if not (out/'started.json').exists():
        save(out/'started.json', {'started_unix':time.time(), 'controller_pid':os.getpid()})
    started = read(out/'started.json')['started_unix']
    finished_groups = set()
    for group in groups:
        save(out/group/'batch_started.json', {'started_unix':started, 'controller_pid':os.getpid()})
        save(out/group/'batch.json', {'status':'running', 'started_unix':started, 'controller_pid':os.getpid(),
                                     'scheduling':'shared two-worker comparison'})
    save(out/'state.json', {'status':'running', 'started_unix':started, 'controller_pid':os.getpid()})
    try:
        with ThreadPoolExecutor(max_workers=policy['jobs']) as pool:
            futures = {pool.submit(launch, out/group, row):group for row in rows for group in groups}
            for future in as_completed(futures):
                group = futures[future]; future.result()
                summary = update_report(out)
                metrics = summary['groups'][group]
                if metrics['completed'] == policy['dev'] and group not in finished_groups:
                    save(out/group/'batch.json', dict(status='completed', started_unix=started,
                         ended_unix=time.time(), summary=report(out/group)))
                    from analyze_mesh_input_experiment import analyze
                    analyze(out/group);finished_groups.add(group)
                save(out/'state.json', dict(status='running', started_unix=started,controller_pid=os.getpid(),
                     completed=summary['completed'], total=summary['total']))
        summary = update_report(out)
        if summary['status'] != 'completed':raise RuntimeError('Comparison ended without all 40 records')
        verify_comparison(out)
        save(out/'state.json', dict(status='completed', started_unix=started,ended_unix=time.time(),
                                   controller_pid=os.getpid(), summary=summary))
    except BaseException as exc:
        save(out/'state.json', dict(status='controller_failed', started_unix=started,
                                   controller_pid=os.getpid(), error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--initialize', action='store_true'); mode.add_argument('--run', action='store_true')
    mode.add_argument('--report', action='store_true')
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args();out=args.output.resolve()
    if args.initialize:
        if args.preflight is None:parser.error('--initialize requires --preflight')
        initialize(out,args.preflight)
    elif args.run:run(out)
    else:print(json.dumps(update_report(out),ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
