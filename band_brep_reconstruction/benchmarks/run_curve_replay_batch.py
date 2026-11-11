"""Frozen 20-dev curve/STEP replay on one historical coarse-target group."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import csv
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import psutil
from run_mesh_input_experiment import read,save,digest
from replay_curve_recovery import initialize as initialize_case


def initialize(group,out):
    if out.exists():raise RuntimeError('Use a new batch directory')
    rows=[r for r in read(group/'manifest.json') if r['cohort']=='dev']
    if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise RuntimeError('Expected the fixed 20 dev IDs')
    out.mkdir(parents=True)
    save(out/'manifest.json',rows)
    save(out/'campaign.json',dict(input_group=str(group),method='direct_distance',stop_after='solid',
        total=20,jobs=2,sample_timeout_seconds=10800.,
        coarse_edge=.03,fine_edge=.003,face_threshold1=.003,face_second_pass=False,
        scope='Reuse exact historical working meshes, bands, topology and vertices; rerun curves through STEP and geometric scoring.',
        comparison='Face and graph scores belong to fixed input topology. Compare STEP success, geometric fidelity, and curve-stage cost.'))
    immutable=[out/'manifest.json',out/'campaign.json',group/'manifest.json',group/'policy.json']
    snapshots=[]
    for row in rows:
        sid=row['id'];print('Freezing',sid,flush=True)
        initialize_case(group,sid,out/'runs'/sid,'direct_distance','solid')
        identity=read(out/'runs'/sid/'identity.json');snapshots.append(identity['code'])
        cfg=read(out/'runs'/sid/'policy.json')['config']
        if (cfg['refinement_coarse_edge'],cfg['refinement_target_edge'],cfg['face_threshold1'],cfg['face_second_pass'])!=(.03,.003,.003,False):
            raise RuntimeError('Input group differs from the requested coarse0.03/fine0.003/single-pass0.003 policy')
        immutable += [out/'runs'/sid/'identity.json',group/'records'/(sid+'.json')]
    if any(s!=snapshots[0] for s in snapshots):raise RuntimeError('Replay code changed while freezing cases')
    save(out/'identity.json',dict(files={str(p):digest(p) for p in immutable}))
    update_report(out)


def verify(out):
    for name,h in read(out/'identity.json')['files'].items():
        if digest(name)!=h:raise RuntimeError('Frozen batch input changed: '+name)


def launch(out,row):
    sid=row['id'];record=out/'records'/(sid+'.json')
    if record.exists():return read(record)
    folder=out/'runs'/sid
    if (folder/'worker_started.json').exists():raise RuntimeError('Unrecorded partial worker; inspect before resuming: '+sid)
    save(folder/'worker_started.json',dict(unix=time.time()))
    snapshot=folder/'source_snapshot';script=snapshot/'band_brep_reconstruction/benchmarks/replay_curve_recovery.py'
    env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    env['PYTHONPATH']=str(snapshot/'veudf/src')+':'+str(snapshot/'band_brep_reconstruction')
    command=[sys.executable,str(script),'--worker','--input-group',read(out/'campaign.json')['input_group'],
             '--id',sid,'--output',str(folder),'--method','direct_distance','--stop-after','solid']
    started=time.monotonic();peak=0;timed_out=False
    with (folder/'worker.log').open('x') as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timeout=read(out/'campaign.json')['sample_timeout_seconds']
        while process.poll() is None:
            try:
                p=psutil.Process(process.pid);memory=p.memory_info().rss
                for child in p.children(recursive=True):
                    try:memory+=child.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,memory)
            except psutil.NoSuchProcess:pass
            elapsed=time.monotonic()-started
            if elapsed>timeout:
                timed_out=True
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                break
            progress={}
            try:
                if (folder/'curve_recovery_progress.json').exists():
                    current=read(folder/'curve_recovery_progress.json')
                    progress={k:current.get(k) for k in ('active_edge','completed','timings')}
            except (OSError,ValueError):pass
            save(out/'active'/(sid+'.json'),dict(id=sid,pid=process.pid,elapsed_seconds=elapsed,
                 peak_rss_bytes=peak,curve_progress=progress))
            time.sleep(1)
    if (folder/'result.json').exists():result=read(folder/'result.json')
    else:result=dict(id=sid,status='failed',failed_stage='worker',error=dict(code='worker_exit',message='Worker exited without result.json'))
    if timed_out:result.update(status='failed',error=dict(code='sample_timeout',message='Replay worker exceeded total 10800 seconds'))
    if process.returncode and result.get('status')=='solid_validated':
        result.update(status='failed',error=dict(code='worker_exit',message='Nonzero worker exit after result'))
    result['process']=dict(pid=process.pid,returncode=process.returncode,seconds=time.monotonic()-started,peak_rss_bytes=peak)
    save(record,result);(out/'active'/(sid+'.json')).unlink(missing_ok=True)
    print(sid,result['status'],result.get('error',{}).get('code',''),flush=True)
    return result


def update_report(out):
    policy=read(out/'campaign.json');group=Path(policy['input_group']);rows=read(out/'manifest.json')
    old={r['id']:read(group/'records'/(r['id']+'.json')) for r in rows}
    records={p.stem:read(p) for p in sorted((out/'records').glob('*.json'))}
    success=lambda r:r.get('status')=='solid_validated'
    joint=lambda sid,r:success(r) and bool(old[sid]['score']['graph_good'])
    counts=dict(total=20,completed=len(records),pending=20-len(records),
        fixed_face_good=sum(bool(r['score']['face_good']) for r in old.values()),
        fixed_graph_good=sum(bool(r['score']['graph_good']) for r in old.values()),
        cad_step=sum(success(r) for r in records.values()),joint_success=sum(joint(sid,r) for sid,r in records.items()))
    counts['historical']=dict(cad_step=sum(success(r) for r in old.values()),joint_success=sum(bool(r['joint_success']) for r in old.values()))
    counts['paired']=dict(gained=[sid for sid,r in records.items() if joint(sid,r) and not old[sid]['joint_success']],
        lost=[sid for sid,r in records.items() if old[sid]['joint_success'] and not joint(sid,r)],
        common=[sid for sid,r in records.items() if old[sid]['joint_success'] and joint(sid,r)])
    counts['failure_counts']=dict(Counter(r.get('error',{}).get('code','unknown') for r in records.values() if not success(r)))
    curve_times=[r['timings']['curve_recovery']+r['timings'].get('global_coverage',0) for r in records.values() if 'curve_recovery' in r.get('timings',{})]
    counts['median_completed_curve_and_coverage_seconds']=statistics.median(curve_times) if curve_times else None
    counts['max_peak_gib']=max((r['process']['peak_rss_bytes']/1024**3 for r in records.values()),default=0)
    save(out/'summary.json',counts)
    lines=['# 粗目标0.03工作网格：直接距离曲线恢复重跑','',
        f"已完成 {counts['completed']}/20；pending {counts['pending']}，10个check不运行。",'',
        '固定原粗0.03、E-only近场细0.003、单阶段面阈值0.003的工作网格、边带、拓扑和顶点；只重做曲线、曲面、STEP及几何评分。',
        '面/图评分属于固定输入，不能把它们当作本次重新分割或恢复拓扑的成绩。两CPU worker，每例从worker启动起总超时10800秒。','',
        '| 指标 | 原粗0.03组 | 本轮 |','|---|---:|---:|',
        f"| 固定面正确 | {counts['fixed_face_good']}/20 | 输入复用 |",f"| 固定图正确 | {counts['fixed_graph_good']}/20 | 输入复用 |",
        f"| CAD STEP | {counts['historical']['cad_step']}/20 | {counts['cad_step']}/20 |",
        f"| 图+STEP | {counts['historical']['joint_success']}/20 | {counts['joint_success']}/20 |",'',
        '## 逐例结果','','| ID | 原图+STEP | 新状态 | 新失败 | 新图+STEP | 曲线+覆盖秒 | worker秒 |', '|---|---|---|---|---|---:|---:|']
    csvrows=[]
    for row in rows:
        sid=row['id'];r=records.get(sid);timing=r.get('timings',{}) if r else {}
        seconds=timing.get('curve_recovery',0)+timing.get('global_coverage',0) if 'curve_recovery' in timing else None
        item=dict(id=sid,old_joint_success=old[sid]['joint_success'],status=r['status'] if r else 'pending',
            error=r.get('error',{}).get('code','') if r else '',joint_success=joint(sid,r) if r else None,
            curve_and_coverage_seconds=seconds,worker_seconds=r['process']['seconds'] if r else None)
        csvrows.append(item)
        lines.append(f"| {sid} | {item['old_joint_success']} | {item['status']} | {item['error'] or '—'} | {item['joint_success']} | {seconds} | {item['worker_seconds']} |")
    lines+=['','## 配对变化','','```json',json.dumps(counts['paired'],ensure_ascii=False,indent=2),'```','',
        '当前失败和超时留在20例分母，pending不记失败。曲线耗时中位仅统计已完成曲线恢复者，失败早退者不填0；与历史阶段的计时边界存在产物写入差异，完整分析须逐例比较。',
        '本轮使用精确CAD距离及采样差分梯度；直接距离优化不查询UDF梯度，复杂初始化和全局覆盖仍可能使用。单例验证产物不充当本批已运行记录。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    return counts


def run(out):
    verify(out);started=time.time();save(out/'state.json',dict(status='running',started_unix=started,controller_pid=os.getpid()))
    try:
        rows=read(out/'manifest.json')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(launch,out,row) for row in rows]
            for f in as_completed(futures):
                f.result();summary=update_report(out)
                save(out/'state.json',dict(status='running',started_unix=started,controller_pid=os.getpid(),summary=summary))
        verify(out);summary=update_report(out)
        save(out/'state.json',dict(status='completed',started_unix=started,ended_unix=time.time(),summary=summary))
    except BaseException as exc:
        save(out/'state.json',dict(status='controller_failed',started_unix=started,error=repr(exc)));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--input-group',type=Path)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--initialize',action='store_true');modes.add_argument('--run',action='store_true');modes.add_argument('--report',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.initialize:
        if args.input_group is None:parser.error('--initialize requires --input-group')
        initialize(args.input_group.resolve(),out)
    elif args.run:run(out)
    else:update_report(out)


if __name__=='__main__':main()
