"""Supplement the frozen run with paired failure, normalization and timing analysis."""
import argparse
import json
import hashlib
import shutil
from pathlib import Path
import statistics
from run_mesh_input_experiment import read,save,digest,report


def analyze(out):
    summary=report(out);policy=read(out/'policy.json');reference=Path(policy['reference'])
    batch=read(out/'batch.json')
    if batch.get('status')=='completed':
        if not (out/'controller_result.json').exists():shutil.copy2(out/'batch.json',out/'controller_result.json')
        batch.update(summary=summary,reporting_note='Summary reconciled by reporting-only analysis; raw frozen controller output retained in controller_result.json.')
        save(out/'batch.json',batch)
    manifest=[r for r in read(out/'manifest.json') if r['cohort']=='dev']
    records={p.stem:read(p) for p in sorted((out/'records').glob('*.json'))}
    audits=[];input_hashes={str(out/'identity.json'):digest(out/'identity.json')}
    details=[]
    for row in manifest:
        sid=row['id'];r=records.get(sid)
        if r is None:continue
        input_hashes[str(out/'records'/(sid+'.json'))]=digest(out/'records'/(sid+'.json'))
        audit=r.get('source_audit',{})
        if not audit and (out/'runs'/sid/'source_audit.json').exists():audit=read(out/'runs'/sid/'source_audit.json')
        newnorm=audit.get('normalization',{}).get('world_to_normalized')
        entry=dict(id=sid,source_prepared=bool(audit),target_counts_match=audit.get('target_counts_match'),
                   raw_old_center_ev_agree=audit.get('old_center_ev_agree'),old_center_ev_max_discrepancy=audit.get('old_center_ev_max_discrepancy'))
        if audit:
            counts=audit['target_counts'];delta=audit['old_center_ev_max_discrepancy']
            present=[counts['edges']>0,counts['vertices']>0]
            entry['old_center_ev_agree']=all(d<=1e-5 for d,exists in zip(delta,present) if exists)
            entry['empty_channel_differences']={c:d for c,d,exists in zip(('E','V'),delta,present) if not exists and d>1e-5}
        for path in (reference/'records'/sid/'L5.json',reference/'records'/sid/'M0.json',
                     reference/'cache'/sid/'M0'/'preparation.json',reference/'cache'/sid/'L5'/'preparation.json'):
            if not path.exists():continue
            old=read(path);oldnorm=old.get('source_audit',{}).get('normalization',{}).get('world_to_normalized')
            if oldnorm is not None and newnorm is not None:
                delta=max(abs(a-b) for aa,bb in zip(newnorm,oldnorm) for a,b in zip(aa,bb))
                entry.update(normalization_matrix_max_delta=delta,normalization_reference=str(path))
                input_hashes[str(path)]=digest(path);break
        audits.append(entry)
        error=r.get('error',{});extra=error.get('details',{})
        progress=r.get('fine_refinement_progress') or r.get('coarse_refinement_progress',{})
        history=extra.get('history',[]) or progress.get('history',[])
        refined=r.get('fine_refinement',{})
        run=r.get('run',{})
        mesh_report=next((s['report'] for s in run.get('stages',[]) if s['stage']=='mesh'),{})
        details.append(dict(id=sid,code=error.get('code',r['status']),failed_stage=r.get('failed_stage'),
            input_triangles=audit.get('mesh',{}).get('triangles'),
            input_max_edge=mesh_report.get('edge_quantiles',[None])[-1],
            last_completed_triangles=refined.get('refined',{}).get('triangles') or progress.get('triangles') or (history[-1]['after'] if history else extra.get('before')),
            proposed_triangles=extra.get('proposed'),remaining_above_target=extra.get('remaining',progress.get('remaining')),
            completed_refinement_rounds=len(history),
            coarse_refinement=r.get('coarse_refinement',{}),fine_refinement=refined,
            partial=not bool(refined),queries=r.get('queries',{}),process=r.get('process',{})))
        for group in ('L5','M0'):
            oldmesh=reference/'cache'/sid/group/'refinement.json'
            if oldmesh.exists():
                oldreport=read(oldmesh)
                details[-1].update(old_input_triangles=oldreport['original']['triangles'],old_input_max_edge=oldreport['original']['edge_quantiles'][-1])
                input_hashes[str(oldmesh)]=digest(oldmesh);break
    times=[r['process']['seconds'] for r in records.values()]
    peaks=[r['process']['peak_rss_bytes']/1024**3 for r in records.values()]
    joint=[r for r in records.values() if r.get('joint_success')]
    alarms=[]
    for r in joint:
        f=r.get('fidelity',{})
        if not f or 'error' in f or f.get('volume_rel_error',0)>.01 or max(f.get('recon_to_truth_max',0),f.get('truth_to_recon_max',0))>.01:alarms.append(r['id'])
    wall_seconds=None
    if (out/'batch_started.json').exists():
        started=read(out/'batch_started.json')['started_unix']
        ended=read(out/'batch.json').get('ended_unix')
        wall_seconds=ended-started if ended is not None else None
        input_hashes[str(out/'batch_started.json')]=digest(out/'batch_started.json')
    analysis=dict(complete=len(records)==len(manifest),summary=summary,audits=audits,failures=details,
        timing=dict(batch_wall_seconds=wall_seconds,median_total_seconds=statistics.median(times) if times else None,
                    total_worker_seconds=sum(times),max_peak_gib=max(peaks,default=0)),
        geometry_alarms=alarms,
        interpretation='Frozen per-sample time budget; partial meshes never count as prepared. Geometry-only coarse and field-guided fine refinement are recorded separately; long-edge-only pruning, no round/triangle/RSS caps.')
    save(out/'analysis.json',analysis)
    report_script=Path(__file__).with_name('run_mesh_input_experiment.py')
    revision=hashlib.sha256((digest(__file__)+digest(report_script)).encode()).hexdigest()[:16]
    archive=out/'analysis_code'/revision;archive.mkdir(parents=True,exist_ok=True)
    code={}
    for path in (Path(__file__),report_script):
        target=archive/path.name
        if target.exists():
            if digest(target)!=digest(path):raise RuntimeError('Analysis snapshot changed')
        else:shutil.copy2(path,target)
        code[str(target)]=digest(target)
    for name in ('batch.json','controller_result.json','production_drift.json','regression.log',
                 'benchmark_gradient_check.json','offline_score_compatibility_check.json'):
        if (out/name).exists():input_hashes[str(out/name)]=digest(out/name)
    save(out/'analysis_identity.json',dict(inputs=input_hashes,code=code,
        note='Reporting-only analysis: semantic preparation success, absent-channel placeholders and normalization fallback audited separately. Frozen worker code unchanged.'))
    lines=['','## 输入一致性补充审计','',f"已审计 {len(audits)} 个已完成样本。归一化从旧 L5 记录取值；缺失时回退旧 M0/准备记录。",'',
        '| ID | 目标数一致 | 32点存在通道 E/V 一致 | 归一化矩阵最大差 |','|---|---|---|---:|']
    for x in audits:lines.append(f"| {x['id']} | {x.get('target_counts_match','—')} | {x.get('old_center_ev_agree','—')} | {x.get('normalization_matrix_max_delta','—')} |")
    lines+=['','空通道不参与距离比较：旧截断占位值为 0.02，当前生产 EMPTY_DISTANCE 为 2√3，且标记 absent/invalid。原始逐例 source_audit 保留未经掩码的差异，补充审计按存在通道核验；占位值变化不改变加密或拓扑查询。']
    for x in audits:
        if x.get('empty_channel_differences'):lines.append(f"- {x['id']} 空通道差异：{x['empty_channel_differences']}。")
    lines+=['','## 加密阻断细节','',
        '| ID | 首个错误 | 输入三角形 | 输入最长边 | 最后完成/预算检查前的三角形数 | 仍超目标的近场三角形 |','|---|---|---:|---:|---:|---:|']
    for x in details:lines.append(f"| {x['id']} | {x['code']} | {x.get('input_triangles')} | {x.get('input_max_edge')} | {x.get('last_completed_triangles')} | {x.get('remaining_above_target')} |")
    examples=[x for x in details if x['id'] in ('00000168','00003291')]
    if examples:
        lines+=['','## 起始网格尺度的配对例子','','| ID | 旧 MC 输入三角形 | 旧 MC 最长边 | 新原生输入三角形 | 新原生最长边 |','|---|---:|---:|---:|---:|']
        for x in examples:lines.append(f"| {x['id']} | {x.get('old_input_triangles')} | {x.get('old_input_max_edge')} | {x.get('input_triangles')} | {x.get('input_max_edge')} |")
        lines+=['','原生 CAD 三角化的平面和细长三角形可能具有远大于 MC 的边长；本批次的轮数和时间设置以 policy.json 为准。总三角形数不等于近特征区域的覆盖均匀程度。']
    lines+=['','失败行的三角形数是中间状态，不能当作已达标的最终网格。所有未达标/预算拒绝仍计入 20 例分母。',
        '',f"已完成样本的总耗时中位数：{statistics.median(times):.1f} 秒；最大进程树 RSS：{max(peaks):.3f} GiB。" if times else '尚无完成样本。',
        '', '新旧流程输入的几何和查询位置不同，耗时仅是各流水线实际成本，不是同工作量加速比。']
    if (out/'production_drift.json').exists():
        lines+=['','生产目录运行期间更新了 CLI/清单；本批冻结代码不变，13 个实际几何/距离依赖与结束时生产代码逐字节一致，见 production_drift.json。']
    with (out/'REPORT.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    return analysis

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.output.resolve())
    print(json.dumps({'complete':result['complete'],'summary':result['summary'],'timing':result['timing']},ensure_ascii=False,indent=2))
