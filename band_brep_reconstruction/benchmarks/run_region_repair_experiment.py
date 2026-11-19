"""Frozen five-arm scalar-only region repair experiment on fixed working meshes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from dataclasses import replace
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
import psutil
from run_mesh_input_experiment import REPO,MODULE,read,save,digest

ARMS={'baseline':dict(region_repair='off',face_mean_filter=True),
      'no_mean':dict(region_repair='off',face_mean_filter=False),
      'small':dict(region_repair='small',face_mean_filter=False),
      'single_face':dict(region_repair='single_face',face_mean_filter=False),
      'joint':dict(region_repair='joint',face_mean_filter=False)}


def initialize(group,comparison,out,jobs=2):
    if out.exists():raise RuntimeError('Use a new frozen output directory')
    if not 1<=jobs<=os.cpu_count():raise ValueError('Invalid CPU worker count')
    from bandbrep.config import load_config
    rows=[r for r in read(group/'manifest.json') if r['cohort']=='dev']
    if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise RuntimeError('Expected fixed 20-dev manifest')
    for name,h in read(group/'identity.json')['code'].items():
        if name.startswith('veudf/src/') and digest(REPO/name)!=h:raise RuntimeError('Scalar dependency changed: '+name)
    out.mkdir(parents=True);save(out/'manifest.json',rows)
    base=read(group/'runs'/rows[0]['id']/'reconstruction/run.json')['config']
    configs={name:load_config(overrides={**base,**options,'curve_recovery':'direct_distance',
        'save_queries':False,'stop_after':'solid'}).to_dict() for name,options in ARMS.items()}
    policy=dict(input_group=str(group),comparison=str(comparison),configs=configs,jobs=jobs,total=100,
        sample_timeout_seconds=10800.,scope='Same immutable input/working meshes and center scalar queries; recompute bands through STEP and offline scores for every arm.',
        fallback='joint -> small -> off; others -> off, only after a changed candidate fails, within one sample deadline; report raw candidates and selected result separately.')
    save(out/'campaign.json',policy)
    common=[out/'campaign.json',out/'manifest.json',group/'identity.json',group/'policy.json']
    cases={}
    names=('input_mesh.npz','working_mesh.npz','center_fields.npz','bands.npz','topology_labels.npz','incidence.json','vertices.json')
    for row in rows:
        sid=row['id'];original=group/'runs'/sid/'reconstruction';run=read(original/'run.json')
        files=[original/'run.json',group/'records'/(sid+'.json'),group/'runs'/sid/'source_audit.json',
               comparison/'records'/(sid+'.json'),Path(row['step'])]
        for name in names:
            if digest(original/name)!=run['artifacts'].get(name):raise RuntimeError('Bad cached artifact '+sid+'/'+name)
            files.append(original/name)
        if digest(row['step'])!=row['source_sha256']:raise RuntimeError('Source changed '+sid)
        cases[sid]={str(p):digest(p) for p in files}
        print('Frozen inputs',sid,flush=True)
    code={}
    for folder in (MODULE/'bandbrep',MODULE/'benchmarks',REPO/'veudf/src/veudf'):
        for source in sorted(folder.rglob('*.py')):
            relative=source.relative_to(REPO);target=out/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);code[str(relative)]=digest(target)
    save(out/'environment.json',dict(python=sys.version,executable=sys.executable,
        packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','pytest','psutil')}))
    common.append(out/'environment.json')
    save(out/'identity.json',dict(common={str(p):digest(p) for p in common},cases=cases,code=code))
    report(out)


def verify(out,sid=None):
    identity=read(out/'identity.json');files=dict(identity['common'])
    for name,h in identity['code'].items():files[str(out/'source_snapshot'/name)]=h
    for case,ff in identity['cases'].items():
        if sid is None or case==sid:files.update(ff)
    for name,h in files.items():
        if digest(name)!=h:raise RuntimeError('Frozen input or source changed: '+name)


def load_bands(folder):
    import numpy as np
    from bandbrep.schemas import BandState
    with np.load(folder/'bands.npz') as a:
        b=BandState(*(a[k] for k in ('points','values','edge_band','vertex_band','edge_region','core_mask')),
            face_labels=a['face_labels'],exclusion_reasons=a['exclusion_reasons'],
            vertex_labels=a['vertex_labels'] if 'vertex_labels' in a else None,
            vertex_centers=a['vertex_centers'] if 'vertex_centers' in a else None)
    return b


def worker(out,sid,arm):
    import fcntl
    import numpy as np
    from bandbrep.config import Config
    from bandbrep.deadline import sample_budget
    from bandbrep.field_provider import CallableUDFProvider
    from bandbrep.reconstruct import reconstruct_from_prepared
    from bandbrep.schemas import SurfaceMesh
    from bandbrep.cache import load_center_fields,load_topology
    from veudf.preprocess.cad import read_normalized,collect_targets,triangulate
    from veudf.preprocess.condition import weld
    from veudf.preprocess.config import Config as DistanceConfig
    from run_mesh_input_experiment import ExactUDF,offline_source
    from experiment_scoring import score,geometry_score,fidelity_score
    folder=out/'runs'/sid/arm;folder.mkdir(parents=True,exist_ok=False)
    result=dict(id=sid,arm=arm,status='failed',joint_success=False);started=time.monotonic();oracle=None
    policy=read(out/'campaign.json');group=Path(policy['input_group']);original=group/'runs'/sid/'reconstruction'
    cfg=Config(**policy['configs'][arm]).validate();reconstruction=None
    try:
        with sample_budget(cfg.sample_timeout_seconds):
            verify(out,sid)
            row=next(r for r in read(out/'manifest.json') if r['id']==sid)
            def mesh(name):
                with np.load(original/name) as a:return SurfaceMesh(a['vertices'],a['triangles'])
            source_mesh=mesh('input_mesh.npz');working=mesh('working_mesh.npz');centers=load_center_fields(original/'center_fields.npz')
            audit=read(group/'runs'/sid/'source_audit.json');source_mesh.report['normalization']=audit['normalization']
            dc=DistanceConfig().validate();shape,norm=read_normalized(row['step'],dc);targets=collect_targets(shape,dc)
            if dict(faces=len(targets.faces),edges=len(targets.edges),vertices=len(targets.vertices))!=audit['target_counts']:raise RuntimeError('Target semantics changed')
            np.testing.assert_allclose(norm['world_to_normalized'],audit['normalization']['world_to_normalized'],atol=1e-10,rtol=0)
            oracle=ExactUDF(targets,dc)
            udf=CallableUDFProvider(oracle.values,fingerprint='production-cad:'+row['source_sha256'],
                presence=('present' if targets.edges else 'absent','present' if len(targets.vertices) else 'absent'))
            provenance=dict(source=str(original),artifacts={name:read(original/'run.json')['artifacts'][name]
                for name in ('input_mesh.npz','working_mesh.npz','center_fields.npz')})
            try:
                reconstruction=reconstruct_from_prepared(source_mesh,working,centers,udf,cfg,folder/'reconstruction',provenance=provenance)
                result['status']='solid_validated'
            except Exception as exc:
                result.update(error=dict(code=getattr(exc,'code',type(exc).__name__),message=str(exc),details=getattr(exc,'details',{})))
            runpath=folder/'reconstruction/run.json'
            if runpath.exists():
                run=read(runpath);result['run']=run;result['failed_stage']=run.get('failed_stage')
                result['repair_attempts']=run.get('repair_attempts',[])
                result['effective_mode']=run.get('effective_config',{}).get('region_repair')
                stages=run.get('stages',[]);previous=0.;timings={}
                for stage in stages:
                    timings[stage['stage']]=stage['elapsed_seconds']-previous;previous=stage['elapsed_seconds']
                # Retried stage deltas include earlier attempts: use explicit total
                # time for cost comparisons, not misleading per-stage speedups.
                result['stage_deltas']=timings
            if result.get('error',{}).get('code')!='sample_timeout':
                raw=triangulate(targets,dc);_,mesh_oracle=weld(raw,dc);source=offline_source(targets,raw,mesh_oracle,norm)
                truthfile=out/'scoring_cache'/(sid+'.npz');truthfile.parent.mkdir(exist_ok=True)
                # Offline-only truth labels shared across arms; no reconstruction
                # module reads this cache or receives the source object.
                with (out/'scoring_cache'/(sid+'.lock')).open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    if truthfile.exists():
                        with np.load(truthfile) as a:truth=tuple(a[k] for k in ('face','edge','vertex'))
                    else:
                        points=centers.points
                        truth=(source.truth_face(points),source.edge.query(points)[2] if source.edge else np.full(len(points),-1),
                               source.vtree.query(points)[1] if source.vtree else np.full(len(points),-1))
                        np.savez_compressed(truthfile,face=truth[0],edge=truth[1],vertex=truth[2])
                        save(out/'scoring_cache'/(sid+'.identity.json'),dict(sha256=digest(truthfile),source=row['source_sha256'],
                            centers=provenance['artifacts']['center_fields.npz'],scope='offline scores only'))
                    fcntl.flock(lock,fcntl.LOCK_UN)
                identity=read(out/'scoring_cache'/(sid+'.identity.json'))
                if digest(truthfile)!=identity['sha256'] or identity['centers']!=provenance['artifacts']['center_fields.npz']:raise RuntimeError('Offline scoring cache changed')
                def score_folder(directory):
                    if not (directory/'incidence.json').exists():return None
                    b=load_bands(directory);np.testing.assert_array_equal(b.points,centers.points)
                    source._scoring_truth=(b.points,*truth)
                    value=score(load_topology(directory),b,source)
                    save(directory/'graph_score.json',value);return value
                result['score']=score_folder(folder/'reconstruction')
                for attempt in result.get('repair_attempts',[]):
                    if attempt.get('selected'):continue
                    matches=sorted((folder/'reconstruction/attempts').glob('*_'+attempt['mode']))
                    if matches:attempt['score']=score_folder(matches[0])
                if reconstruction is not None:
                    result['geometry']=geometry_score(reconstruction['graph'].vertices,reconstruction['curves'],source)
                    result['fidelity']=fidelity_score(reconstruction['solid'],source)
                result['joint_success']=result['status']=='solid_validated' and bool((result.get('score') or {}).get('graph_good'))
    except Exception as exc:
        result.update(status='failed',joint_success=False,error=dict(code=getattr(exc,'code',type(exc).__name__),message=str(exc)),traceback=traceback.format_exc())
    if oracle is not None:result['queries']=oracle.report()
    result['seconds']=time.monotonic()-started
    save(folder/'result.json',result)
    print(sid,arm,result['status'],result.get('error',{}).get('code',''),flush=True)


def launch(out,task):
    sid,arm=task;record=out/'records'/sid/(arm+'.json')
    if record.exists():return read(record)
    folder=out/'runs'/sid/arm
    if folder.exists():raise RuntimeError('Unrecorded partial result: '+str(folder))
    start=time.monotonic();peak=0;timeout=False
    snapshot=out/'source_snapshot';script=snapshot/'band_brep_reconstruction/benchmarks/run_region_repair_experiment.py'
    env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    env['PYTHONPATH']=str(snapshot/'veudf/src')+':'+str(snapshot/'band_brep_reconstruction')
    logpath=out/'logs'/sid/(arm+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('x') as log:
        p=subprocess.Popen([sys.executable,str(script),'--worker','--output',str(out),'--id',sid,'--arm',arm],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            elapsed=time.monotonic()-start
            try:
                process=psutil.Process(p.pid);rss=process.memory_info().rss
                for child in process.children(recursive=True):
                    try:rss+=child.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss)
            except psutil.NoSuchProcess:pass
            save(out/'active'/(sid+'_'+arm+'.json'),dict(id=sid,arm=arm,pid=p.pid,seconds=elapsed,peak_rss_bytes=peak))
            if elapsed>10800:
                timeout=True
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    p.wait()
                break
            time.sleep(1)
    result=read(folder/'result.json') if (folder/'result.json').exists() else dict(id=sid,arm=arm,status='failed',joint_success=False,error=dict(code='worker_exit',message=str(p.returncode)))
    if timeout:result.update(status='failed',joint_success=False,error=dict(code='sample_timeout',message='Total arm worker deadline exceeded'))
    if p.returncode and result['status']=='solid_validated':result.update(status='failed',joint_success=False,error=dict(code='worker_exit',message=str(p.returncode)))
    result['process']=dict(pid=p.pid,returncode=p.returncode,seconds=time.monotonic()-start,peak_rss_bytes=peak)
    save(record,result);(out/'active'/(sid+'_'+arm+'.json')).unlink(missing_ok=True)
    print('Completed',sid,arm,result['status'],result.get('error',{}).get('code',''),flush=True)
    return result


def report(out):
    policy=read(out/'campaign.json');rows=read(out/'manifest.json');comparison=Path(policy['comparison'])
    old={r['id']:read(comparison/'records'/(r['id']+'.json')) for r in rows}
    historical_graph={r['id']:read(Path(policy['input_group'])/'records'/(r['id']+'.json'))['score'] for r in rows}
    records={arm:{p.parent.name:read(p) for p in (out/'records').glob('*/'+arm+'.json')} for arm in ARMS}
    summary=dict(total=100,completed=sum(map(len,records.values())),groups={},historical=dict(face_good=17,graph_good=15,cad_step=12,joint_success=12))
    lines=['# 固定粗0.03网格：面区域修复五组对照','',
        '固定20个dev，各组20例，10个check不运行。精确CAD标量E/V，细目标0.003、单阶段阈值0.003；从bands起重算。',
        f"{policy['jobs']}个CPU worker，每例每组总超时10800秒，包含所有修复回退和评分。GT仅用于离线评分。",
        '历史11-09曲线回放复用了拓扑/顶点；本轮总耗时包含重建这些阶段，不能直接将其解释为曲线算法变慢。','',
        '| 组 | 完成/20 | 面正确 | 图正确 | STEP | 图+STEP | 成功回退数 | worker中位秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    for arm,rr in records.items():
        good=lambda r:r.get('status')=='solid_validated'
        item=dict(total=20,completed=len(rr),pending=20-len(rr),face_good=sum(bool((r.get('score') or {}).get('face_good')) for r in rr.values()),
            graph_good=sum(bool((r.get('score') or {}).get('graph_good')) for r in rr.values()),cad_step=sum(good(r) for r in rr.values()),
            joint_success=sum(bool(r.get('joint_success')) for r in rr.values()),
            successful_fallbacks=sum(good(r) and len(r.get('repair_attempts',[]))>1 for r in rr.values()),
            failure_counts=dict(Counter(r.get('error',{}).get('code','unknown') for r in rr.values() if not good(r))),
            median_worker_seconds=statistics.median(r['process']['seconds'] for r in rr.values()) if rr else None,
            historical_gained=[sid for sid,r in rr.items() if r.get('joint_success') and not (old[sid]['status']=='solid_validated' and historical_graph[sid]['graph_good'])],
            historical_lost=[sid for sid,r in rr.items() if old[sid]['status']=='solid_validated' and historical_graph[sid]['graph_good'] and not r.get('joint_success')])
        reference=records['baseline']
        item['baseline_gained']=[sid for sid,r in rr.items() if sid in reference and r.get('joint_success') and not reference[sid].get('joint_success')]
        item['baseline_lost']=[sid for sid,r in rr.items() if sid in reference and reference[sid].get('joint_success') and not r.get('joint_success')]
        summary['groups'][arm]=item
        lines.append(f"| {arm} | {len(rr)} | {item['face_good']} | {item['graph_good']} | {item['cad_step']} | {item['joint_success']} | {item['successful_fallbacks']} | {item['median_worker_seconds']} |")
        for row in rows:
            sid=row['id'];r=rr.get(sid)
            csvrows.append(dict(id=sid,arm=arm,status=r['status'] if r else 'pending',error=r.get('error',{}).get('code','') if r else '',
                face_good=(r.get('score') or {}).get('face_good') if r else None,graph_good=(r.get('score') or {}).get('graph_good') if r else None,
                joint_success=r.get('joint_success') if r else None,effective_mode=r.get('effective_mode') if r else None,
                attempts=len(r.get('repair_attempts',[])) if r else None,worker_seconds=r['process']['seconds'] if r else None,
                surface_error=max(r['fidelity']['recon_to_truth_max'],r['fidelity']['truth_to_recon_max']) if r and 'fidelity' in r else None))
    lines+=['',f"已完成{summary['completed']}/100；pending不计失败。历史面17/20、图15/20、STEP及图+STEP均12/20。",'',
        'baseline保留两项过滤；no_mean仅关闭均值过滤；small修复小核心；single_face修复单面边带；joint联合修复并尝试无独立E谷线的双面分隔带归并。',
        '回退成功计入最终结果，但不计作归并本身成功。逐候选动作、原始面归属和场证据、失败候选的独立图评分见runs及attempts；metrics.csv列有效分支。','',
        '## 相对历史配对','', '```json',json.dumps({a:dict(gained=g['historical_gained'],lost=g['historical_lost']) for a,g in summary['groups'].items()},ensure_ascii=False,indent=2),'```']
    save(out/'summary.json',summary);(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    return summary


def run(out):
    verify(out);started=time.time();save(out/'state.json',dict(status='running',started_unix=started,pid=os.getpid()))
    try:
        tasks=[(r['id'],arm) for r in read(out/'manifest.json') for arm in ARMS]
        with ThreadPoolExecutor(max_workers=read(out/'campaign.json')['jobs']) as pool:
            for future in as_completed([pool.submit(launch,out,task) for task in tasks]):
                future.result();summary=report(out);save(out/'state.json',dict(status='running',started_unix=started,summary=summary))
        verify(out);summary=report(out);save(out/'state.json',dict(status='completed',started_unix=started,ended_unix=time.time(),summary=summary))
    except BaseException as exc:
        save(out/'state.json',dict(status='controller_failed',started_unix=started,error=repr(exc)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-group',type=Path);p.add_argument('--comparison',type=Path)
    p.add_argument('--id');p.add_argument('--arm',choices=list(ARMS));p.add_argument('--jobs',type=int,default=2)
    modes=p.add_mutually_exclusive_group(required=True)
    for name in ('initialize','run','report','worker'):modes.add_argument('--'+name,action='store_true')
    args=p.parse_args();out=args.output.resolve()
    if args.initialize:
        if args.input_group is None or args.comparison is None:p.error('--initialize requires --input-group and --comparison')
        initialize(args.input_group.resolve(),args.comparison.resolve(),out,args.jobs)
    elif args.run:run(out)
    elif args.worker:
        if not args.id or not args.arm:p.error('--worker requires --id and --arm')
        worker(out,args.id,args.arm)
    else:report(out)


if __name__=='__main__':main()
