"""Paired, frozen native-mesh/L5 benchmark against the historical F30 dev cohort.

The reconstruction receives only a mesh and scalar E/V queries.
CAD identities and nearest-support diagnostics stay in this benchmark module.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
MODULE=REPO/'band_brep_reconstruction'
sys.path[:0]=[str(MODULE),str(REPO/'veudf/src'),str(Path(__file__).parent)]


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    from bandbrep.schemas import jsonable
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(jsonable(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def configuration(timeout=10800.,coarse_edge=.02,*,fine_edge=.005,
                  face_threshold1=.005,face_second_pass=True):
    from bandbrep.config import Config
    # Match the last L5 experiment's feature tolerance, batching and logging.
    # All remaining settings are the mesh/L5 implementation's defaults.
    return Config(feature_sdf_tolerance=.004,query_batch_size=65536,save_queries=False,
                  sample_timeout_seconds=timeout,refinement_coarse_edge=coarse_edge,
                  refinement_target_edge=fine_edge,face_threshold1=face_threshold1,
                  face_second_pass=face_second_pass)


def freeze(out,reference,native_reference,timeout,coarse_edge=.02,*,fine_edge=.005,
           face_threshold1=.005,face_second_pass=True):
    cfg=configuration(timeout,coarse_edge,fine_edge=fine_edge,
                      face_threshold1=face_threshold1,face_second_pass=face_second_pass).validate()
    if out.exists():raise RuntimeError('Use --run / --report on an existing experiment; never refreeze it')
    rows=read(reference/'manifest.json')
    inputs={str(reference/'manifest.json'):digest(reference/'manifest.json'),
            str(native_reference/'manifest.json'):digest(native_reference/'manifest.json')}
    if read(native_reference/'manifest.json')!=rows:raise RuntimeError('Native reference cohort differs')
    for r in rows:
        if digest(r['step'])!=r['source_sha256']:raise RuntimeError('Changed source STEP: '+r['id'])
        inputs[r['step']]=r['source_sha256']
        if r['cohort']=='dev':
            for group in ('M0','L5'):
                p=reference/'records'/r['id']/(group+'.json');inputs[str(p)]=digest(p)
            p=Path(r['reference_cache'])/'centers.npz';inputs[str(p)]=digest(p)
            for name in ('source_audit.json','condition/vertices.npy','condition/triangles.npy','condition/meta.json'):
                p=native_reference/'runs'/r['id']/name;inputs[str(p)]=digest(p)
            p=native_reference/'records'/(r['id']+'.json');inputs[str(p)]=digest(p)
    out.mkdir(parents=True)
    save(out/'manifest.json',rows)
    save(out/'policy.json',dict(date=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),reference=str(reference),cohort='dev',count=20,
        check='10 historical check models remain unrun',groups=['mesh_coarse_fine'],group='mesh_coarse_fine',jobs=2,
        native_reference=str(native_reference),sample_timeout_seconds=timeout,
        config=cfg.to_dict(),field_source='production exact CAD E/V; oracle presence',
        udf_gradient='central differences of scalar UDF samples; step=1e-5; no analytic derivative callback',
        refinement=f'geometry-only coarse {coarse_edge:g}; E-only near-field fine {fine_edge:g}; long-edge-only pruning',
        refinement_selection='edge_udf_only',
        mesh_source='production read_normalized/check_solid_faces/collect_targets/validate_geometry/audit_shape/triangulate/weld',
        sampling='geometry-only condition, no full training pools required',
        comparison='same source hashes and dev split; compare historical M0 and L5; not a learned-field experiment',
        geometry_change='MC/grid surface replaced by production native CAD triangulation; target semantics/normalization audited',
        budgets='One total wall-clock deadline from worker launch, including verification/source/condition/refinement/reconstruction/offline scoring; no round, triangle or RSS stopping budget.',
        caveat='Native CAD target canonicalization/filter changes, if any, are reported; rejected inputs remain in denominator.'))
    code={}
    paths=list((MODULE/'bandbrep').rglob('*.py'))+list((REPO/'veudf/src/veudf').rglob('*.py'))+list((MODULE/'benchmarks').glob('*.py'))
    for path in sorted(paths):
        relative=path.relative_to(REPO);dst=out/'source_snapshot'/relative
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst);code[str(relative)]=digest(dst)
    from veudf.preprocess.pipeline import runtime_identity
    save(out/'environment.json',dict(python=sys.version,executable=sys.executable,packages=runtime_identity(),
                                    threads='OMP/OPENBLAS/MKL/NUMEXPR=1',gpu=False))
    inputs[str(out/'manifest.json')]=digest(out/'manifest.json')
    inputs[str(out/'policy.json')]=digest(out/'policy.json')
    save(out/'identity.json',dict(code=code,inputs=inputs,created_unix=time.time()))
    print('Frozen',out,flush=True)


def verify(out):
    identity=read(out/'identity.json')
    for relative,h in identity['code'].items():
        if digest(out/'source_snapshot'/relative)!=h:raise RuntimeError('Frozen code changed: '+relative)
    for path,h in identity['inputs'].items():
        if digest(path)!=h:raise RuntimeError('Frozen input changed: '+path)


class ExactUDF:
    """Production CAD distance values; reconstruction samples its own gradients."""
    def __init__(self,targets,cfg):
        import numpy as np
        from veudf.preprocess.distance import EdgeOracle,VertexOracle
        self.np=np;self.targets=targets;self.edge=EdgeOracle(targets.edges,cfg);self.vertex=VertexOracle(targets.vertices,cfg)
        self.last=None;self.results=None
        self.counts=dict(evaluation_rows=0,edge_cache_rows=0,actual_edge_rows=0)
    def query(self,q):
        np=self.np
        if self.last is not None and np.array_equal(q,self.last):
            self.counts['edge_cache_rows']+=len(q);return self.results
        self.counts['actual_edge_rows']+=len(q)
        self.results=self.edge.query(q),self.vertex.query(q);self.last=q.copy()
        return self.results
    def values(self,q):
        self.counts['evaluation_rows']+=len(q);e,v=self.query(q)
        return self.np.column_stack((e['distance'],v['distance']))
    def report(self):return dict(**self.counts,cad_distance_solver_calls=self.edge.calls)


def prepare_source(row,folder):
    import numpy as np
    from veudf.preprocess.config import Config
    from veudf.preprocess.cad import read_normalized,collect_targets,triangulate
    from veudf.preprocess.filtering import check_solid_faces,audit_shape
    from veudf.preprocess.pipeline import validate_geometry
    from veudf.preprocess.condition import weld
    from veudf.io.condition import write_condition
    from bandbrep.surface_mesh import load_mesh
    cfg=Config().validate()
    solid,normalization=read_normalized(row['step'],cfg)
    check_solid_faces(solid,cfg);targets=collect_targets(solid,cfg);validate_geometry(targets,cfg)
    filters=audit_shape(solid,targets,cfg)
    raw=triangulate(targets,cfg);mesh,oracle=weld(raw,cfg)
    write_condition(folder/'condition',mesh,count=cfg.geometry_surface_pool_count,build_seed=cfg.seed,normalization=normalization)
    np.save(folder/'triangle_face_ids.npy',raw[2],allow_pickle=False)
    save(folder/'truth_topology.json',targets.topology)
    old_counts=row['historical_graph']['truth'];counts=dict(faces=len(targets.faces),edges=len(targets.edges),vertices=len(targets.vertices))
    udf=ExactUDF(targets,cfg)
    # Historical normalization is stored in the last L5 record.
    previous=read(Path(folder.parents[1])/'policy.json')
    record=read(Path(previous['reference'])/'records'/row['id']/'L5.json')
    oldnorm=record.get('source_audit',{}).get('normalization')
    if oldnorm is None:
        oldnorm=read(Path(previous['reference'])/'records'/row['id']/'M0.json').get('source_audit',{}).get('normalization')
    delta=float(np.max(np.abs(np.array(normalization['world_to_normalized'])-np.array(oldnorm['world_to_normalized'])))) if oldnorm else None
    if delta is not None and delta>1e-10:raise RuntimeError('Normalization differs from historical cohort')
    with np.load(Path(row['reference_cache'])/'centers.npz',allow_pickle=False) as a:
        ids=np.linspace(0,len(a['points'])-1,min(32,len(a['points']))).astype(int)
        q=a['points'][ids];expected=a['values'][ids,1:]
    error=np.abs(udf.values(q)-expected).max(0)
    audit=dict(target_counts=counts,historical_target_counts=old_counts,target_counts_match=counts==old_counts,
        normalization=normalization,normalization_matrix_max_delta=delta,old_center_ev_max_discrepancy=error.tolist(),
        old_center_ev_agree=bool(np.all(error[np.array([bool(targets.edges),bool(len(targets.vertices))])]<=1e-5)),
        empty_channel_policy='Absent E/V excluded from distance equivalence; legacy clip .02 versus production EMPTY_DISTANCE',filters=filters,
        mesh=dict(vertices=len(mesh[0]),triangles=len(mesh[1]),sha256=oracle.fingerprint),
        topology_changed=counts!=old_counts,scope='32 finite cross-checks, not a global target-equivalence proof')
    native=read(Path(previous['native_reference'])/'runs'/row['id']/'source_audit.json')
    if audit['mesh']['sha256']!=native['mesh']['sha256']:raise RuntimeError('Native input mesh differs from four-round reference')
    audit['native_reference_mesh_matches']=True
    save(folder/'source_audit.json',audit)
    return load_mesh(folder/'condition'),udf,targets,raw,oracle,audit


def offline_source(targets,raw,oracle,normalization):
    from types import SimpleNamespace
    from scipy.spatial import cKDTree
    from offline_support import PolylineDistance
    edge=PolylineDistance(targets.edges) if targets.edges else None
    return SimpleNamespace(targets=targets,mesh=raw,normalization=normalization,edge=edge,
        vtree=cKDTree(targets.vertices) if len(targets.vertices) else None,
        truth_face=lambda q:raw[2][oracle.closest(q)[1]],sdf=oracle.query)


REFINEMENT_STAGES = ('coarse_refinement', 'fine_refinement')


def refinement_progress(folder):
    return {stage+'_progress':read(path) for stage in REFINEMENT_STAGES
            if (path:=folder/'reconstruction'/(stage+'_progress.json')).exists()}


def collect_refinement(result, folder):
    result.update(refinement_progress(folder))
    run=result.get('run',{})
    for stage in run.get('stages',[]):
        if stage['stage'] in REFINEMENT_STAGES:result[stage['stage']]=stage['report']


def worker(out,model_id):
    from bandbrep.deadline import sample_budget
    with sample_budget(read(out/'policy.json')['sample_timeout_seconds']):
        return worker_sample(out,model_id)


def worker_sample(out,model_id):
    import numpy as np
    from bandbrep.deadline import checkpoint
    from bandbrep.config import Config
    from bandbrep.field_provider import CallableUDFProvider
    from bandbrep.reconstruct import reconstruct
    from bandbrep.cache import load_topology
    from bandbrep.schemas import BandState
    from experiment_scoring import score,geometry_score,fidelity_score
    verify(out)
    row=next(r for r in read(out/'manifest.json') if r['id']==model_id)
    folder=out/'runs'/model_id;folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();result=dict(id=model_id,cohort=row['cohort'],group='mesh_coarse_fine',status='running',joint_success=False)
    udf=None;reconstruction=None;stage='source_preparation'
    save(folder/'state.json',dict(stage=stage,unix=time.time(),pid=os.getpid()))
    try:
        mesh,udf,targets,raw,oracle,audit=prepare_source(row,folder)
        result['source_audit']=audit
        provider=CallableUDFProvider(udf.values,fingerprint='production-cad:'+row['source_sha256'],
            presence=('present' if targets.edges else 'absent','present' if len(targets.vertices) else 'absent'))
        stage='reconstruction';save(folder/'state.json',dict(stage=stage,unix=time.time(),pid=os.getpid()))
        try:reconstruction=reconstruct(mesh,provider,Config(**read(out/'policy.json')['config']),folder/'reconstruction')
        except Exception as exc:
            result.update(status='failed',error=dict(code=getattr(exc,'code',type(exc).__name__),message=str(exc),details=getattr(exc,'details',{})))
        else:result['status']='solid_validated'
        runpath=folder/'reconstruction/run.json'
        if runpath.exists():
            run=read(runpath);result['run']=run
            result['failed_stage']=run.get('failed_stage')
            stages={s['stage']:s for s in run.get('stages',[])}
            if 'fine_refinement' in stages:
                result['preparation_seconds']=time.monotonic()-start-run['elapsed_seconds']+stages['fine_refinement']['elapsed_seconds']
            collect_refinement(result,folder)
        if result.get('error',{}).get('code')!='sample_timeout' and (folder/'reconstruction/incidence.json').exists():
            stage='offline_scoring';save(folder/'state.json',dict(stage=stage,unix=time.time(),pid=os.getpid()))
            checkpoint(stage)
            source=offline_source(targets,raw,oracle,audit['normalization'])
            checkpoint(stage)
            topo=load_topology(folder/'reconstruction')
            with np.load(folder/'reconstruction/bands.npz') as a:
                bands=BandState(*(a[k] for k in ('points','values','edge_band','vertex_band','edge_region','core_mask')),
                    face_labels=a['face_labels'],exclusion_reasons=a['exclusion_reasons'],
                    vertex_labels=a['vertex_labels'] if 'vertex_labels' in a else None,
                    vertex_centers=a['vertex_centers'] if 'vertex_centers' in a else None)
            result['score']=score(topo,bands,source);checkpoint(stage);save(folder/'graph_score.json',result['score'])
            if reconstruction is not None:
                result['geometry']=geometry_score(reconstruction['graph'].vertices,reconstruction['curves'],source)
                result['fidelity']=fidelity_score(reconstruction['solid'],source)
                checkpoint(stage)
            result['joint_success']=result['status']=='solid_validated' and result['score']['graph_good']
    except Exception as exc:
        result.update(status='failed',joint_success=False,failed_stage=stage,
            error=dict(code=getattr(exc,'code',type(exc).__name__),message=str(exc),details=getattr(exc,'details',{})),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-start
    if 'preparation_seconds' not in result:result['preparation_seconds']=result['seconds']
    result['reconstruction_seconds']=result['seconds']-result['preparation_seconds']
    if udf is not None:result['queries']=udf.report()
    collect_refinement(result,folder)
    save(folder/'outcome.json',result)
    print(model_id,result['status'],result.get('error',{}).get('code',''),flush=True)


def launch(out,row):
    import psutil
    model_id=row['id'];record=out/'records'/(model_id+'.json')
    if record.exists():return read(record)
    folder=out/'runs'/model_id
    if folder.exists():raise RuntimeError('Unrecorded partial run; inspect before resuming: '+str(folder))
    env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    frozen=out/'source_snapshot'
    env['PYTHONPATH']=str(frozen/'veudf/src')+':'+str(frozen/'band_brep_reconstruction')
    script=frozen/'band_brep_reconstruction/benchmarks/run_mesh_input_experiment.py'
    logs=out/'logs';logs.mkdir(exist_ok=True)
    runpath=folder/'reconstruction/run.json'
    start=time.monotonic();prep_done=None;peak=0;reason=None;last_stage='source_preparation'
    with (logs/(model_id+'.log')).open('w') as log:
        p=subprocess.Popen([sys.executable,str(script),'--worker','--output',str(out),'--id',model_id],
                           env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        policy=read(out/'policy.json')
        while p.poll() is None:
            try:
                process=psutil.Process(p.pid);rss=process.memory_info().rss
                for child in process.children(recursive=True):
                    try:rss+=child.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss)
            except psutil.NoSuchProcess:pass
            runpath=folder/'reconstruction/run.json'
            if runpath.exists():
                try:
                    run=read(runpath)
                    if any(s['stage']=='fine_refinement' for s in run.get('stages',[])) and prep_done is None:prep_done=time.monotonic()
                    last_stage=run.get('failed_stage') or (run.get('stages') or [{}])[-1].get('stage','mesh')
                except (ValueError,OSError):pass
            try:progress_data=refinement_progress(folder)
            except (ValueError,OSError):progress_data={}
            if prep_done is None:
                for stage in REFINEMENT_STAGES:
                    data=progress_data.get(stage+'_progress')
                    if data:last_stage=stage+':'+data['event']
            clock=time.monotonic()
            if clock-start>policy['sample_timeout_seconds']:reason='sample_timeout'
            if reason:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            save(out/'active'/(model_id+'.json'),dict(id=model_id,pid=p.pid,elapsed_seconds=clock-start,
                last_completed_stage=last_stage,preparation_completed=prep_done is not None,peak_rss_bytes=peak,refinement_progress=progress_data))
            time.sleep(1)
    path=folder/'outcome.json'
    result=read(path) if path.exists() else dict(id=model_id,cohort='dev',group='mesh_coarse_fine',status='failed',joint_success=False)
    if reason or p.returncode!=0:
        result.update(status='failed',joint_success=False,error={'code':reason or 'worker_exit','message':str(p.returncode),
            'details':{'timeout_seconds':policy['sample_timeout_seconds']}},failed_stage=last_stage)
    if runpath.exists() and 'run' not in result:
        try:result['run']=read(runpath)
        except ValueError:pass
    collect_refinement(result,folder)
    audit=folder/'source_audit.json'
    if audit.exists() and 'source_audit' not in result:result['source_audit']=read(audit)
    score_path=folder/'graph_score.json'
    if score_path.exists() and 'score' not in result:result['score']=read(score_path)
    result['process']=dict(pid=p.pid,returncode=p.returncode,seconds=time.monotonic()-start,peak_rss_bytes=peak,
                           preparation_completed=prep_done is not None)
    save(record,result)
    (out/'active'/(model_id+'.json')).unlink(missing_ok=True)
    print(model_id,result['status'],result.get('error',{}).get('code',''),flush=True)
    return result


def report(out):
    rows=[r for r in read(out/'manifest.json') if r['cohort']=='dev']
    policy=read(out/'policy.json');reference=Path(policy['reference'])
    primary=policy['group']
    near_field='E近场' if policy.get('refinement_selection')=='edge_udf_only' else 'E/V近场'
    policy_note=f"先纯几何粗加密至{policy['config']['refinement_coarse_edge']:g}，再对{near_field}细化至{policy['config']['refinement_target_edge']:g}；仅切超长边，仅设每例总超时。"
    config=policy['config']
    policy_note+=f" 面分割第一阈值{config['face_threshold1']:g}，第二阶段{'启用' if config.get('face_second_pass',True) else '关闭'}。"
    records={p.stem:read(p) for p in sorted((out/'records').glob('*.json'))}
    summary=dict(total=len(rows),completed=len(records),pending=len(rows)-len(records),groups={},paired={})
    all_groups={primary:list(records.values())}
    for g in ('M0','L5'):all_groups['old_'+g]=[read(reference/'records'/r['id']/(g+'.json')) for r in rows]
    if policy.get('native_reference'):
        all_groups['native_four_rounds']=[read(Path(policy['native_reference'])/'records'/(r['id']+'.json')) for r in rows]
    for group,rr in all_groups.items():
        summary['groups'][group]=dict(total=len(rows),completed=len(rr),
            prepared=sum(bool(r.get('fine_refinement',{}).get('target_reached')) if group==primary else bool(r.get('refinement')) if group=='native_four_rounds' else r.get('status')!='preparation_failed' and bool(r.get('process')) for r in rr),
            face_good=sum(bool(r.get('score',{}).get('face_good')) for r in rr),
            graph_good=sum(bool(r.get('score',{}).get('graph_good')) for r in rr),
            cad_step=sum(r.get('run',{}).get('status',r.get('status'))=='solid_validated' for r in rr),
            joint_success=sum(bool(r.get('joint_success')) for r in rr))
    for group in ('M0','L5'):
        old={r['id']:read(reference/'records'/r['id']/(group+'.json')) for r in rows}
        summary['paired'][group]=dict(
            new_success=[sid for sid,r in records.items() if r.get('joint_success') and not old[sid].get('joint_success')],
            lost_success=[sid for sid,r in records.items() if old[sid].get('joint_success') and not r.get('joint_success')],
            graph_lost=[sid for sid,r in records.items() if old[sid].get('score',{}).get('graph_good') and not r.get('score',{}).get('graph_good')],
            common_success=[sid for sid,r in records.items() if old[sid].get('joint_success') and r.get('joint_success')])
    if policy.get('native_reference'):
        old={r['id']:read(Path(policy['native_reference'])/'records'/(r['id']+'.json')) for r in rows}
        summary['paired']['native_four_rounds']={
            'new_success':[sid for sid,r in records.items() if r.get('joint_success') and not old[sid].get('joint_success')],
            'lost_success':[sid for sid,r in records.items() if old[sid].get('joint_success') and not r.get('joint_success')]}
    summary['failure_counts']=dict(Counter(r.get('error',{}).get('code',r['status']) for r in records.values() if r['status']=='failed'))
    save(out/'summary.json',summary)
    columns=['id','old_M0','old_L5','new_status','new_code','face_good','graph_good','joint_success','input_triangles','coarse_triangles','refined_triangles','coarse_rounds','fine_rounds','coarse_seconds','fine_seconds','last_observed_triangles','target_reached','seconds','peak_gib']
    csvrows=[]
    lines=['# 原生网格粗加密 + 近场细加密：同源 F30 dev 配对实验','',f"已完成 {len(records)}/20；10 个 check 不运行。精确 CAD E/V 与 oracle presence，无模型权重。",'',
           '| 流程 | 固定分母 | 已完成 | 面正确 | 图正确 | CAD STEP | 图+STEP |','|---|---:|---:|---:|---:|---:|---:|']
    for g,x in summary['groups'].items():lines.append(f"| {g} | {x['total']} | {x['completed']} | {x['face_good']} | {x['graph_good']} | {x['cad_step']} | {x['joint_success']} |")
    lines+=['','源 STEP 与 dev 划分一致。'+policy_note+' 特征 SDF 容差仍为0.004；旧 MC 数字来自历史实验，未重跑。',
            '','每例总计时包含输入核验、源准备、condition、加密、拓扑、几何、STEP 和离线评分；超时与数值错误保留在分母。RSS/三角形数量仅观测，不作为实验硬上限。旧结果不覆盖。',
            '','| ID | 旧 M0 图+STEP | 旧 L5 图+STEP | 新状态 | 首个错误 | 新图+STEP |','|---|---|---|---|---|---|']
    for row in rows:
        sid=row['id'];r=records.get(sid,{})
        a=read(reference/'records'/sid/'M0.json');b=read(reference/'records'/sid/'L5.json')
        code=r.get('error',{}).get('code','');refine=r.get('fine_refinement',{});coarse=r.get('coarse_refinement',{})
        progress=r.get('fine_refinement_progress') or r.get('coarse_refinement_progress',{})
        lines.append(f"| {sid} | {bool(a.get('joint_success'))} | {bool(b.get('joint_success'))} | {r.get('status','pending')} | {code} | {bool(r.get('joint_success')) if r else '—'} |")
        csvrows.append(dict(zip(columns,[sid,bool(a.get('joint_success')),bool(b.get('joint_success')),r.get('status','pending'),code,
            r.get('score',{}).get('face_good'),r.get('score',{}).get('graph_good'),r.get('joint_success'),
            r.get('source_audit',{}).get('mesh',{}).get('triangles'),coarse.get('refined',{}).get('triangles'),refine.get('refined',{}).get('triangles'),
            len(coarse.get('history',r.get('coarse_refinement_progress',{}).get('history',[]))),
            len(refine.get('history',r.get('fine_refinement_progress',{}).get('history',[]))),
            coarse.get('elapsed_seconds'),refine.get('elapsed_seconds'),progress.get('triangles'),bool(refine.get('target_reached')),
            r.get('process',{}).get('seconds'),r.get('process',{}).get('peak_rss_bytes',0)/1024**3])))
    lines+=['','## 配对变化','',json.dumps(summary['paired'],ensure_ascii=False,indent=2),'','## 首个阻断','',json.dumps(summary['failure_counts'],ensure_ascii=False,indent=2)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(csvrows)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=MODULE/'outputs/mesh_density_20261028_v2')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--initialize',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--report',action='store_true');mode.add_argument('--worker',action='store_true')
    p.add_argument('--id')
    p.add_argument('--native-reference',type=Path,default=MODULE/'outputs/native_mesh_L5_20261101_v1')
    p.add_argument('--coarse-edge',type=float,default=.02,help='global coarse-refinement target, frozen at initialization')
    p.add_argument('--sample-timeout',type=float,default=10800.,help='total wall-clock seconds per sample, including all preparation and scoring')
    p.add_argument('--fine-edge',type=float,default=.005)
    p.add_argument('--face-threshold1',type=float,default=.005)
    p.add_argument('--face-second-pass',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();out=a.output.resolve()
    if a.initialize:freeze(out,a.reference.resolve(),a.native_reference.resolve(),a.sample_timeout,a.coarse_edge,
                          fine_edge=a.fine_edge,face_threshold1=a.face_threshold1,face_second_pass=a.face_second_pass)
    elif a.worker:worker(out,a.id)
    elif a.report:print(json.dumps(report(out),ensure_ascii=False,indent=2))
    else:
        verify(out);rows=[r for r in read(out/'manifest.json') if r['cohort']=='dev']
        started=time.time()
        save(out/'batch_started.json',dict(started_unix=started,controller_pid=os.getpid()))
        save(out/'batch.json',dict(status='running',started_unix=started,pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(launch,out,r) for r in rows]
            for future in as_completed(jobs):future.result();report(out)
        summary=report(out);save(out/'batch.json',dict(status='completed',started_unix=started,ended_unix=time.time(),summary=summary))
        from analyze_mesh_input_experiment import analyze
        analyze(out)
        print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
