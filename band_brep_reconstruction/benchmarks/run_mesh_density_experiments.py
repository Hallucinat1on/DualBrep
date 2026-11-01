"""Frozen F30 mesh-density experiments: same surface, freshly queried exact E/V."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace, asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(HERE)]
PRIMARY=('M0','U1','L5','L25')
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'PYTHONPATH':str(ROOT)+':'+str(HERE)+':/tmp/evudf-test-deps'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,data):
    from bandbrep.schemas import jsonable
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(jsonable(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def code_hashes():
    paths=sorted(list((ROOT/'bandbrep').rglob('*.py'))+list(HERE.glob('*.py')))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def freeze(a):
    out=Path(a.output).resolve();ref=Path(a.reference).resolve()
    if (out/'identity.json').exists():raise RuntimeError('Use --run/--status for existing experiment')
    rows=read(ref/'manifest.json');inputs={str(ref/'manifest.json'):digest(ref/'manifest.json')}
    migrations=[]
    for r in rows:
        step=Path(a.step_root).resolve()/r['id']/Path(r['step']).name
        if digest(step)!=r['source_sha256']:raise RuntimeError('STEP differs from old F30: '+r['id'])
        r['old_step_path']=r['step'];r['step']=str(step)
        inputs[str(step)]=r['source_sha256']
        cache=ref/'cache'/r['id']/'B';prep=read(cache/'prepared.json')
        for name,h in prep['artifact_hashes'].items():
            p=cache/name
            if digest(p)!=h:raise RuntimeError('Reference cache differs: '+str(p))
            inputs[str(p)]=h
        inputs[str(cache/'prepared.json')]=digest(cache/'prepared.json')
        r['reference_cache']=str(cache)
        r['historical_graph']=read(ref/'runs'/r['id']/'C004'/'graph_score.json')
        migrations.append(dict(id=r['id'],step_sha256=r['source_sha256'],cache=str(cache),
                               raw_training_assets_present=Path(r['raw']).exists(),
                               reused='verified B grid/MC mesh/centers only; CAD target set rebuilt and checked'))
    out.mkdir(parents=True,exist_ok=True)
    save(out/'manifest.json',rows)
    policy=dict(date='2026-10-28',reference=str(ref),step_root=str(Path(a.step_root).resolve()),
        groups=PRIMARY,dev=20,check=10,jobs=a.jobs,prepare_timeout=a.prepare_timeout,
        reconstruction_timeout=a.timeout,memory_limit_gib=8,max_triangles=1500000,
        max_rounds=4,rho_edge=.02,rho_vertex=.03,condition_resolution=256,feature_sdf_tolerance=.004,
        selection=dict(min_dev_net_rescues=2,no_step_regression=True,no_graph_regression=True,
                       no_new_geometry_alarm=True,common_distance_max_increase=.0002,
                       tie_break='fewer queried points, then less time, then group name'),
        auxiliary=['U1-count40','Area-M0','Area-diagnostic-best'],
        notes=['User authorized prepare 7200s / reconstruct 3600s instead of original 1800/600.',
               'Old raw/prepared training directories missing; immutable MC grid caches remain and are verified.',
               'F30 dev/check are previously exposed internal comparisons, not new final test.'])
    save(out/'policy.json',policy);save(out/'input_migration.json',migrations)
    for p in (out/'manifest.json',out/'policy.json'):inputs[str(p)]=digest(p)
    identity=dict(code=code_hashes(),inputs=inputs)
    save(out/'identity.json',identity)
    for relative in identity['code']:
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    import importlib.metadata
    save(out/'environment.json',dict(python=sys.version,executable=sys.executable,
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','trimesh')},
        threads={k:ENV[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    print('Frozen same F30 cohort at '+str(out),flush=True)


def verify_identity(out,inputs=True):
    frozen=read(out/'identity.json')
    if frozen['code']!=code_hashes():raise RuntimeError('Experiment code changed; new output identity required')
    if inputs:
        for p,h in frozen['inputs'].items():
            if digest(p)!=h:raise RuntimeError('Experiment input changed: '+p)


def geometry_group(group):
    if group=='U1-count40':return 'U1'
    if group.startswith('Area-'):return group[5:]
    return group


def settings(group):
    from bandbrep.mesh_densification import RefinementConfig
    g=geometry_group(group)
    return RefinementConfig(method='none' if g=='M0' else 'uniform' if g=='U1' else 'local',
                            target_edge=.0025 if g=='L25' else .005)


def source_and_field(row,progress):
    import numpy as np
    from experiment_fields import BenchmarkCAD
    from alignment_fields import AlignmentField
    class TrackedAlignmentField(AlignmentField):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.seen_positions=set();self.position_requests=0
        def track(self,xyz):
            p=np.asarray(xyz,np.float64).reshape(-1,3).copy();p[p==0]=0
            self.position_requests+=len(p)
            packed=np.ascontiguousarray(p).view(np.dtype((np.void,24))).ravel()
            self.seen_positions.update(v.tobytes() for v in np.unique(packed))
        def evaluate(self,xyz):
            self.track(xyz);return super().evaluate(xyz)
        def gradient(self,xyz,channel,step=1e-5):
            if channel in (1,2):self.track(xyz)
            return super().gradient(xyz,channel,step)
        def report(self):
            data=super().report();data['ev_position_requests']=self.position_requests
            data['ev_unique_positions']=len(self.seen_positions)
            data['ev_repeated_positions']=self.position_requests-len(self.seen_positions)
            data['uniqueness_scope']='evaluate and E/V gradient positions; exact float64 coordinates with signed-zero normalized; includes failed requests'
            return data
    source=BenchmarkCAD(row['step'])
    counts=dict(faces=len(source.targets.faces),edges=len(source.targets.edges),vertices=len(source.targets.vertices))
    if counts!=row['historical_graph']['truth']:raise RuntimeError('Regenerated CAD target counts differ from old F30')
    cache=Path(row['reference_cache'])
    grid=np.load(cache/'grid.npy',mmap_mode='r')
    provider=TrackedAlignmentField(source,grid,'B',row['source_sha256']+':density',progress=progress)
    with np.load(cache/'centers.npz') as f:
        ids=np.linspace(0,len(f['points'])-1,min(32,len(f['points']))).astype(int)
        q=f['points'][ids];expected=f['values'][ids]
    actual=provider.evaluate(q).values
    error=np.abs(actual-expected).max(axis=0)
    if not np.all(error<=np.array([2e-6,1e-5,2e-6])):
        raise RuntimeError('New CAD/normalization or grid disagrees with frozen center fields: '+str(error))
    return source,provider,dict(target_counts=counts,center_checks=len(q),max_field_discrepancy=error.tolist(),
                                normalization=source.normalization,
                                truth_mesh='regenerated CAD triangulation; old raw GT mesh unavailable')


def cfg_for(group):
    from bandbrep.config import load_config
    return load_config(preset='recommended',overrides=dict(resolution=256,max_resolution=256,
        query_batch_size=65536,feature_sdf_tolerance=.004,save_queries=False,
        face_filter_size=40 if group=='U1-count40' else 10))


def area_segmenter(mesh,original):
    """Explicit experiment-only paired area filtering, not a production setting."""
    import numpy as np
    from bandbrep.mesh_densification import areas
    from bandbrep.vendor.dualbrep_segmentation import hierarchical_segmentation
    area=areas(mesh);minimum=10*float(np.median(areas(original)))
    text=inspect.getsource(hierarchical_segmentation)
    text=text.replace('cluster_sizes = np.bincount(comp_labels, minlength=n_comp)',
                      'cluster_sizes = np.bincount(comp_labels, weights=area[active_indices], minlength=n_comp)')
    text=text.replace('weights=udf[active_indices]', 'weights=udf[active_indices]*area[active_indices]')
    text=text.replace('sub_sizes = np.bincount(sub_labels, minlength=n_sub)',
                      'sub_sizes = np.bincount(sub_labels, weights=area[eligible_indices], minlength=n_sub)')
    env=dict(hierarchical_segmentation.__globals__);env['area']=area
    exec(compile(text,'<density-area-filter>','exec'),env)
    function=env['hierarchical_segmentation']
    def wrapped(*args,**kwargs):
        kwargs['filter_size']=minimum
        count,labels,reasons,report=function(*args,**kwargs)
        report.update(filter_units='surface_area',minimum_area=minimum,mean_weighting='triangle_area')
        return count,labels,reasons,report
    return wrapped


def worker(a):
    import numpy as np
    from bandbrep.schemas import SurfaceMesh, require
    from bandbrep.mesh_densification import densify_surface_mesh, areas
    from bandbrep.surface_mesh import evaluate_batched, orient_and_validate
    from experiment_fields import ScalarOnly
    from run_reconstruction_experiments import graph_case, save_graph, full
    out=Path(a.output);verify_identity(out,False)
    row=next(r for r in read(out/'manifest.json') if r['id']==a.model_id)
    target=out/('cache' if a.phase=='prepare' else 'runs')/row['id']/(geometry_group(a.group) if a.phase=='prepare' else a.group)
    target.mkdir(parents=True,exist_ok=True)
    def progress(stage):save(target/'progress.json',dict(stage=stage,unix=time.time(),pid=os.getpid()))
    start=time.monotonic();result=dict(id=row['id'],group=a.group,phase=a.phase,status='running')
    provider=None
    current_stage='source_audit'
    try:
        progress('source_audit');source,provider,audit=source_and_field(row,progress)
        result['source_audit']=audit;cfg=cfg_for(a.group)
        with np.load(Path(row['reference_cache'])/'mesh.npz') as f:
            original=SurfaceMesh(f['vertices'],f['triangles'].astype(np.int64))
        path=out/'cache'/row['id']/geometry_group(a.group)
        if a.phase=='prepare':
            mesh,parent,report=densify_surface_mesh(original,provider,settings(a.group),progress)
            progress('surface_validation');orient_and_validate(mesh,ScalarOnly(provider),cfg)
            points=mesh.vertices[mesh.triangles].mean(axis=1)
            progress('final_exact_center_fields')
            values=evaluate_batched(provider,points,8192)
            np.savez_compressed(path/'refined_mesh.npz',vertices=mesh.vertices,triangles=mesh.triangles)
            # A center barycentric map and checked parent IDs certify the same source surface.
            base=original.vertices[original.triangles[parent]]
            u,w=base[:,1]-base[:,0],base[:,2]-base[:,0];q=points-base[:,0]
            normal=np.cross(u,w);den=(normal*normal).sum(1)
            b=(np.cross(q,w)*normal).sum(1)/den;c=(np.cross(u,q)*normal).sum(1)/den
            bary=np.column_stack((1-b-c,b,c))
            require(bary.min()>=-1e-6 and np.isfinite(bary).all(),'refinement_parent_map','Invalid original-face coordinates')
            deviation=np.linalg.norm(points-np.einsum('ni,nij->nj',bary,base),axis=1)
            require(deviation.max()<=1e-8,'refinement_parent_map','Child center left original triangle')
            np.savez_compressed(path/'parent_map.npz',parent_triangle=parent,center_barycentric=bary)
            np.savez_compressed(path/'centers.npz',points=points,values=values)
            save(path/'refinement.json',report)
            result.update(status='prepared',refinement=report,
                parent_max_error=float(deviation.max()),surface=mesh.report,
                artifacts={n:digest(path/n) for n in ('refined_mesh.npz','parent_map.npz','centers.npz','refinement.json')})
        else:
            prepared=read(path/'preparation.json')
            require(prepared['status']=='prepared','preparation_failed','Mesh/field preparation failed')
            for n,h in prepared['artifacts'].items():
                require(digest(path/n)==h,'cache_corrupt','New mesh cache changed',file=n)
            with np.load(path/'refined_mesh.npz') as f:mesh=SurfaceMesh(f['vertices'],f['triangles'],prepared['surface'])
            with np.load(path/'centers.npz') as f:points,values=f['points'],f['values']
            require(np.array_equal(points,mesh.vertices[mesh.triangles].mean(axis=1)),
                    'center_cache_mismatch','Final centers changed')
            provider.center_points,provider.center_values=points,values
            if a.group.startswith('Area-'):
                import bandbrep.bands
                bandbrep.bands.hierarchical_segmentation=area_segmenter(mesh,original)
            progress('topology')
            bands,topo,scoring=graph_case(source,mesh,points,values,provider,cfg)
            save_graph(target,bands,topo);save(target/'graph_score.json',scoring)
            save(target/'band_report.json',bands.report)
            result['score']=scoring;save(target/'outcome.json',result)
            progress('reconstruction')
            reconstruction=full(source,mesh,bands,topo,cfg,target,field=provider)
            result.update(status=reconstruction['status'],full=reconstruction,
                          joint_success=reconstruction['status']=='solid_validated' and scoring['graph_good'])
        result['queries']=provider.report()
    except Exception as exc:
        result.update(status='failed',error=dict(code=getattr(exc,'code',type(exc).__name__),message=str(exc),
                                                details=getattr(exc,'details',{})),traceback=traceback.format_exc())
        if (target/'progress.json').exists():result['last_stage']=read(target/'progress.json')['stage']
        if provider is not None:result['queries']=provider.report()
    result['seconds']=time.monotonic()-start
    save(target/('preparation.json' if a.phase=='prepare' else 'outcome.json'),result)
    print(a.model_id,a.group,a.phase,result['status'],result.get('error',{}).get('code',''),flush=True)
    return 0


def launch(out,model_id,group,phase):
    target=out/('cache' if phase=='prepare' else 'runs')/model_id/(geometry_group(group) if phase=='prepare' else group)
    target.mkdir(parents=True,exist_ok=True)
    if (target/'process.json').exists():
        prior=read(target/'process.json')
        if prior['status'] in ('completed','timeout','memory_limit','process_failed'):return prior
    policy=read(out/'policy.json');limit=policy['prepare_timeout'] if phase=='prepare' else policy['reconstruction_timeout']
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--phase',phase,'--model-id',model_id,
             '--group',group,'--output',str(out)]
    start=time.monotonic();peak=0;status='running'
    with (target/'worker.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=ENV,start_new_session=True)
        save(target/'process.json',dict(status='running',pid=child.pid,started_unix=time.time(),phase=phase))
        while child.poll() is None:
            pending=[child.pid];seen=set();rss=0
            while pending:
                pid=pending.pop()
                if pid in seen:continue
                seen.add(pid)
                try:
                    rss+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                    for task in Path(f'/proc/{pid}/task').iterdir():
                        pending.extend(map(int,(task/'children').read_text().split()))
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
            peak=max(peak,rss)
            if time.monotonic()-start>limit:status='timeout'
            if peak>policy['memory_limit_gib']*1024**3:status='memory_limit'
            if status!='running':
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
            time.sleep(1)
        child.wait()
        if status=='running':status='completed' if child.returncode==0 else 'process_failed'
    result=dict(status=status,pid=child.pid,returncode=child.returncode,phase=phase,
                seconds=time.monotonic()-start,peak_process_tree_bytes=peak,timeout_seconds=limit)
    save(target/'process.json',result)
    return result


def task(out,row,group):
    prep=launch(out,row['id'],group,'prepare')
    cache=out/'cache'/row['id']/geometry_group(group)
    data=read(cache/'preparation.json') if (cache/'preparation.json').exists() else {}
    if prep['status']!='completed' or data.get('status')!='prepared':
        result=dict(id=row['id'],group=group,cohort=row['cohort'],status='preparation_failed',
            error=data.get('error',dict(code=prep['status'],message='Preparation process did not complete')),
            preparation=prep)
    else:
        process=launch(out,row['id'],group,'run');folder=out/'runs'/row['id']/group
        result=read(folder/'outcome.json') if (folder/'outcome.json').exists() else dict(id=row['id'],group=group)
        result.update(cohort=row['cohort'],preparation=prep,process=process)
        if process['status']!='completed':result.update(status=process['status'],joint_success=False)
    save(out/'records'/row['id']/(group+'.json'),result)
    return result


def all_records(out):
    return [read(p) for p in sorted((out/'records').glob('*/*.json'))]


def selection(out):
    data={(r['id'],r['group']):r for r in all_records(out) if r['cohort']=='dev'}
    ids=[r['id'] for r in read(out/'manifest.json') if r['cohort']=='dev']
    output={};eligible=[]
    for group in PRIMARY[1:]:
        rescues=[];lost=[];graph_lost=[];alarms=[];geometry_lost=[];queries=0;seconds=0
        for sid in ids:
            a,b=data[(sid,'M0')],data[(sid,group)]
            aa,bb=bool(a.get('joint_success')),bool(b.get('joint_success'))
            if bb and not aa:rescues.append(sid)
            if aa and not bb:lost.append(sid)
            if a.get('score',{}).get('graph_good') and not b.get('score',{}).get('graph_good'):graph_lost.append(sid)
            f=b.get('full',{}).get('fidelity',{})
            if bb and (not f or 'error' in f or f.get('volume_rel_error',0)>.01 or max(f.get('recon_to_truth_max',0),f.get('truth_to_recon_max',0))>.01):alarms.append(sid)
            if aa and bb:
                af=a.get('full',{}).get('fidelity',{})
                if not af or 'error' in af or max(f.get('recon_to_truth_max',float('inf')),f.get('truth_to_recon_max',float('inf')))-max(af.get('recon_to_truth_max',0),af.get('truth_to_recon_max',0))>.0002:
                    geometry_lost.append(sid)
            p=out/'cache'/sid/group/'preparation.json'
            if p.exists():
                prepared=read(p);queries+=sum(c.get('ev_queries',0) for c in prepared.get('queries',{}).get('counts',{}).values())
            seconds+=b['preparation']['seconds']+b.get('process',{}).get('seconds',0)
        ok=len(rescues)-len(lost)>=2 and not lost and not graph_lost and not alarms and not geometry_lost
        output[group]=dict(eligible=ok,rescues=rescues,lost=lost,graph_lost=graph_lost,
            geometry_alarms=alarms,common_geometry_regressions=geometry_lost,ev_queries=queries,total_seconds=seconds)
        if ok:eligible.append(group)
    best=min(eligible,key=lambda g:(output[g]['ev_queries'],output[g]['total_seconds'],g)) if eligible else None
    diagnostic=max(PRIMARY[1:],key=lambda g:(sum(bool(data[(sid,g)].get('joint_success')) for sid in ids),
                    sum(bool(data[(sid,g)].get('score',{}).get('graph_good')) for sid in ids),-output[g]['ev_queries']))
    value=dict(selected=best,diagnostic_best=diagnostic,candidates=output,rule=read(out/'policy.json')['selection'])
    save(out/'selection.json',value);return value


def run_batch(a):
    out=Path(a.output).resolve();verify_identity(out)
    rows=read(out/'manifest.json');dev=[r for r in rows if r['cohort']=='dev'];check=[r for r in rows if r['cohort']=='check']
    policy=read(out/'policy.json');started=time.time()
    def phase(name,jobs):
        save(out/'progress.json',dict(state='running',phase=name,pid=os.getpid(),started_unix=started,
             completed_records=len(all_records(out)),planned_phase_jobs=len(jobs),updated_unix=time.time()))
        with ThreadPoolExecutor(max_workers=policy['jobs']) as pool:
            pending={pool.submit(task,out,row,g):(row,g) for row,g in jobs}
            for future in as_completed(pending):
                row,g=pending[future];result=future.result()
                print(name,row['id'],g,result['status'],flush=True)
                fatal={'refinement_topology_changed','refinement_geometry_changed','refinement_orientation',
                       'refinement_nonmanifold','refinement_parent_map','cache_corrupt','center_cache_mismatch',
                       'field_outside_band','invalid_field_query'}
                prep_file=out/'cache'/row['id']/geometry_group(g)/'preparation.json'
                prep_data=read(prep_file) if prep_file.exists() else {}
                code=result.get('error',{}).get('code')
                if code in fatal or (prep_data.get('last_stage')=='source_audit' and prep_data.get('status')=='failed'):
                    for remaining in pending:remaining.cancel()
                    raise RuntimeError('Stop implementation for invariant/input failure: '+row['id']+' '+g+' '+str(code))
                save(out/'progress.json',dict(state='running',phase=name,pid=os.getpid(),started_unix=started,
                    completed_records=len(all_records(out)),updated_unix=time.time()))
    pilot=[]
    for layer in ('10-25','26-80','81-100'):
        pilot.extend(sorted([r for r in dev if r['layer']==layer],key=lambda r:r['id'])[:2])
    save(out/'pilot_manifest.json',[r['id'] for r in pilot])
    phase('pilot_dev',[(r,g) for r in pilot for g in PRIMARY])
    phase('dev_primary',[(r,g) for r in dev if r not in pilot for g in PRIMARY])
    chosen=selection(out)
    phase('dev_auxiliary',[(r,g) for r in dev for g in ('U1-count40','Area-M0','Area-'+chosen['diagnostic_best'])])
    if chosen['selected']:
        phase('check_pair',[(r,g) for r in check for g in ('M0',chosen['selected'])])
    save(out/'progress.json',dict(state='complete',phase='complete',pid=os.getpid(),started_unix=started,
        finished_unix=time.time(),completed_records=len(all_records(out)),selection=chosen))
    report(out)


def report(out):
    import csv
    rows=all_records(out);groups={}
    for cohort in ('dev','check'):
        for g in sorted(set(r['group'] for r in rows if r['cohort']==cohort)):
            rr=[r for r in rows if r['cohort']==cohort and r['group']==g]
            codes=[]
            for r in rr:
                codes.append(r.get('error',r.get('full',{}).get('error',{})).get('code',r['status']))
            groups[cohort+':'+g]=dict(total=len(rr),face_good=sum(bool(r.get('score',{}).get('face_good')) for r in rr),
                graph_good=sum(bool(r.get('score',{}).get('graph_good')) for r in rr),
                step=sum(r['status']=='solid_validated' for r in rr),joint=sum(bool(r.get('joint_success')) for r in rr),
                first_errors=dict(Counter(codes)),success_ids=[r['id'] for r in rr if r.get('joint_success')])
    save(out/'summary.json',dict(groups=groups,selection=read(out/'selection.json') if (out/'selection.json').exists() else None))
    lines=['# 网格加密后重新查询 UDF 实验','',
        '复用 F30 同模型、同条件256网格。原训练资产目录缺失，使用已核验旧MC缓存，并重建CAD目标、检查旧中心标量一致性。',
        '每例准备7200秒、后处理3600秒，150万三角形，8GiB进程树RSS。局部组目标未达或资源失败保留分母。','',
        '| 集合/组 | 总数 | 面诊断 | 图诊断 | CAD STEP | 图与STEP共同通过 |', '|---|---:|---:|---:|---:|---:|']
    for name,r in groups.items():lines.append(f"| {name} | {r['total']} | {r['face_good']} | {r['graph_good']} | {r['step']} | {r['joint']} |")
    lines+=['','详细逐例记录见 `records/`，选择规则和新增/退化见 `selection.json`，查询及内存见各任务输出。','',
            '## 逐例首个结果','', '| 模型 | 集合 | 组 | 结果 | 图正确 |', '|---|---|---|---|---|']
    for r in rows:
        code=r.get('error',r.get('full',{}).get('error',{})).get('code',r['status'])
        lines.append(f"| {r['id']} | {r['cohort']} | {r['group']} | {code} | {r.get('score',{}).get('graph_good','—')} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    columns=['id','cohort','group','status','code','face_good','graph_good','joint_success',
             'original_triangles','refined_triangles','prepare_seconds','reconstruction_seconds',
             'prepare_ev_requests','prepare_unique_positions','peak_prepare_bytes','peak_run_bytes']
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:
            prep_path=out/'cache'/r['id']/geometry_group(r['group'])/'preparation.json'
            p=read(prep_path) if prep_path.exists() else {};refine=p.get('refinement',{})
            writer.writerow(dict(id=r['id'],cohort=r['cohort'],group=r['group'],status=r['status'],
                code=r.get('error',r.get('full',{}).get('error',{})).get('code',''),
                face_good=r.get('score',{}).get('face_good'),graph_good=r.get('score',{}).get('graph_good'),
                joint_success=r.get('joint_success',False),original_triangles=refine.get('original',{}).get('triangles'),
                refined_triangles=refine.get('refined',{}).get('triangles'),prepare_seconds=r['preparation']['seconds'],
                reconstruction_seconds=r.get('process',{}).get('seconds'),
                prepare_ev_requests=p.get('queries',{}).get('ev_position_requests'),
                prepare_unique_positions=p.get('queries',{}).get('ev_unique_positions'),
                peak_prepare_bytes=r['preparation'].get('peak_process_tree_bytes'),
                peak_run_bytes=r.get('process',{}).get('peak_process_tree_bytes')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--initialize',action='store_true')
    p.add_argument('--run',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--phase',choices=('prepare','run'))
    p.add_argument('--model-id');p.add_argument('--group')
    p.add_argument('--step-root',default='/opt/data/private/yihengxu/Datasets/abc/step/0000')
    p.add_argument('--reference',default=str(ROOT/'outputs/alignment30_20261019_v3'))
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--prepare-timeout',type=int,default=7200)
    p.add_argument('--timeout',type=int,default=3600);a=p.parse_args()
    if a.initialize:freeze(a)
    elif a.worker:
        # Each isolated child is launched with BLAS/OpenMP thread limits in ENV.
        return worker(a)
    elif a.run:
        try:run_batch(a)
        except BaseException as exc:
            save(Path(a.output)/'progress.json',dict(state='manager_failed',pid=os.getpid(),error=repr(exc),unix=time.time()))
            raise
    elif a.report:report(Path(a.output))
    elif a.status:
        out=Path(a.output);print(json.dumps(read(out/'progress.json'),ensure_ascii=False,indent=2))
        for f in sorted((out/'cache').glob('*/*/process.json'))+sorted((out/'runs').glob('*/*/process.json')):
            if read(f)['status']=='running':print('running',f.parent,read(f.parent/'progress.json') if (f.parent/'progress.json').exists() else '')
    else:p.error('Choose --initialize, --run, --status, --report, or --worker')
    return 0


if __name__=='__main__':sys.exit(main())
