"""Reproducible E0-E8 benchmark. Truth injection is confined to this driver."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import numpy as np
from bandbrep.config import Config, PRESETS
from bandbrep.schemas import VertexNode, LogicalGraph, EdgeInstance, write_json, require
from bandbrep.surface_mesh import extract_surface
from bandbrep.bands import build_bands
from bandbrep.topology import recover_topology
from bandbrep.vertex_localization import localize_vertices
from bandbrep.edge_geometry import build_edge_geometry
from bandbrep.curve_fitting import fit_shared_curves, CurveRecord
from bandbrep.geometry_refinement import validate_structure_coverage
from bandbrep.postprocess import orient_boundaries
from bandbrep.surface_fitting import initialize_face_surfaces
from experiment_fields import BenchmarkCAD, ScalarOnly, PerturbedField
from experiment_scoring import score, geometry_score, fidelity_score

S30='03 07 08 09 22 24 31 35 39 49 54 57 60 61 63 64 65 68 70 72 77 78 79 93 102 103 104 105 106 116'.split()
S30=[f'{int(x):08d}' for x in S30]
S8=[f'{int(x):08d}' for x in '07 09 22 61 68 70 93 102'.split()]
# 'all' is the library 'recommended' preset; 'base' switches its four improvements off
# (the 2026-10-16 baseline).  Leave-one-out variants drop one improvement each.
_VERTEX=dict(vertex_domains='local',infer_short_edges=True)
_SURFACE=dict(surface_candidates='revolution',curve_candidates='arcs')
_CAD=dict(periodic_frame_planning=True)
ALL=Config(**PRESETS['recommended'])
BASE=replace(ALL,vertex_domains='threshold',infer_short_edges=False,surface_candidates='basic',curve_candidates='basic',periodic_frame_planning=False)
VARIANTS={'base':BASE,'all':ALL,
    'no_vertex':replace(BASE,**_SURFACE,**_CAD),'no_surface':replace(BASE,**_VERTEX,**_CAD),
    'no_cad':replace(BASE,**_VERTEX,**_SURFACE)}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    files=sorted(list((HERE.parent/'bandbrep').rglob('*.py'))+list(HERE.glob('experiment_*.py'))+[Path(__file__)])
    return hashlib.sha256(''.join(str(p.relative_to(HERE.parent))+digest(p) for p in files).encode()).hexdigest()


def error(exc):return {'code':getattr(exc,'code',type(exc).__name__),'message':str(exc),'details':getattr(exc,'details',{})}


def prepare(source, cfg, root, field=None):
    provider=ScalarOnly(field or source)
    key=hashlib.sha256(json.dumps({'provider':provider.fingerprint,'source_hash':source_hash(),
        'resolution':cfg.resolution,'max_resolution':cfg.max_resolution,'max_triangles':cfg.max_triangles,
        'surface_tolerance':cfg.surface_tolerance,'field_error':cfg.field_error,'bounds':cfg.bounds,
        'allow_residual':cfg.allow_surface_residual_exceedance},sort_keys=True).encode()).hexdigest()[:16]
    path=root/'cache'/f'{key}.npz';report=path.with_suffix('.json')
    if path.exists() and report.exists():
        metadata=json.loads(report.read_text());require(digest(path)==metadata['sha256'],'cache_corrupt','Experiment mesh cache hash mismatch')
        with np.load(path) as data:
            mesh=SurfaceMesh(data['vertices'],data['triangles'],metadata['surface'])
            points=data['points'];values=data['values']
        return mesh,points,values,provider
    start=time.monotonic();mesh=extract_surface(provider,cfg)
    points=mesh.vertices[mesh.triangles].mean(axis=1);values=provider.evaluate(points).values
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,vertices=mesh.vertices,triangles=mesh.triangles,points=points,values=values)
    write_json(report,{'sha256':digest(path),'surface':mesh.report,'elapsed_seconds':time.monotonic()-start,'provider':provider.fingerprint})
    return mesh,points,values,provider


class FixedCenters(ScalarOnly):
    def __init__(self,provider,points,values):
        super().__init__(provider);self.points=points;self.values=values
    def evaluate(self,p):
        if np.shape(p)==self.points.shape and np.array_equal(p,self.points):return self.result(self.values)
        return super().evaluate(p)


def segment(mesh,points,values,provider,cfg):
    _,bands=build_bands(mesh,FixedCenters(provider,points,values),cfg)
    bands.points=points
    return bands


def topology(mesh,bands,cfg,provider=None):
    return recover_topology(mesh,bands,bands.face_labels,cfg,provider)


def save_graph(folder,bands,topo):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'topology_labels.npz',face_labels=topo.face_labels,edge_labels=topo.edge_labels,vertex_labels=topo.vertex_labels,contacts=topo.contacts)
    write_json(folder/'incidence.json',{'face_count':topo.face_count,'edges':topo.edges,'vertices':topo.vertices,'report':topo.report})


def diagnose_edges(bands,topo,vertices,provider,cfg):
    output=[]
    for item in topo.edges:
        try:
            graph=build_edge_geometry(replace(topo,edges=[item]),bands,vertices,provider,cfg)
            curves=fit_shared_curves(graph,provider,cfg)
            output.append({'id':item['id'],'status':'curves_ready','kind':curves[item['id']].kind})
        except Exception as exc:output.append({'id':item['id'],'status':'failed','error':error(exc)})
    return output


def truth_geometry(source,cfg):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle
    vertices=[VertexNode(i,p,0.,0.) for i,p in enumerate(source.targets.vertices)]
    edges=[];curves={}
    for item,edge,p in zip(source.targets.topology['edges'],source.targets.edges,source.edge.pieces):
        eid=item['edge_id'];c=BRepAdaptor_Curve(edge);p=p[::max(1,len(p)//4000)]
        ends=np.array([[q.X(),q.Y(),q.Z()] for q in (c.Value(c.FirstParameter()),c.Value(c.LastParameter()))])
        p=np.vstack((ends[0],p,ends[1]));p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-10]]
        vids=list(item['vertex_ids'])
        edges.append(EdgeInstance(eid,tuple(item['face_ids']),p,not vids,vids,-1,0.))
        if c.GetType()==GeomAbs_Line:
            curves[eid]=CurveRecord(eid,'line',{'p0':ends[0],'p1':ends[1]},0.)
        elif c.GetType()==GeomAbs_Circle:
            circle=c.Circle()
            xyz=lambda q:np.array([q.X(),q.Y(),q.Z()])
            curves[eid]=CurveRecord(eid,'circle',{'center':xyz(circle.Location()),'radius':circle.Radius(),
                'x':xyz(circle.XAxis().Direction()),'y':xyz(circle.YAxis().Direction()),'normal':xyz(circle.Axis().Direction()),
                'start':c.FirstParameter(),'sweep':c.LastParameter()-c.FirstParameter()},0.)
        else:
            fitted=fit_shared_curves(LogicalGraph(len(source.targets.faces),vertices,[edges[-1]]),ScalarOnly(source),cfg)
            curves[eid]=fitted[eid]
    return LogicalGraph(len(source.targets.faces),vertices,edges),curves


def full(source,mesh,bands,topo,cfg,folder,injection='none',diagnostics=False,field=None):
    folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();stage='vertices';stages=[];vertices=[];curves={}
    result={'config':cfg.to_dict(),'injection':injection,'status':'running'}
    try:
        provider=ScalarOnly(field or source)
        if injection=='gt_geometry':
            stage='gt_geometry_construction'
            labels=source.truth_face(bands.points);bands=replace(bands,face_labels=labels,core_mask=np.ones(len(labels),bool))
            graph,curves=truth_geometry(source,cfg);vertices=graph.vertices
            topo=replace(topo,face_count=len(source.targets.faces),face_labels=labels)
        else:
            if injection=='gt_vertices':
                stage='gt_vertex_mapping'
                scored=score(topo,bands,source)
                require(all(v>=0 for v in scored['vertex_map']),'gt_vertex_mapping_failed','Candidate vertex has no valid truth support')
                vertices=[VertexNode(v['id'],source.targets.vertices[scored['vertex_map'][v['id']]],0.,0.) for v in topo.vertices]
            else:vertices=localize_vertices(provider,bands,topo,cfg)
            write_json(folder/'vertices.json',vertices);stages.append('vertices');stage='curves'
            if diagnostics:write_json(folder/'edge_diagnostics.json',diagnose_edges(bands,topo,vertices,provider,cfg))
            graph=build_edge_geometry(topo,bands,vertices,provider,cfg)
            curves=fit_shared_curves(graph,provider,cfg)
        write_json(folder/'logical_graph.json',graph);write_json(folder/'curves.json',curves)
        stages.append('curves');stage='coverage'
        result['coverage']=validate_structure_coverage(mesh,bands,curves,provider,cfg)
        result['geometry']=geometry_score(vertices,curves,source)
        stage='postprocess';orient_boundaries(mesh,topo,graph,curves);stages.append(stage)
        write_json(folder/'cad_graph.json',graph)
        stage='surfaces';patches=initialize_face_surfaces(mesh,bands.face_labels,graph,curves,provider,cfg)
        write_json(folder/'face_patches.json',patches);stages.append(stage)
        stage='cad'
        from bandbrep.occ_assembly import assemble_brep,verify_export_and_roundtrip
        solid,mapping=assemble_brep(graph,curves,patches,cfg,folder)
        result['cad']=verify_export_and_roundtrip(solid,graph,curves,patches,mapping,cfg,folder)
        write_json(folder/'logical_to_cad.json',mapping);stages.append(stage)
        result['status']='solid_validated'
        try:result['fidelity']=fidelity_score(solid,source)
        except Exception as exc:result['fidelity']={'error':error(exc)}
    except Exception as exc:
        result.update(status='failed',failed_stage=stage,error=error(exc))
    result.update(stages=stages,elapsed_seconds=time.monotonic()-start)
    write_json(folder/'result.json',result)
    return result


def graph_case(source,mesh,points,values,provider,cfg):
    bands=segment(mesh,points,values,provider,cfg);topo=topology(mesh,bands,cfg,provider)
    return bands,topo,score(topo,bands,source)


def graph_worker(step,root,phase):
    run_source=source_hash()
    require(not (root/'result.json').exists(),'output_exists','Use a new experiment output directory')
    start=time.monotonic();source=BenchmarkCAD(step);root.mkdir(parents=True,exist_ok=True)
    validation=source.validation();write_json(root/'field_validation.json',validation)
    require(validation['edge_reference']['passed'],'edge_reference_failed','Fast field reference error exceeds 1e-5',validation=validation)
    mesh,points,values,provider=prepare(source,BASE,root)
    bands,topo,scored=graph_case(source,mesh,points,values,provider,BASE)
    save_graph(root/'baseline',bands,topo)
    output={'step':str(step),'provider':source.fingerprint,'validation':validation,'surface':mesh.report,
        'baseline':scored,'config':BASE.to_dict(),'cases':{},'source_hash':run_source,'batch_status':'running',
        'variant':os.environ.get('BANDBREP_VARIANT','base'),
        'vertex_domains':bands.report.get('vertex_domains'),'short_edges':topo.report.get('short_edges')}
    write_json(root/'result.json',output)
    if phase=='graph':
        cases={'edge_distance_tau04':replace(BASE,band_connectivity='edge',band_propagation='distance_only',tau_vertex=.04),
            'edge_noncore_tau04':replace(BASE,band_connectivity='edge',tau_vertex=.04),
            'vertex_distance_tau04':replace(BASE,band_propagation='distance_only',tau_vertex=.04),
            'vertex_noncore_tau04':replace(BASE,tau_vertex=.04),
            'vertex_distance_tau02':replace(BASE,band_propagation='distance_only'),
            'filter3':replace(BASE,face_filter_size=3),'filter5':replace(BASE,face_filter_size=5),
            'filter1':replace(BASE,face_filter_size=1),'threshold_half':replace(BASE,face_threshold1=.0025,face_threshold2=.005),
            'no_second_pass':replace(BASE,face_threshold2=float(values[:,1].max()+1)),
            'tau01':replace(BASE,tau_vertex=.01),'tau03':replace(BASE,tau_vertex=.03)}
        for name,cfg in cases.items():
            b,t,s=graph_case(source,mesh,points,values,provider,cfg);output['cases'][name]={'score':s,'config':cfg.to_dict()}
            write_json(root/'result.json',output)
        labels=source.truth_face(points).copy();labels[~bands.core_mask]=-1
        injected=replace(bands,face_labels=labels)
        output['cases']['gt_core_labels']={'score':score(topology(mesh,injected,BASE),injected,source),'injection':True}
    if phase in ('full','isolate','holdout','pipeline'):
        output['full']=full(source,mesh,bands,topo,BASE,root/'full',diagnostics=phase=='full')
        write_json(root/'result.json',output)
        if phase=='isolate':
            output['gt_vertices']=full(source,mesh,bands,topo,BASE,root/'gt_vertices',injection='gt_vertices')
            output['gt_geometry']=full(source,mesh,bands,topo,BASE,root/'gt_geometry',injection='gt_geometry')
    if phase=='noise':
        for amplitude in (.0001,.0005,.001,.002):
            noisy=PerturbedField(source,amplitude)
            try:
                nm,npnt,nval,nprovider=prepare(source,BASE,root,noisy)
                b,t,s=graph_case(source,nm,npnt,nval,nprovider,BASE)
                output['cases'][f'noise_{amplitude}']={'score':s,'amplitude_bound':amplitude,'surface':nm.report,'full':full(source,nm,b,t,BASE,root/f'noise_{amplitude}',field=noisy)}
            except Exception as exc:output['cases'][f'noise_{amplitude}']={'error':error(exc),'amplitude_bound':amplitude}
            write_json(root/'result.json',output)
    require(source_hash()==run_source,'source_changed','Experiment sources changed while worker was running')
    output.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_hash=run_source,batch_status='complete',field_queries=source.queries,sdf_queries=source.sdf_queries)
    write_json(root/'result.json',output)


def select_holdout(data,root,seed=20261016,exclude_manifests=(),name='holdout_manifest.json'):
    from bandbrep.vendor.oracle.config import Config as OC
    from bandbrep.vendor.oracle.cad import read_normalized,collect_targets
    used={next((data/m).glob('*.step')).name.split('_')[1] for m in S30}
    for path in exclude_manifests:used|={r['group'] for r in json.loads(Path(path).read_text())['selected']}
    rng=np.random.default_rng(seed);folders=sorted(p for p in data.iterdir() if p.is_dir())
    order=rng.permutation(len(folders));selected=[];screened=[];bins=[0,0,0,0]
    seen=set(used);hashes=set()
    for index in order:
        if len(selected)==100:break
        folder=folders[index];files=sorted(folder.glob('*.step'))
        if not files:continue
        path=files[0];group=path.name.split('_')[1]
        if group in seen:continue
        try:
            solid,_=read_normalized(path,OC());targets=collect_targets(solid,OC())
            nf=len(targets.faces);bucket=0 if nf<=8 else 1 if nf<=25 else 2 if nf<=80 else 3
            if nf>250 or bins[bucket]>=25:
                screened.append({'step':str(path),'reason':'quota_or_face_budget','faces':nf});continue
            sha=digest(path)
            if sha in hashes:continue
            seen.add(group);hashes.add(sha);bins[bucket]+=1
            selected.append({'id':folder.name,'step':str(path),'group':group,'sha256':sha,'counts':{'faces':nf,'edges':len(targets.edges),'vertices':len(targets.vertices)},'stratum':bucket})
        except Exception as exc:screened.append({'step':str(path),'error':error(exc)})
    require(len(selected)==100,'holdout_selection_incomplete','Could not fill 100-model strata',bins=bins)
    write_json(root/name,{'seed':seed,'selected':selected,'screened':screened,'strata':bins,'excluded_groups':sorted(used),
        'excluded_manifests':[str(p) for p in exclude_manifests],'scope':'25 each F<=8,9-25,26-80,81-250; first STEP per folder; one model per source document group; not ABC population estimate'})
    return selected


def launch(job,timeout):
    step,root,phase=job;root.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(Path(__file__)), '--worker','--step',str(step),'--output',str(root),'--phase',phase]
    before=time.monotonic()
    with (root/'worker.log').open('w') as log:
        try:
            completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=False)
            status='complete' if completed.returncode==0 else 'process_failed'
            row={'phase':phase,'step':str(step),'output':str(root),'status':status,'returncode':completed.returncode}
        except subprocess.TimeoutExpired:row={'phase':phase,'step':str(step),'output':str(root),'status':'timeout','timeout':timeout}
    row['wall_seconds']=time.monotonic()-before
    write_json(root/'process.json',row);print(phase,Path(step).parent.name,row['status'],round(row['wall_seconds'],1),flush=True)
    return row


def main():
    global BASE
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--phase',choices=('graph','full','isolate','holdout','noise','all','pipeline'),default='graph')
    p.add_argument('--worker',action='store_true');p.add_argument('--step',type=Path);p.add_argument('--jobs',type=int,default=2);p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--variant',choices=sorted(VARIANTS),default=None,help='configuration variant; workers inherit it via BANDBREP_VARIANT')
    p.add_argument('--holdout-manifest',type=Path,help='reuse a frozen holdout_manifest.json for the pipeline phase')
    p.add_argument('--ids',help='comma-separated model ids (pipeline phase only)')
    p.add_argument('--sets',default='s30,holdout',help='pipeline model sets: s30,holdout')
    args=p.parse_args();root=args.output.resolve()
    if args.variant:os.environ['BANDBREP_VARIANT']=args.variant
    BASE=VARIANTS[os.environ.get('BANDBREP_VARIANT','base')]
    if args.worker:
        try:graph_worker(args.step,root,args.phase)
        except Exception as exc:
            root.mkdir(parents=True,exist_ok=True);write_json(root/'failure.json',{'error':error(exc),'trace':traceback.format_exc()});raise
        return
    require(args.data is not None,'data_required','Supply --data STEP folder')
    root.mkdir(parents=True,exist_ok=True)
    import OCC,scipy
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    manifest=[{'id':m,'step':str(next((args.data/m).glob('*.step'))),'sha256':digest(next((args.data/m).glob('*.step')))} for m in S30]
    write_json(root/'manifest.json',{'models':manifest,'S8':S8,'source_hash':source_hash(),'environment':{'python':sys.version,'OCC':OCC.VERSION,'numpy':np.__version__,'scipy':scipy.__version__},
        'base_config':BASE.to_dict(),'variant':os.environ.get('BANDBREP_VARIANT','base'),'timeout_seconds':args.timeout,
        'holdout_manifest':str(args.holdout_manifest) if args.holdout_manifest else None})
    if args.phase=='pipeline':
        require(args.holdout_manifest is not None,'holdout_required','pipeline reuses a frozen holdout manifest')
        hold=json.loads(args.holdout_manifest.read_text())['selected']
        sets={'s30':manifest,'holdout':[{'id':r['id'],'step':r['step']} for r in hold]}
        sets={k:v for k,v in sets.items() if k in args.sets.split(',')}
        keep=set(args.ids.split(',')) if args.ids else None
        for name,models in sets.items():
            jobs=[(Path(r['step']),root/name/r['id'],'pipeline') for r in models if keep is None or r['id'] in keep]
            if not jobs:continue
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:rows=list(pool.map(lambda j:launch(j,args.timeout),jobs))
            write_json(root/f'{name}_processes.json',rows)
        return
    phases=['graph','full','isolate','holdout','noise'] if args.phase=='all' else [args.phase]
    for phase in phases:
        selected=select_holdout(args.data,root) if phase=='holdout' else [r for r in manifest if phase not in ('full','noise') or r['id'] in S8]
        jobs=[(Path(r['step']),root/phase/r['id'],phase) for r in selected]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:rows=list(pool.map(lambda j:launch(j,args.timeout),jobs))
        write_json(root/f'{phase}_processes.json',rows)


if __name__=='__main__':main()
