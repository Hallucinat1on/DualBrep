"""Replay curves and optional STEP from immutable mesh/topology/vertex caches.

Only the curve algorithm changes. This does not bypass reconstruct() resume
identity checks and never writes into the historical experiment directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_mesh_input_experiment import REPO, MODULE, read, save, digest


def initialize(group,model_id,out,method,stop_after):
    if out.exists():raise RuntimeError('Use a new replay output directory')
    original=group/'runs'/model_id/'reconstruction'
    run=read(original/'run.json')
    for name in ('input_mesh.npz','working_mesh.npz','bands.npz','topology_labels.npz','incidence.json','vertices.json'):
        if name not in run['artifacts'] or digest(original/name)!=run['artifacts'][name]:
            raise RuntimeError('Unverified historical stage artifact: '+name)
    # Scalar observations were made with this dependency version. Do not replace
    # it with an upstream clipped-field implementation during a curve-only test.
    for name,expected in read(group/'identity.json')['code'].items():
        if name.startswith('veudf/src/') and digest(REPO/name)!=expected:
            raise RuntimeError('Distance dependency differs from cached observations: '+name)
    from bandbrep.config import load_config
    cfg=load_config(overrides={**run['config'],'curve_recovery':method,'save_queries':False,
                               'stop_after':'solid' if stop_after=='solid' else 'curves'})
    row=next(r for r in read(group/'manifest.json') if r['id']==model_id and r['cohort']=='dev')
    if digest(row['step'])!=row['source_sha256']:raise RuntimeError('Historical STEP changed')
    out.mkdir(parents=True)
    save(out/'policy.json',dict(input_group=str(group),id=model_id,method=method,stop_after=stop_after,
        config=cfg.to_dict(),row=row,scope='Fixed working mesh, topology and vertices; fresh curve and downstream stages'))
    files=[original/n for n in ('run.json','input_mesh.npz','working_mesh.npz','bands.npz','topology_labels.npz','incidence.json','vertices.json')]
    files += [group/'manifest.json',group/'identity.json',Path(row['step']),group/'runs'/model_id/'source_audit.json',out/'policy.json']
    code={}
    for folder in (MODULE/'bandbrep',MODULE/'benchmarks',REPO/'veudf/src/veudf'):
        for path in sorted(folder.rglob('*.py')):
            relative=path.relative_to(REPO);target=out/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            code[str(relative)]=digest(target)
    save(out/'identity.json',dict(inputs={str(p):digest(p) for p in files},code=code))


def worker(out):
    import numpy as np
    from bandbrep.config import Config
    from bandbrep.deadline import sample_budget
    from bandbrep.field_provider import CallableUDFProvider,MeshFieldProvider
    from bandbrep.schemas import SurfaceMesh,BandState,VertexNode,write_json
    from bandbrep.cache import load_topology
    from bandbrep.curve_distance_fitting import recover_distance_curves
    from bandbrep.edge_geometry import build_edge_geometry
    from bandbrep.curve_fitting import fit_shared_curves
    from bandbrep.geometry_refinement import validate_structure_coverage
    from bandbrep.postprocess import orient_boundaries
    from bandbrep.surface_fitting import initialize_face_surfaces
    from bandbrep.occ_assembly import assemble_brep,verify_export_and_roundtrip
    from veudf.preprocess.config import Config as DistanceConfig
    from veudf.preprocess.cad import read_normalized,collect_targets,triangulate
    from veudf.preprocess.condition import weld
    from run_mesh_input_experiment import ExactUDF,offline_source
    from experiment_scoring import geometry_score,fidelity_score
    identity=read(out/'identity.json')
    for name,h in identity['inputs'].items():
        if digest(name)!=h:raise RuntimeError('Replay input changed: '+name)
    for name,h in identity['code'].items():
        if digest(out/'source_snapshot'/name)!=h:raise RuntimeError('Replay snapshot changed: '+name)
    policy=read(out/'policy.json');cfg=Config(**policy['config']).validate()
    source_dir=Path(policy['input_group'])/'runs'/policy['id'];original=source_dir/'reconstruction'
    def mesh(name):
        with np.load(original/name,allow_pickle=False) as data:
            return SurfaceMesh(data['vertices'],data['triangles'])
    input_mesh=mesh('input_mesh.npz');working=mesh('working_mesh.npz')
    old_audit=read(source_dir/'source_audit.json');input_mesh.report['normalization']=old_audit['normalization']
    with np.load(original/'bands.npz',allow_pickle=False) as data:
        bands=BandState(*(data[k] for k in ('points','values','edge_band','vertex_band','edge_region','core_mask')),
                        face_labels=data['face_labels'],exclusion_reasons=data['exclusion_reasons'])
    topology=load_topology(original)
    vertices=[VertexNode(**{**v,'position':np.asarray(v['position'])}) for v in read(original/'vertices.json')]
    started=time.monotonic();stage='source';result=dict(id=policy['id'],status='running',timings={})
    with sample_budget(cfg.sample_timeout_seconds):
        try:
            distance_cfg=DistanceConfig().validate()
            shape,norm=read_normalized(policy['row']['step'],distance_cfg);targets=collect_targets(shape,distance_cfg)
            assert dict(faces=len(targets.faces),edges=len(targets.edges),vertices=len(targets.vertices))==old_audit['target_counts']
            np.testing.assert_allclose(norm['world_to_normalized'],old_audit['normalization']['world_to_normalized'],atol=1e-10,rtol=0)
            oracle=ExactUDF(targets,distance_cfg)
            udf=CallableUDFProvider(oracle.values,fingerprint='production-cad:'+policy['row']['source_sha256'],
                presence=('present' if targets.edges else 'absent','present' if len(targets.vertices) else 'absent'))
            provider=MeshFieldProvider(input_mesh,udf,cfg)
            stage='curves';tick=time.monotonic()
            if cfg.curve_recovery=='direct_distance':
                graph,curves,report=recover_distance_curves(topology,bands,vertices,provider,cfg,
                    progress=lambda data:save(out/'curve_recovery_progress.json',data))
            else:
                graph=build_edge_geometry(topology,bands,vertices,provider,cfg)
                curves=fit_shared_curves(graph,provider,cfg);report={}
            result['timings']['curve_recovery']=time.monotonic()-tick
            result['distance_fitting']=report
            write_json(out/'logical_graph.json',graph);write_json(out/'curves.json',curves)
            tick=time.monotonic();result['coverage']=validate_structure_coverage(working,bands,curves,provider,cfg)
            result['timings']['global_coverage']=time.monotonic()-tick
            result['curve_kinds']={str(i):c.kind for i,c in curves.items()};result['status']='curves_ready'
            if policy['stop_after']=='solid':
                stage='postprocess';tick=time.monotonic();orient_boundaries(working,topology,graph,curves)
                write_json(out/'cad_graph.json',graph);result['timings'][stage]=time.monotonic()-tick
                stage='surfaces';tick=time.monotonic()
                patches=initialize_face_surfaces(working,topology.face_labels,graph,curves,provider,cfg)
                write_json(out/'face_patches.json',patches);result['timings'][stage]=time.monotonic()-tick
                stage='cad';tick=time.monotonic();solid,mapping=assemble_brep(graph,curves,patches,cfg,out)
                result['cad']=verify_export_and_roundtrip(solid,graph,curves,patches,mapping,cfg,out)
                result['timings'][stage]=time.monotonic()-tick;result['status']='solid_validated'
                stage='scoring';raw=triangulate(targets,distance_cfg);_,mesh_oracle=weld(raw,distance_cfg)
                source=offline_source(targets,raw,mesh_oracle,norm)
                result['geometry']=geometry_score(vertices,curves,source)
                result['fidelity']=fidelity_score(solid,source)
            result['queries']=oracle.report()
        except Exception as exc:
            result.update(status='failed',failed_stage=stage,error=dict(code=getattr(exc,'code',type(exc).__name__),
                message=str(exc),details=getattr(exc,'details',{})))
            raise
        finally:
            result['seconds']=time.monotonic()-started;save(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ('id','status','timings','seconds')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-group',type=Path,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=('projected','direct_distance'),default='direct_distance')
    parser.add_argument('--stop-after',choices=('curves','solid'),default='curves')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:worker(out);return
    initialize(args.input_group.resolve(),args.id,out,args.method,args.stop_after)
    snapshot=out/'source_snapshot';env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    env['PYTHONPATH']=str(snapshot/'veudf/src')+':'+str(snapshot/'band_brep_reconstruction')
    script=snapshot/'band_brep_reconstruction/benchmarks/replay_curve_recovery.py'
    with (out/'worker.log').open('w') as log:
        subprocess.run([sys.executable,str(script),*sys.argv[1:],'--worker'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(out/'result.json')
    print(json.dumps({k:result[k] for k in ('id','status','timings','seconds')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
