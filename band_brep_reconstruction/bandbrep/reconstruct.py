"""Staged reconstruction; no target IDs are read by this module."""
import json
import time
from pathlib import Path
import numpy as np
from . import __version__
from .schemas import write_json,jsonable,require,SurfaceMesh,BandState,VertexNode
from .surface_mesh import extract_surface
from .bands import build_bands
from .face_seeds import extract_face_seeds
from .topology import recover_topology
from .vertex_localization import localize_vertices
from .edge_geometry import build_edge_geometry
from .geometry_refinement import validate_structure_coverage
from .curve_fitting import fit_shared_curves
from .postprocess import orient_boundaries
from .surface_fitting import initialize_face_surfaces


def reconstruct(provider,cfg,output_dir,face_model=None,resume=False):
    cfg.validate();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    from .cache import cache_key,file_hash,load_graph,load_records,load_topology
    key,cache_input=cache_key(provider,cfg,face_model)
    prior=None
    if (out/'run.json').exists():
        require(resume,'output_exists','Use --resume for matching immutable caches or a new directory')
        prior=json.loads((out/'run.json').read_text())
        require(prior.get('cache_key')==key,'cache_mismatch','Input/config/code changed; use a new output directory')
        for filename,digest in prior.get('artifacts',{}).items():
            require((out/filename).exists() and file_hash(out/filename)==digest,'cache_corrupt','Stage artifact hash mismatch',file=filename)
        if prior['status']=='solid_validated':return prior
    accepted={v['stage']:v['report'] for v in (prior or {}).get('stages',[])}
    artifacts=dict((prior or {}).get('artifacts',{}))
    run={'version':__version__,'config':cfg.to_dict(),'provider':provider.fingerprint,'presence':provider.presence(),
         'status':'running','stages':[],'cache_key':key,'cache_inputs':cache_input,'artifacts':artifacts,'field_error':provider.error,'strict_field_error_bound':provider.strict_error_bound}
    if cfg.save_queries:
        from .query_recording import RecordingProvider
        provider=RecordingProvider(provider,out/'queries',cfg.query_batch_size)
    start=time.monotonic();stage='initialization';write_json(out/'run.json',run)
    def finish(status,objects=None):
        if hasattr(provider,'flush'):provider.flush()
        run.update(status=status,elapsed_seconds=time.monotonic()-start);write_json(out/'run.json',run)
        write_json(out/'quality.json',{'status':status,'stages':run['stages'],'complete_solid':status=='solid_validated'})
        return objects or run
    def mark(name,report):
        for path in out.iterdir():
            if path.suffix in ('.json','.npz','.brep') and path.name not in ('run.json','quality.json'):
                artifacts[path.name]=file_hash(path)
        run['artifacts']=artifacts
        run['stages'].append({'stage':name,'elapsed_seconds':time.monotonic()-start,'report':report});write_json(out/'run.json',run)
        print(f'[{name}] {json.dumps(jsonable(report),ensure_ascii=False)}',flush=True)
    try:
        base=getattr(provider,'provider',provider)
        if getattr(base,'normalized',False):
            for channel,tau,present in ((1,max(cfg.tau_edge,cfg.face_threshold2),provider.presence()[0]),(2,cfg.tau_vertex,provider.presence()[1])):
                if present=='present':require(tau+cfg.field_error<base.clips[channel],
                    'saturated_band_threshold','Band threshold reaches a clipped UDF plateau; reduce threshold / supply metric unsaturated field')
        if hasattr(provider,'validation'):write_json(out/'field_validation.json',provider.validation)
        stage='surface'
        if stage in accepted:
            a=np.load(out/'sdf_surface.npz');mesh=SurfaceMesh(a['vertices'],a['triangles'],accepted[stage])
        else:mesh=extract_surface(provider,cfg)
        np.savez_compressed(out/'sdf_surface.npz',vertices=mesh.vertices,triangles=mesh.triangles)
        mark(stage,mesh.report)
        stage='bands'
        if stage in accepted:
            a=np.load(out/'bands.npz')
            optional=[a[k] if k in a.files else None for k in ('vertex_labels','vertex_centers')]
            bands=BandState(*(a[k] for k in ('points','values','edge_band','vertex_band','edge_region','core_mask')),accepted[stage], a['face_labels'], a['exclusion_reasons'],*optional)
        else:mesh,bands=build_bands(mesh,provider,cfg)
        extra={k:getattr(bands,k) for k in ('vertex_labels','vertex_centers') if getattr(bands,k) is not None}
        np.savez_compressed(out/'bands.npz',points=bands.points,values=bands.values,
            edge_band=bands.edge_band,vertex_band=bands.vertex_band,edge_region=bands.edge_region,core_mask=bands.core_mask,
            face_labels=bands.face_labels,exclusion_reasons=bands.exclusion_reasons,**extra)
        mark(stage,bands.report)
        stage='seeds'
        if stage in accepted:seeds=np.load(out/'face_seeds.npz')['labels'];report=accepted[stage]
        else:seeds,report=extract_face_seeds(mesh,bands,provider,cfg)
        np.savez_compressed(out/'face_seeds.npz',labels=seeds);mark(stage,report)
        if cfg.stop_after=='seeds':return finish('face_seeds_ready')
        stage='graph'
        topology=load_topology(out) if stage in accepted else recover_topology(mesh,bands,seeds,cfg,provider)
        np.savez_compressed(out/'topology_labels.npz',face_labels=topology.face_labels,
            edge_labels=topology.edge_labels,vertex_labels=topology.vertex_labels,contacts=topology.contacts)
        write_json(out/'incidence.json',dict(face_count=topology.face_count,edges=topology.edges,
            vertices=topology.vertices,report=topology.report,
            face_edges={f:[e['id'] for e in topology.edges if f in e['faces']] for f in range(topology.face_count)},
            face_vertices={f:[v['id'] for v in topology.vertices if f in v['faces']] for f in range(topology.face_count)}))
        mark(stage,topology.report)
        if cfg.stop_after=='graph':return finish('logical_graph_ready')
        stage='vertices'
        if stage in accepted:
            vertices=[VertexNode(**{**v,'position':np.asarray(v['position'])}) for v in json.loads((out/'vertices.json').read_text())]
        else:vertices=localize_vertices(provider,bands,topology,cfg)
        write_json(out/'vertices.json',vertices)
        mark(stage,{'count':len(vertices),'max_distance_residual':max((v.residual for v in vertices),default=0.)})
        if cfg.stop_after=='vertices':return finish('vertices_ready')
        stage='curves'
        if stage in accepted:
            graph=load_graph(out/'logical_graph.json');curves=load_records(out/'curves.json','curve')
        else:
            graph=build_edge_geometry(topology,bands,vertices,provider,cfg)
            curves=fit_shared_curves(graph,provider,cfg)
        write_json(out/'logical_graph.json',graph);write_json(out/'curves.json',curves)
        np.savez_compressed(out/'refined_geometry.npz',**{f'edge_{e.id}':e.samples for e in graph.edges},
            vertices=np.array([v.position for v in vertices]).reshape(-1,3))
        coverage=accepted.get(stage,{}).get('coverage') if stage in accepted else validate_structure_coverage(mesh,bands,curves,provider,cfg)
        mark(stage,{'coverage':coverage,'kinds':{i:c.kind for i,c in curves.items()},
            'max_fit_error':max((c.residual for c in curves.values()),default=0.)})
        if cfg.stop_after=='curves':return finish('curves_ready')
        stage='postprocess'
        if stage in accepted:graph=load_graph(out/'cad_graph.json')
        else:orient_boundaries(mesh,topology,graph,curves)
        write_json(out/'cad_graph.json',graph);mark(stage,{'face_cycles':{f:len(c) for f,c in graph.cycles.items()}})
        stage='surfaces'
        if stage in accepted:patches=load_records(out/'face_patches.json','patch')
        else:patches=initialize_face_surfaces(mesh,seeds,graph,curves,provider,cfg,face_model)
        write_json(out/'face_patches.json',patches);mark(stage,{i:p.report for i,p in patches.items()})
        if cfg.stop_after=='surfaces':return finish('surfaces_ready')
        stage='cad'
        from .occ_assembly import assemble_brep,verify_export_and_roundtrip
        solid,mapping=assemble_brep(graph,curves,patches,cfg,out)
        report=verify_export_and_roundtrip(solid,graph,curves,patches,mapping,cfg,out)
        write_json(out/'logical_to_cad.json',mapping);mark(stage,report)
        return finish('solid_validated',{'solid':solid,'graph':graph,'curves':curves,'patches':patches,'mapping':mapping,'report':run})
    except Exception as exc:
        if hasattr(provider,'flush'):provider.flush()
        run.update(status='failed',failed_stage=stage,error={'code':getattr(exc,'code',type(exc).__name__),
            'message':str(exc),'details':getattr(exc,'details',{})},elapsed_seconds=time.monotonic()-start)
        write_json(out/'run.json',run);write_json(out/'quality.json',run)
        raise
