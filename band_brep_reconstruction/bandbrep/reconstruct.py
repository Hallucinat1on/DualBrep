"""Staged reconstruction; no target IDs are read by this module."""
import json
import time
import hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
from . import __version__
from .deadline import checkpoint, timed_reconstruction
from .schemas import write_json,jsonable,require,SurfaceMesh,BandState,VertexNode
from .surface_mesh import validate_mesh
from .field_provider import MeshFieldProvider
from .mesh_densification import coarse_refine_surface, fine_refine_surface
from .bands import build_bands
from .face_seeds import extract_face_seeds
from .topology import recover_topology
from .vertex_localization import localize_vertices
from .edge_geometry import build_edge_geometry
from .geometry_refinement import validate_structure_coverage
from .curve_fitting import fit_shared_curves
from .postprocess import orient_boundaries
from .surface_fitting import initialize_face_surfaces


def _reconstruct(mesh,udf,cfg,output_dir,resume=False,prepared=None):
    """Recover B-rep from a closed normalized mesh and continuous metric E/V."""
    cfg.validate();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    from .cache import cache_key,file_hash,load_graph,load_records,load_topology,load_center_fields
    key,cache_input=cache_key(mesh,udf,cfg)
    if prepared is not None:
        from veudf.geometry.sampling import mesh_hash
        working,centers,provenance=prepared
        require(bool(provenance),'prepared_identity_missing','Prepared replay requires verified provenance')
        evidence=hashlib.sha256()
        for array in (centers.points,centers.values):
            a=np.ascontiguousarray(array)
            evidence.update(str((a.shape,a.dtype.str)).encode());evidence.update(a)
        cache_input['prepared_prefix']=dict(working_mesh=mesh_hash(working.vertices,working.triangles),
            center_evidence=evidence.hexdigest(),provider=centers.provider_fingerprint,
            provenance=jsonable(provenance))
        key=hashlib.sha256(json.dumps(cache_input,sort_keys=True,allow_nan=False).encode()).hexdigest()
    prior=None
    if (out/'run.json').exists():
        require(resume,'output_exists','Use --resume for matching immutable caches or a new directory')
        prior=json.loads((out/'run.json').read_text())
        require(prior.get('cache_key')==key,'cache_mismatch','Input/config/code changed; use a new output directory')
        for filename,digest in prior.get('artifacts',{}).items():
            require((out/filename).exists() and file_hash(out/filename)==digest,'cache_corrupt','Stage artifact hash mismatch',file=filename)
        if prior['status']=='solid_validated':return prior
        require(prior.get('effective_config',{}).get('region_repair',cfg.region_repair)==cfg.region_repair,
                'repair_resume_branch','Failed fallback has a different effective policy; replay into a new output directory')
    accepted={v['stage']:v['report'] for v in (prior or {}).get('stages',[])}
    artifacts=dict((prior or {}).get('artifacts',{}))
    run={'version':__version__,'config':cfg.to_dict(),'udf':udf.fingerprint,
         'status':'running','stages':[],'cache_key':key,'cache_inputs':cache_input,'artifacts':artifacts}
    provider=None;attempt_history=[];branch_started=time.monotonic();requested=cfg
    start=time.monotonic();stage='initialization';write_json(out/'run.json',run)
    def finish(status,objects=None):
        checkpoint(stage)
        if hasattr(provider,'flush'):provider.flush()
        run.update(status=status,elapsed_seconds=time.monotonic()-start,query_report=provider.query_report,
                   effective_config=cfg.to_dict(),repair_attempts=attempt_history+[dict(
                       mode=cfg.region_repair,status=status,selected=True,seconds=time.monotonic()-branch_started)])
        write_json(out/'run.json',run)
        write_json(out/'quality.json',{'status':status,'stages':run['stages'],'complete_solid':status=='solid_validated'})
        return objects or run
    stage_files = {
        'mesh': ('input_mesh.npz',),
        'coarse_refinement': ('coarse_mesh.npz',),
        'fine_refinement': ('working_mesh.npz', 'center_fields.npz'),
        'region_repair': ('region_repair.json',),
        'bands': ('bands.npz',), 'seeds': ('face_seeds.npz',),
        'graph': ('topology_labels.npz', 'incidence.json'), 'vertices': ('vertices.json',),
        'curves': ('logical_graph.json', 'curves.json', 'refined_geometry.npz'),
        'postprocess': ('cad_graph.json',), 'surfaces': ('face_patches.json',),
        'cad': ('logical_to_cad.json', 'curves_and_faces.brep', 'reconstruction.step'),
    }
    def mark(name,report):
        checkpoint(name)
        for filename in stage_files[name]:
            artifacts[filename] = file_hash(out / filename)
        run['artifacts']=artifacts
        run['stages'].append({'stage':name,'elapsed_seconds':time.monotonic()-start,'report':report});write_json(out/'run.json',run)
        print(f'[{name}] {json.dumps(jsonable(report),ensure_ascii=False)}',flush=True)
    try:
        stage='mesh'
        checkpoint(stage)
        mesh=validate_mesh(mesh,cfg)
        input_mesh = mesh
        np.savez_compressed(out/'input_mesh.npz',vertices=mesh.vertices,triangles=mesh.triangles)
        mark(stage,mesh.report)
        if prepared is None:
            stage='coarse_refinement'
            checkpoint(stage)
            if stage in accepted:
                with np.load(out/'coarse_mesh.npz', allow_pickle=False) as a:
                    mesh=SurfaceMesh(a['vertices'],a['triangles'])
                    coarse_parent=a['parent_triangles']
                report=accepted[stage]
            else:
                mesh,coarse_parent,report=coarse_refine_surface(mesh,cfg.refinement_coarse_edge,
                    lambda data:write_json(out/'coarse_refinement_progress.json',data))
                np.savez_compressed(out/'coarse_mesh.npz',vertices=mesh.vertices,triangles=mesh.triangles,
                                    parent_triangles=coarse_parent)
            mark(stage,report)
            stage='fine_refinement'
            checkpoint(stage)
            provider=MeshFieldProvider(input_mesh,udf,cfg)
            run.update(provider=provider.fingerprint,presence=provider.presence(),
                       presence_support=provider.presence_support,normalization=provider.normalization)
            if cfg.save_queries:
                from .query_recording import RecordingProvider
                provider=RecordingProvider(provider,out/'queries',cfg.query_batch_size)
            if stage in accepted:
                with np.load(out/'working_mesh.npz', allow_pickle=False) as a:
                    mesh=SurfaceMesh(a['vertices'],a['triangles'])
                center_fields=load_center_fields(out/'center_fields.npz')
                center_fields.validate(mesh.vertices[mesh.triangles].mean(axis=1),provider)
                report=accepted[stage]
            else:
                mesh,fine_parent,report,center_fields=fine_refine_surface(mesh,provider,cfg,
                    lambda data:write_json(out/'fine_refinement_progress.json',data))
                np.savez_compressed(out/'working_mesh.npz',vertices=mesh.vertices,triangles=mesh.triangles,
                                    parent_triangles=coarse_parent[fine_parent])
                np.savez_compressed(out/'center_fields.npz',points=center_fields.points,values=center_fields.values,
                                    provider_fingerprint=np.asarray(center_fields.provider_fingerprint))
            mark(stage,report)
        else:
            mesh,center_fields,provenance=prepared
            require(bool(provenance),'prepared_identity_missing','Prepared replay requires verified provenance')
            provider=MeshFieldProvider(input_mesh,udf,cfg)
            center_fields.validate(mesh.vertices[mesh.triangles].mean(axis=1),provider)
            run.update(provider=provider.fingerprint,presence=provider.presence(),normalization=provider.normalization,
                       prepared_source=provenance)
            np.savez_compressed(out/'working_mesh.npz',vertices=mesh.vertices,triangles=mesh.triangles)
            np.savez_compressed(out/'center_fields.npz',points=center_fields.points,values=center_fields.values,
                                provider_fingerprint=np.asarray(center_fields.provider_fingerprint))
            stage='fine_refinement';mark(stage,dict(reused=True,source=provenance))
        prefix_stages=list(run['stages']);prefix_artifacts=dict(artifacts)
        variants=[cfg]
        if cfg.region_repair=='joint':variants.append(replace(cfg,region_repair='small'))
        if cfg.region_repair!='off':variants.append(replace(cfg,region_repair='off'))
        variants=variants[:cfg.region_repair_max_branches]
        for branch,candidate in enumerate(variants):
            cfg=candidate;branch_started=time.monotonic();run['effective_config']=cfg.to_dict()
            try:
                stage='bands'
                checkpoint(stage)
                if stage in accepted:
                    a=np.load(out/'bands.npz')
                    optional=[a[k] if k in a.files else None for k in ('vertex_labels','vertex_centers')]
                    bands=BandState(*(a[k] for k in ('points','values','edge_band','vertex_band','edge_region','core_mask')),accepted[stage], a['face_labels'], a['exclusion_reasons'],*optional)
                else:mesh,bands=build_bands(mesh,provider,cfg,center_fields)
                for name in ('raw_face_labels','raw_exclusion_reasons','trusted_core_mask','edge_support_mask','absorbed_mask'):
                    if stage in accepted and name in a.files:setattr(bands,name,a[name])
                if stage in accepted and bands.raw_face_labels is not None:
                    from .region_repair import RegionState
                    bands.repair_state=RegionState(bands.raw_face_labels,bands.raw_exclusion_reasons,
                        bands.face_labels,bands.trusted_core_mask,bands.absorbed_mask,
                        a['protected_e'],a['protected_v'],bands.report['region_repair'])
                extra={k:getattr(bands,k) for k in ('vertex_labels','vertex_centers','raw_face_labels',
                    'raw_exclusion_reasons','trusted_core_mask','edge_support_mask','absorbed_mask') if getattr(bands,k) is not None}
                if hasattr(bands,'repair_state'):
                    extra.update(protected_e=bands.repair_state.protected_e,protected_v=bands.repair_state.protected_v)
                    write_json(out/'region_repair.json',bands.report['region_repair'])
                    mark('region_repair',bands.report['region_repair'])
                np.savez_compressed(out/'bands.npz',points=bands.points,values=bands.values,
                    edge_band=bands.edge_band,vertex_band=bands.vertex_band,edge_region=bands.edge_region,core_mask=bands.core_mask,
                    face_labels=bands.face_labels,exclusion_reasons=bands.exclusion_reasons,**extra)
                mark(stage,bands.report)
                stage='seeds'
                checkpoint(stage)
                if stage in accepted:seeds=np.load(out/'face_seeds.npz')['labels'];report=accepted[stage]
                else:seeds,report=extract_face_seeds(mesh,bands,provider,cfg)
                np.savez_compressed(out/'face_seeds.npz',labels=seeds);mark(stage,report)
                if cfg.stop_after=='seeds':return finish('face_seeds_ready')
                stage='graph'
                checkpoint(stage)
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
                checkpoint(stage)
                if stage in accepted:
                    vertices=[VertexNode(**{**v,'position':np.asarray(v['position'])}) for v in json.loads((out/'vertices.json').read_text())]
                else:vertices=localize_vertices(provider,bands,topology,cfg)
                write_json(out/'vertices.json',vertices)
                mark(stage,{'count':len(vertices),'max_distance_residual':max((v.residual for v in vertices),default=0.)})
                if cfg.stop_after=='vertices':return finish('vertices_ready')
                stage='curves'
                checkpoint(stage)
                distance_report=None
                if stage in accepted:
                    graph=load_graph(out/'logical_graph.json');curves=load_records(out/'curves.json','curve')
                elif cfg.curve_recovery=='direct_distance':
                    from .curve_distance_fitting import recover_distance_curves
                    graph,curves,distance_report=recover_distance_curves(topology,bands,vertices,provider,cfg,
                        progress=lambda data:write_json(out/'curve_recovery_progress.json',data))
                else:
                    graph=build_edge_geometry(topology,bands,vertices,provider,cfg)
                    curves=fit_shared_curves(graph,provider,cfg)
                write_json(out/'logical_graph.json',graph);write_json(out/'curves.json',curves)
                np.savez_compressed(out/'refined_geometry.npz',**{f'edge_{e.id}':e.samples for e in graph.edges},
                    vertices=np.array([v.position for v in vertices]).reshape(-1,3))
                from .region_repair import validate_repair_coverage
                repair_coverage=validate_repair_coverage(bands,curves,vertices,provider,cfg)
                coverage=accepted.get(stage,{}).get('coverage') if stage in accepted else validate_structure_coverage(mesh,bands,curves,provider,cfg)
                mark(stage,{'coverage':coverage,'repair_coverage':repair_coverage,'kinds':{i:c.kind for i,c in curves.items()},
                    'method':cfg.curve_recovery,'distance_fitting':distance_report or graph.report.get('distance_fitting'),
                    'max_fit_error':max((c.residual for c in curves.values()),default=0.)})
                if cfg.stop_after=='curves':return finish('curves_ready')
                stage='postprocess'
                checkpoint(stage)
                if stage in accepted:graph=load_graph(out/'cad_graph.json')
                else:orient_boundaries(mesh,topology,graph,curves)
                write_json(out/'cad_graph.json',graph);mark(stage,{'face_cycles':{f:len(c) for f,c in graph.cycles.items()}})
                stage='surfaces'
                checkpoint(stage)
                if stage in accepted:patches=load_records(out/'face_patches.json','patch')
                else:patches=initialize_face_surfaces(mesh,seeds,graph,curves,provider,cfg,trusted=bands.trusted_core_mask)
                write_json(out/'face_patches.json',patches);mark(stage,{i:p.report for i,p in patches.items()})
                if cfg.stop_after=='surfaces':return finish('surfaces_ready')
                stage='cad'
                checkpoint(stage)
                from .occ_assembly import assemble_brep,verify_export_and_roundtrip
                solid,mapping=assemble_brep(graph,curves,patches,cfg,out)
                report=verify_export_and_roundtrip(solid,graph,curves,patches,mapping,cfg,out)
                write_json(out/'logical_to_cad.json',mapping);mark(stage,report)
                return finish('solid_validated',{'solid':solid,'graph':graph,'curves':curves,'patches':patches,'mapping':mapping,'report':run})
            except Exception as exc:
                changed=bool(getattr(locals().get('bands'), 'report', {}).get('region_repair',{}).get('accepted',0))
                attempt=dict(mode=cfg.region_repair,status='failed',selected=False,stage=stage,
                    error={'code':getattr(exc,'code',type(exc).__name__),'message':str(exc)},
                    seconds=time.monotonic()-branch_started,
                    repair=getattr(locals().get('bands'),'report',{}).get('region_repair'))
                attempt_history.append(attempt);run['repair_attempts']=attempt_history
                if not changed or branch+1==len(variants) or getattr(exc,'code',None)=='sample_timeout':raise
                # Preserve failed geometry and its own immutable identity before
                # retrying from the SAME fine mesh and checked scalar centers.
                archive=out/'attempts'/f'{branch:02d}_{cfg.region_repair}'
                archive.mkdir(parents=True,exist_ok=False)
                write_json(archive/'run.json',{**run,'status':'failed','failed_stage':stage})
                names={name for key,files in stage_files.items() if key not in ('mesh','coarse_refinement','fine_refinement') for name in files}
                names.add('curve_recovery_progress.json')
                names.add('reconstruction.unverified.step')
                for path in list(out.glob('rejected_face_*.brep')):names.add(path.name)
                for name in names:
                    if (out/name).exists():(out/name).rename(archive/name)
                artifacts.clear();artifacts.update(prefix_artifacts)
                run['stages']=list(prefix_stages);run['artifacts']=artifacts
                accepted={};write_json(out/'run.json',run)
    except Exception as exc:
        if hasattr(provider,'flush'):provider.flush()
        run.update(status='failed',failed_stage=stage,error={'code':getattr(exc,'code',type(exc).__name__),
            'message':str(exc),'details':getattr(exc,'details',{})},elapsed_seconds=time.monotonic()-start)
        write_json(out/'run.json',run);write_json(out/'quality.json',run)
        raise


@timed_reconstruction
def reconstruct(mesh,udf,cfg,output_dir,resume=False):
    return _reconstruct(mesh,udf,cfg,output_dir,resume=resume)


def reconstruct_from_prepared(input_mesh,working_mesh,center_fields,udf,cfg,output_dir,*,provenance,resume=False):
    """Explicit immutable-prefix replay; never bypass the normal resume key.

    Caller verifies source artifact hashes. CenterFields additionally validates
    exact points and the scalar provider identity before any repair operation.
    """
    from .deadline import sample_budget
    with sample_budget(cfg.sample_timeout_seconds):
        return _reconstruct(input_mesh,udf,cfg,output_dir,resume=resume,
                            prepared=(working_mesh,center_fields,provenance))
