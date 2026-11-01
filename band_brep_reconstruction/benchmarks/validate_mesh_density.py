"""Analytic smoke checks for each density group; never supplies instance IDs to stages."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from run_mesh_density_experiments import save, settings
from bandbrep.config import load_config
from bandbrep.demo_fields import box,cylinder,sphere_with_bore,split_sphere,hollow_sphere
from bandbrep.mesh_densification import densify_surface_mesh
from bandbrep.surface_mesh import extract_surface
from bandbrep.bands import build_bands
from bandbrep.topology import recover_topology
from bandbrep.vertex_localization import localize_vertices
from bandbrep.edge_geometry import build_edge_geometry
from bandbrep.curve_fitting import fit_shared_curves
from bandbrep.geometry_refinement import validate_structure_coverage
from bandbrep.postprocess import orient_boundaries
from bandbrep.surface_fitting import initialize_face_surfaces
from bandbrep.occ_assembly import assemble_brep,verify_export_and_roundtrip


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(preset='recommended',overrides={'query_batch_size':65536,'feature_sdf_tolerance':.004})
    result={}
    for name,make,expected in [('box',box,(6,12,8)),('cylinder',cylinder,(3,2,0)),
        ('sphere_bore',sphere_with_bore,(2,2,0)),('split_sphere',split_sphere,(2,1,0)),('hollow_sphere',hollow_sphere,(2,0,0))]:
        provider=make();base=extract_surface(provider,cfg)
        for group in ('M0','U1','L5','L25'):
            start=time.monotonic();path=out/name/group;path.mkdir(parents=True,exist_ok=True)
            entry=dict(name=name,group=group,expected=expected)
            try:
                mesh,parent,report=densify_surface_mesh(base,provider,settings(group))
                entry['refinement']=report
                _,bands=build_bands(mesh,provider,cfg)
                topo=recover_topology(mesh,bands,bands.face_labels,cfg,provider)
                entry['counts']=(topo.face_count,len(topo.edges),len(topo.vertices))
                vertices=localize_vertices(provider,bands,topo,cfg)
                graph=build_edge_geometry(topo,bands,vertices,provider,cfg)
                curves=fit_shared_curves(graph,provider,cfg)
                validate_structure_coverage(mesh,bands,curves,provider,cfg)
                orient_boundaries(mesh,topo,graph,curves)
                patches=initialize_face_surfaces(mesh,bands.face_labels,graph,curves,provider,cfg)
                solid,mapping=assemble_brep(graph,curves,patches,cfg,path)
                verify_export_and_roundtrip(solid,graph,curves,patches,mapping,cfg,path)
                entry['status']='solid_validated';entry['counts_match']=entry['counts']==expected
            except Exception as exc:
                entry.update(status='failed',code=getattr(exc,'code',type(exc).__name__),message=str(exc))
            entry['seconds']=time.monotonic()-start
            result[name+':'+group]=entry
            save(out/'report.json',result)
            print(name,group,entry['status'],entry.get('code',''),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
