"""Induced triangle graph CC, adapted from DualBrep clustering.py (no size filtering)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .schemas import require
from .surface_mesh import face_adjacency,mesh_edges

def core_components(pairs,active):
    ids=np.flatnonzero(active);labels=np.full(len(active),-1,np.int32)
    if not len(ids):return labels
    remap=np.full(len(active),-1,np.int64);remap[ids]=np.arange(len(ids))
    pairs=pairs[active[pairs].all(axis=1)]
    graph=coo_matrix((np.ones(len(pairs)),(remap[pairs[:,0]],remap[pairs[:,1]])),shape=(len(ids),len(ids))).tocsr()
    _,local=connected_components(graph,directed=False);labels[ids]=local
    return labels

def extract_face_seeds(mesh,bands,provider,cfg):
    pairs=face_adjacency(mesh.triangles)
    edges,_,counts=mesh_edges(mesh.triangles)
    shared_edges=edges[counts==2]
    def cut_core_labels(core):
        return core_components(pairs[core[shared_edges].any(axis=1)],core[mesh.triangles].any(axis=1))
    labels=cut_core_labels(bands.core_mask)
    require(np.any(labels>=0),'no_face_core','All faces swallowed by bands; reduce thresholds / refine')
    count=int(labels.max()+1);report={'count':count,'sizes':np.bincount(labels[labels>=0]).tolist(),'threshold_sweep':[]}
    present=provider.presence()
    for factor in cfg.stability_factors:
        active=np.ones(len(bands.values),bool)
        if present[0]=='present':active&=bands.values[:,1]>=cfg.tau_edge*factor
        if present[1]=='present':active&=bands.values[:,2]>=cfg.tau_vertex*factor
        candidate=cut_core_labels(active)
        matches=set(zip(labels[(labels>=0)&(candidate>=0)].tolist(),candidate[(labels>=0)&(candidate>=0)].tolist()))
        bijective=len(matches)==count==int(candidate.max()+1) and len({a for a,b in matches})==count and len({b for a,b in matches})==count
        report['threshold_sweep'].append({'factor':factor,'count':int(candidate.max()+1),'bijective_overlap':bijective})
    report['stable']=all(v['bijective_overlap'] for v in report['threshold_sweep'])
    if cfg.require_stability:require(report['stable'],'unstable_face_seeds','Face cores vary with thresholds; inspect/refine rather than area-filter',report=report)
    return labels,report


def verify_seed_refinement(mesh,bands,labels,report,provider,cfg):
    from dataclasses import replace
    from .bands import build_bands
    fine_cfg=replace(cfg,max_band_edge=cfg.max_band_edge/2,verify_band_stability=False)
    fine_mesh,fine_bands=build_bands(mesh,provider,fine_cfg)
    fine_labels,fine_report=extract_face_seeds(fine_mesh,fine_bands,provider,fine_cfg)
    parent=labels[fine_bands.parent_triangles]
    pairs=set(zip(parent[(parent>=0)&(fine_labels>=0)].tolist(),fine_labels[(parent>=0)&(fine_labels>=0)].tolist()))
    coarse_count=int(labels.max()+1);fine_count=int(fine_labels.max()+1)
    stable=len(pairs)==coarse_count==fine_count and len({a for a,b in pairs})==coarse_count and len({b for a,b in pairs})==fine_count
    refinement={'coarse_count':coarse_count,'fine_count':fine_count,'coarse_triangles':len(mesh.triangles),
                'fine_triangles':len(fine_mesh.triangles),'bijective_overlap':stable,'fine_max_band_edge':fine_cfg.max_band_edge}
    if cfg.require_stability:require(stable,'unstable_mesh_seeds','Face core split/merge under further mesh refinement',report=refinement)
    fine_report['mesh_refinement_stability']=refinement
    fine_bands.report['coarse_level']=bands.report
    return fine_mesh,fine_bands,fine_labels,fine_report
