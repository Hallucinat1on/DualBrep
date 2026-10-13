import numpy as np
from .schemas import BandState,require
from .surface_mesh import mesh_edges,split_edges,evaluate_batched


def build_bands(mesh,provider,cfg):
    presence=provider.presence()
    require(presence[0]!='unknown' and presence[1]!='unknown','presence_unknown',
            'Provide predicted or verified edge/vertex existence; absence cannot be inferred from missed minima')
    history=[];parents=np.arange(len(mesh.triangles))
    for level in range(cfg.max_refinements+1):
        vals=evaluate_batched(provider,mesh.vertices,cfg.query_batch_size)
        tri=mesh.vertices[mesh.triangles];centers=tri.mean(axis=1)
        cv=evaluate_batched(provider,centers,cfg.query_batch_size)
        radius=np.linalg.norm(tri-centers[:,None],axis=2).max(axis=1)
        near=np.zeros(len(tri),bool)
        if presence[0]=='present':near|=cv[:,1]-radius-cfg.field_error<=cfg.tau_edge*max(cfg.stability_factors)
        if presence[1]=='present':near|=cv[:,2]-radius-cfg.field_error<=cfg.tau_vertex*max(cfg.stability_factors)
        edges,inv,_=mesh_edges(mesh.triangles)
        length=np.linalg.norm(mesh.vertices[edges[:,0]]-mesh.vertices[edges[:,1]],axis=1)
        selected=np.unique(inv[near].ravel());selected=selected[length[selected]>cfg.max_band_edge]
        history.append({'level':level,'triangles':len(tri),'split_edges':len(selected)})
        if not len(selected):break
        require(level<cfg.max_refinements,'band_resolution_limit','Near-band cells remain too large',history=history)
        require(len(tri)+6*len(selected)<=cfg.max_triangles,'mesh_resource_limit','Refinement triangle budget exceeded',history=history)
        mesh,parent=split_edges(mesh,selected,provider,max_step=cfg.surface_tolerance)
        parents=parents[parent]
    eb=vals[:,1]<cfg.tau_edge if presence[0]=='present' else np.zeros(len(vals),bool)
    vb=vals[:,2]<cfg.tau_vertex if presence[1]=='present' else np.zeros(len(vals),bool)
    core=~(eb|vb)
    # A partially cut triangle still contains a core polygon. Adjacency is
    # permitted only across the outside-band portion of its shared edge.
    tc=core[mesh.triangles].any(axis=1)
    return mesh,BandState(vals,eb,vb,eb&~vb,core,tc,{'refinement':history,'strict_band_certification':False,
          'classification':'linear core cut topology after vertex/centroid distance-radius refinement'},parents)
