import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra,connected_components
from .schemas import ArrivalState,require
from .surface_mesh import mesh_edges


def propagate_arrivals(mesh,bands,seed_labels,cfg):
    """All touching sources retained; paths exclude vertex band and other cores."""
    n=len(mesh.vertices);k=int(seed_labels.max()+1)
    require(n*k<=50000000,'arrival_resource_limit','Arrival matrix exceeds budget',nodes=n,sources=k)
    core_owner=np.full(n,-1,np.int32);contacts={}
    for f in range(k):
        nodes=np.unique(mesh.triangles[seed_labels==f]);nodes=nodes[bands.core_mask[nodes]];contacts[f]=nodes
        require(np.all((core_owner[nodes]<0)|(core_owner[nodes]==f)),'core_contact_conflict','Different face cores share a vertex; refine mesh')
        core_owner[nodes]=f
    ordinary=(core_owner<0)&~bands.vertex_band
    # Cells outside the core cover but not inside the nominal band belong to the
    # conservative contact collar; this avoids disconnected discretization gaps.
    edges,_,_=mesh_edges(mesh.triangles)
    length=np.linalg.norm(mesh.vertices[edges[:,0]]-mesh.vertices[edges[:,1]],axis=1)
    costs=np.full((k,n),np.inf)
    pairs=edges[ordinary[edges].all(axis=1)]
    remap=np.full(n,-1,int);ids=np.flatnonzero(ordinary);remap[ids]=np.arange(len(ids))
    graph=coo_matrix((np.repeat(np.linalg.norm(mesh.vertices[pairs[:,0]]-mesh.vertices[pairs[:,1]],axis=1),2),
         (remap[pairs[:,[0,1]]].ravel(),remap[pairs[:,[1,0]]].ravel())),shape=(len(ids),len(ids))).tocsr()
    _,regions=connected_components(graph,directed=False)
    for f in range(k):
        allowed=ordinary|(core_owner==f)
        pair=edges[allowed[edges].all(axis=1)];weights=np.linalg.norm(mesh.vertices[pair[:,0]]-mesh.vertices[pair[:,1]],axis=1)
        g=coo_matrix((np.repeat(weights,2),(pair[:,[0,1]].ravel(),pair[:,[1,0]].ravel())),shape=(n,n)).tocsr()
        # min_only is genuinely multi-source, not a dense distance matrix.
        costs[f]=dijkstra(g,directed=False,indices=contacts[f],min_only=True)
        costs[f,bands.vertex_band]=np.inf
    reachable=np.isfinite(costs).any(axis=0)
    require(np.all(reachable[ordinary]),'unexplained_band','Band component has no face source')
    winner=np.argmin(costs,axis=0).astype(np.int32);winner[~reachable]=-1
    tri_labels=np.full(len(mesh.triangles),-1,np.int32)
    for i,ids3 in enumerate(mesh.triangles):
        finite=np.isfinite(costs[:,ids3]).all(axis=1)
        if finite.any():tri_labels[i]=np.flatnonzero(finite)[np.argmin(costs[finite][:,ids3].sum(axis=1))]
        else:
            labs=winner[ids3];labs=labs[labs>=0]
            if len(labs):tri_labels[i]=np.bincount(labs).argmax()
    return ArrivalState(costs,winner,tri_labels,contacts)
