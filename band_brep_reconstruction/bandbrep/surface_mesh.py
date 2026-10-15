import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .schemas import SurfaceMesh,require,ReconstructionError


def mesh_edges(triangles):
    directed=triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    edges,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    return edges,inverse.reshape(-1,3),counts

def face_adjacency(triangles):
    edges,inv,counts=mesh_edges(triangles)
    ids=np.argsort(inv.ravel(),kind='stable')
    starts=np.r_[0,np.cumsum(counts)]
    shared=np.flatnonzero(counts==2)
    pair=np.column_stack((ids[starts[shared]]//3,ids[starts[shared]+1]//3))
    return pair

def components(pairs,n):
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n)).tocsr()
    return connected_components(graph,directed=False)

def normals(mesh):
    p=mesh.vertices[mesh.triangles]
    g=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    return g/np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-30)

def evaluate_batched(provider,xyz,batch=8192):
    parts=[]
    for i in range(0,len(xyz),batch):
        result=provider.evaluate(xyz[i:i+batch])
        require(result.valid.all(),'invalid_field_query','Provider marks a required query invalid')
        parts.append(result.values)
    return np.concatenate(parts) if parts else np.empty((0,3))

def extract_surface(provider,cfg):
    from dataclasses import replace
    resolution=cfg.resolution;attempts=[]
    while True:
        try:
            mesh=_extract_surface(provider,replace(cfg,resolution=resolution))
            mesh.report['resolution_attempts']=attempts+[{'resolution':resolution,'accepted':True}]
            return mesh
        except ReconstructionError as exc:
            if exc.code not in ('surface_error','nonmanifold_surface','no_surface') or resolution>=cfg.max_resolution:raise
            attempts.append({'resolution':resolution,'accepted':False,'reason':exc.code})
            resolution=min(cfg.max_resolution,2*(resolution-1)+1)

def _extract_surface(provider,cfg):
    r=cfg.resolution;lo,hi=cfg.bounds;axis=np.linspace(lo,hi,r)
    grid=np.empty(r**3,np.float32)
    for start in range(0,len(grid),cfg.query_batch_size):
        ids=np.arange(start,min(start+cfg.query_batch_size,len(grid)))
        xyz=np.column_stack((axis[ids//(r*r)],axis[(ids//r)%r],axis[ids%r]))
        grid[start:start+len(ids)]=provider.sdf(xyz)
    grid=grid.reshape(r,r,r)
    require(np.isfinite(grid).all(),'invalid_sdf','Nonfinite SDF grid')
    boundary=np.concatenate([grid[0].ravel(),grid[-1].ravel(),grid[:,0].ravel(),grid[:,-1].ravel(),grid[:,:,0].ravel(),grid[:,:,-1].ravel()])
    require(np.all(boundary>cfg.field_error),'domain_boundary','SDF surface reaches query box; enlarge bounds')
    require(grid.min()<0 and grid.max()>0,'no_surface','No SDF sign change; thin features may be unresolved')
    try:
        import mcubes
        v,f=mcubes.marching_cubes(grid,0.)
    except ImportError:
        from skimage.measure import marching_cubes
        v,f,_,_=marching_cubes(grid,level=0.,allow_degenerate=False)
    v=np.asarray(v,float)*(hi-lo)/(r-1)+lo;f=np.asarray(f,np.int64)
    # MC may duplicate vertices when the isovalue equals a grid sample exactly.
    # Coalesce only at roundoff scale of this generated grid, not CAD proximity.
    _,first,inverse=np.unique(np.round(v,12),axis=0,return_index=True,return_inverse=True)
    v=v[first];f=inverse[f]
    # Strip MC degenerate facets before exact index compaction.
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    f=f[np.linalg.norm(cross,axis=1)>1e-14]
    used,inv=np.unique(f,return_inverse=True);v=v[used];f=inv.reshape(-1,3)
    require(len(f)<=cfg.max_triangles,'mesh_resource_limit','SDF mesh exceeds triangle budget',triangles=len(f))
    mesh=SurfaceMesh(v,f)
    orient_and_validate(mesh,provider,cfg)
    mesh.report.update(resolution=r,voxel_size=(hi-lo)/(r-1),strict_topology_guarantee=False)
    return mesh

def orient_and_validate(mesh,provider,cfg):
    edges,inv,counts=mesh_edges(mesh.triangles)
    require(np.all(counts==2),'nonmanifold_surface','Surface must be closed and edge-manifold',boundary_edges=int((counts==1).sum()))
    _,labels=components(face_adjacency(mesh.triangles),len(mesh.triangles))
    normal=normals(mesh);centers=mesh.vertices[mesh.triangles].mean(axis=1)
    for cid in np.unique(labels):
        ids=np.flatnonzero(labels==cid);sample=ids[::max(1,len(ids)//300)]
        step=max(cfg.surface_tolerance*2,1e-4)
        delta=provider.sdf(centers[sample]+step*normal[sample])-provider.sdf(centers[sample]-step*normal[sample])
        if np.median(delta)<0:mesh.triangles[ids]=mesh.triangles[ids][:,[0,2,1]]
        require(np.mean(delta>0)>.8 or np.mean(delta<0)>.8,'orientation_ambiguous','Inconsistent SDF orientation')
    directed=mesh.triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    e,iv,_=mesh_edges(mesh.triangles)
    signs=np.where(directed[:,0]==e[iv.ravel(),0],1,-1)
    require(np.all(np.bincount(iv.ravel(),weights=signs)==0),'orientation_inconsistent','Shared halfedges are not oppositely oriented')
    residual=np.abs(provider.sdf(mesh.vertices))
    mesh.report.update(components=int(labels.max()+1),vertices=len(mesh.vertices),triangles=len(mesh.triangles),
        max_sdf_residual=float(residual.max()),boundary_edges=0)
    exceeded=bool(residual.max()>cfg.surface_tolerance)
    # Retain the normal resolution retry. Only the final resolution can proceed
    # with a residual warning; all structural checks above remain mandatory.
    permitted=cfg.allow_surface_residual_exceedance and cfg.resolution>=cfg.max_resolution
    mesh.report.update(surface_tolerance=cfg.surface_tolerance,
        surface_residual_exceeded=exceeded,
        surface_residual_warning=exceeded and permitted,
        sdf_residual_quantiles=np.quantile(residual,[.5,.95,.99,1.]).tolist(),
        sdf_residual_exceedance_fraction=float(np.mean(residual>cfg.surface_tolerance)))
    require(not exceeded or permitted,'surface_error','MC surface exceeds SDF error budget',error=float(residual.max()))
    return mesh
