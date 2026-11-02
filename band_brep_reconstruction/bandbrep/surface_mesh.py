import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .schemas import SurfaceMesh,require


def mesh_edges(triangles):
    directed = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    if not len(directed):
        return np.empty((0, 2), np.int64), np.empty((0, 3), np.int64), np.empty(0, np.int64)
    base = int(directed.max()) + 1
    require(directed.min() >= 0 and base * base <= np.iinfo(np.int64).max,
            'mesh_index_range', 'Vertex indices exceed the edge encoding range')
    lo, hi = directed.min(1), directed.max(1)
    keys, inverse, counts = np.unique(lo * base + hi, return_inverse=True, return_counts=True)
    return np.column_stack((keys // base, keys % base)), inverse.reshape(-1, 3), counts


def face_adjacency(triangles, edge_index=None):
    edges, inv, counts = mesh_edges(triangles) if edge_index is None else edge_index
    ids = np.argsort(inv.ravel(), kind='stable')
    starts = np.r_[0, np.cumsum(counts)]
    shared = np.flatnonzero(counts == 2)
    return np.column_stack((ids[starts[shared]] // 3, ids[starts[shared] + 1] // 3))


def vertex_adjacency(triangles):
    n=len(triangles)
    if not n:return np.empty((0,2),dtype=np.int64)
    incidence=coo_matrix((np.ones(triangles.size,dtype=np.int8),
        (np.repeat(np.arange(n),3),triangles.ravel())),shape=(n,int(triangles.max())+1)).tocsr()
    graph=(incidence@incidence.T).tocoo();keep=graph.row<graph.col
    return np.column_stack((graph.row[keep],graph.col[keep])).astype(np.int64)


def components(pairs,n):
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n)).tocsr()
    return connected_components(graph,directed=False)

def normals(mesh):
    p=mesh.vertices[mesh.triangles]
    g=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    return g/np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-30)

def evaluate_batched(provider,xyz,batch=8192):
    from .field_provider import checked_values
    return checked_values(provider,xyz,batch)


def load_mesh(path):
    """Read a production condition directory or an already normalized mesh file."""
    from pathlib import Path
    import json
    p=Path(path)
    report={}
    if p.is_dir():
        if (p/'condition').is_dir():p=p/'condition'
        from veudf.geometry.sampling import mesh_hash
        from veudf.schema import GEOMETRY_VERSION
        meta=json.loads((p/'meta.json').read_text())
        require(meta.get('geometry_version')==GEOMETRY_VERSION,'mesh_schema','Unsupported mesh condition')
        v=np.load(p/'vertices.npy',allow_pickle=False);f=np.load(p/'triangles.npy',allow_pickle=False)
        require(v.dtype==np.float64 and f.dtype==np.int64,'mesh_schema','Expected float64 vertices and int64 triangles')
        require(mesh_hash(v,f)==meta['mesh_sha256'],'mesh_hash','Condition mesh changed')
        report={'normalization':meta.get('normalization',{}),'geometry_version':meta['geometry_version']}
    else:
        import trimesh
        mesh=trimesh.load(p,force='mesh',process=False)
        v,f=np.asarray(mesh.vertices,float),np.asarray(mesh.faces,np.int64)
    return SurfaceMesh(v,f,report)


def validate_mesh(mesh,cfg):
    from .mesh_densification import mesh_metrics
    v,f=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    require(v.ndim==2 and v.shape[1:]==(3,) and len(v)>0 and np.isfinite(v).all(),
            'invalid_mesh','Expected finite [N,3] vertices')
    require(f.ndim==2 and f.shape[1:]==(3,) and len(f)>0 and np.issubdtype(f.dtype,np.integer)
            and f.min()>=0 and f.max()<len(v),'invalid_mesh','Invalid triangle indices')
    require(np.abs(v).max()<=1.000001,'mesh_scale','Mesh must already use the model coordinates in [-1,1]')
    report=mesh_metrics(mesh)
    require(report['volume']>0,'mesh_orientation','Input mesh must have positive oriented volume')
    mesh.report.update(report)
    return mesh
