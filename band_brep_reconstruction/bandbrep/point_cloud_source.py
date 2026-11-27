"""Point-cloud input adapter: normal orientation, watertight proxy mesh, and raw-point face mapping."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree

from .schemas import SurfaceMesh, require
from .surface_mesh import normals as mesh_normals


def estimate_oriented_normals(points, k=20):
    """Estimate consistently outward-oriented normals on an unoriented point cloud via PCA + MST."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    require(len(pts) >= 16 and np.isfinite(pts).all(), 'invalid_point_cloud',
            'Point cloud must contain at least 16 finite 3D points')
    k_eff = min(k, len(pts))
    tree = cKDTree(pts)
    _, nbrs = tree.query(pts, k=k_eff)

    nrms = np.empty_like(pts)
    for i in range(len(pts)):
        q = pts[nbrs[i]] - pts[nbrs[i]].mean(axis=0)
        _, evecs = np.linalg.eigh(q.T @ q)
        nrms[i] = evecs[:, 0]

    # Build Riemannian MST weighted by 1 - |n_i · n_j| to propagate consistent orientation.
    k_g = min(10, len(pts))
    src = np.repeat(np.arange(len(pts)), k_g - 1)
    dst = nbrs[:, 1:k_g].ravel()
    w = 1.0 - np.abs(np.einsum('ij,ij->i', nrms[src], nrms[dst])) + 1e-6
    graph = csr_matrix((w, (src, dst)), shape=(len(pts), len(pts)))
    mst = minimum_spanning_tree(graph + graph.T).tocoo()
    adj = [[] for _ in range(len(pts))]
    for u, v in zip(mst.row, mst.col):
        adj[u].append(v)
        adj[v].append(u)

    # Anchor outward sign at the point farthest from the centroid.
    centroid = pts.mean(axis=0)
    root = int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))
    if float(np.dot(nrms[root], pts[root] - centroid)) < 0.0:
        nrms[root] = -nrms[root]

    visited = np.zeros(len(pts), dtype=bool)
    visited[root] = True
    stack = [root]
    while stack:
        u = stack.pop()
        nu = nrms[u]
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if float(np.dot(nrms[v], nu)) < 0.0:
                    nrms[v] = -nrms[v]
                stack.append(v)
    return nrms


def build_proxy_mesh(points, normals=None, *, depth=8):
    """Reconstruct a closed, manifold, outward-oriented proxy mesh from a point cloud (Screened Poisson)."""
    import pymeshlab

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if normals is None:
        nrms = estimate_oriented_normals(pts)
    else:
        nrms = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        require(nrms.shape == pts.shape and np.isfinite(nrms).all(), 'invalid_point_cloud',
                'Point cloud normals must match point coordinates and be finite')
        nrms = nrms / np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-12)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=pts, v_normals_matrix=nrms))
    ms.generate_surface_reconstruction_screened_poisson(
        depth=int(depth), pointweight=16.0, samplespernode=1.0, preclean=True, threads=1
    )
    _pct = getattr(pymeshlab, 'PercentageValue', getattr(pymeshlab, 'Percentage', float))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_merge_close_vertices(threshold=_pct(1e-4))
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=25)
    ms.meshing_remove_unreferenced_vertices()

    m = ms.current_mesh()
    verts = np.asarray(m.vertex_matrix(), dtype=np.float64)
    tris = np.asarray(m.face_matrix(), dtype=np.int64)
    # Drop any residual near-zero-area sliver triangles and re-compact vertices if needed.
    if len(tris):
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        if np.any(area2 <= 1e-12):
            ms.meshing_merge_close_vertices(threshold=_pct(1e-3))
            ms.meshing_remove_null_faces()
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_remove_null_faces()
            ms.meshing_remove_unreferenced_vertices()
            m = ms.current_mesh()
            verts = np.asarray(m.vertex_matrix(), dtype=np.float64)
            tris = np.asarray(m.face_matrix(), dtype=np.int64)
    require(len(verts) >= 4 and len(tris) >= 4, 'proxy_mesh_failed',
            'Screened Poisson failed to produce a valid proxy mesh')

    # Ensure positive signed volume (outward orientation).
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    vol = float(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum()) / 6.0
    if vol < 0.0:
        tris = tris[:, ::-1].copy()

    return SurfaceMesh(verts, tris, dict(source='point_cloud_poisson_proxy', poisson_depth=int(depth)))


def map_point_cloud_to_faces(raw_points, raw_normals, cut_mesh, face_labels, cfg):
    """Map raw high-precision point cloud onto cut proxy mesh face regions (proxy topology + raw geometry)."""
    from veudf.geometry.mesh import MeshSDFOracle

    pts = np.asarray(raw_points, dtype=np.float64).reshape(-1, 3)
    nrms = np.asarray(raw_normals, dtype=np.float64).reshape(-1, 3)
    oracle = MeshSDFOracle(cut_mesh.vertices, cut_mesh.triangles)
    tri_n = mesh_normals(cut_mesh)

    sq, tids, _ = oracle.closest(pts)
    dist = np.sqrt(np.maximum(sq, 0.0))
    align = np.einsum('ij,ij->i', nrms, tri_n[tids])
    ok = (dist <= cfg.feature_sdf_limit) & (align >= 0.25)

    regions = {}
    if not ok.any():
        return regions

    sel_pts = pts[ok]
    sel_nrms = nrms[ok]
    sel_lbls = np.asarray(face_labels, dtype=int)[tids[ok]]
    rng = np.random.default_rng(cfg.seed)
    for face in np.unique(sel_lbls):
        idx = np.flatnonzero(sel_lbls == face)
        if len(idx) > cfg.face_samples:
            idx = rng.choice(idx, size=cfg.face_samples, replace=False)
        regions[int(face)] = (sel_pts[idx], sel_nrms[idx])
    return regions
