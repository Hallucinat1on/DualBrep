"""Conforming linear refinement of an existing surface; only scalar fields are queried."""
from dataclasses import dataclass, asdict
import time
import numpy as np
from .schemas import SurfaceMesh, require
from .surface_mesh import mesh_edges, face_adjacency, components, evaluate_batched


@dataclass(frozen=True)
class RefinementConfig:
    method: str = 'none'
    target_edge: float = .005
    rho_edge: float = .02
    rho_vertex: float = .03
    max_rounds: int = 4
    max_triangles: int = 1500000
    query_batch: int = 8192


def areas(mesh):
    p = mesh.vertices[mesh.triangles]
    return .5*np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)


def mesh_metrics(mesh):
    v, f = mesh.vertices, mesh.triangles
    edges, inv, counts = mesh_edges(f)
    require(np.all(counts == 2), 'refinement_nonmanifold', 'Refinement surface has open/nonmanifold edges')
    directed = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    signs = np.where(directed[:, 0] == edges[inv.ravel(), 0], 1, -1)
    require(np.all(np.bincount(inv.ravel(), weights=signs) == 0), 'refinement_orientation', 'Refinement changed halfedge orientation')
    a = areas(mesh)
    require(np.all(a > 0), 'refinement_degenerate', 'Refinement has zero-area triangles')
    p = v[f]
    volume = float(np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()/6)
    lengths = np.linalg.norm(v[edges[:, 0]]-v[edges[:, 1]], axis=1)
    nc, _ = components(face_adjacency(f), len(f))
    return dict(vertices=len(v), triangles=len(f), edges=len(edges), components=int(nc),
                euler=int(len(v)-len(edges)+len(f)), area=float(a.sum()), volume=volume,
                edge_quantiles=np.quantile(lengths, [.5, .95, .99, 1]).tolist(),
                area_quantiles=np.quantile(a, [.5, .95, .99, 1]).tolist())


def split_edges(mesh, marked, max_triangles):
    """Split globally shared marked edges, triangulating every incident face (red/green)."""
    v, f = mesh.vertices, mesh.triangles
    edges, inv, _ = mesh_edges(f)
    require(np.shape(marked) == (len(edges),), 'refinement_mask', 'Edge mask has wrong length')
    cuts = marked[inv]
    counts = cuts.sum(axis=1)
    total = int(np.sum(counts+1))
    require(total <= max_triangles, 'mesh_resource_limit', 'Conforming refinement exceeds triangle budget',
            before=len(f), proposed=total, max_triangles=max_triangles)
    mids = np.full(len(edges), -1, np.int64)
    mids[marked] = len(v)+np.arange(int(marked.sum()))
    nv = np.vstack((v, v[edges[marked]].mean(axis=1)))
    mf = mids[inv]
    code = cuts@np.array([1, 2, 4])
    output, parents = [], []

    def add(ids, tri):
        tri = np.asarray(tri, np.int64)
        if not len(ids): return
        output.append(tri.reshape(-1, 3))
        parents.append(np.repeat(ids, tri.shape[1]))

    ids = np.flatnonzero(code == 0)
    add(ids, f[ids, None, :])
    for c, rotation in ((1, 0), (2, 1), (4, 2)):
        ids = np.flatnonzero(code == c)
        a, b, z = np.roll(f[ids], -rotation, axis=1).T
        m = mf[ids, rotation]
        add(ids, np.stack((np.column_stack((a, m, z)), np.column_stack((m, b, z))), axis=1))
    for c, rotation in ((3, 0), (6, 1), (5, 2)):
        ids = np.flatnonzero(code == c)
        x, y, z = np.roll(f[ids], -rotation, axis=1).T
        xy, yz = mf[ids, rotation], mf[ids, (rotation+1)%3]
        # Choose the shorter diagonal of the remaining quadrilateral.
        first = np.linalg.norm(nv[x]-nv[yz], axis=1) <= np.linalg.norm(nv[xy]-nv[z], axis=1)
        for take in (first, ~first):
            ii = ids[take]; a, b, d, m, n = x[take], y[take], z[take], xy[take], yz[take]
            tip = np.column_stack((b, n, m))
            if take is first:
                t1, t2 = np.column_stack((a, m, n)), np.column_stack((a, n, d))
            else:
                t1, t2 = np.column_stack((a, m, d)), np.column_stack((m, n, d))
            add(ii, np.stack((tip, t1, t2), axis=1))
    ids = np.flatnonzero(code == 7)
    a, b, c = f[ids].T
    ab, bc, ca = mf[ids].T
    add(ids, np.stack((np.column_stack((a, ab, ca)), np.column_stack((ab, b, bc)),
                       np.column_stack((ca, bc, c)), np.column_stack((ab, bc, ca))), axis=1))
    return SurfaceMesh(nv, np.vstack(output), dict(mesh.report)), np.concatenate(parents), dict(
        marked_edges=int(marked.sum()), face_split_histogram=np.bincount(counts, minlength=4).tolist())


def local_selection(mesh, vertex_values, center_values, presence, cfg):
    p = mesh.vertices[mesh.triangles]
    length = np.linalg.norm(p-p[:, [1, 2, 0]], axis=2).max(axis=1)
    near = np.zeros(len(length), bool)
    for channel, rho, exists in ((1, cfg.rho_edge, presence[0]), (2, cfg.rho_vertex, presence[1])):
        if exists == 'present':
            minimum = vertex_values[mesh.triangles, channel].min(axis=1)
            near |= (minimum-length <= rho) | (center_values[:, channel] <= rho)
    return near & (length > cfg.target_edge*(1+1e-10)), near, length


def densify_surface_mesh(mesh, provider, cfg, progress=None):
    """Return mesh, original parent-triangle IDs, and a complete refinement report."""
    require(cfg.method in ('none', 'uniform', 'local'), 'refinement_method', 'Unknown refinement method')
    require(cfg.max_rounds > 0 and cfg.target_edge > 0, 'refinement_config', 'Invalid refinement settings')
    started = time.monotonic()
    original = mesh_metrics(mesh)
    require(len(mesh.triangles) <= cfg.max_triangles, 'mesh_resource_limit', 'Original mesh exceeds budget')
    parent = np.arange(len(mesh.triangles), dtype=np.int64)
    history = []
    vertex_values = None
    rounds = 0 if cfg.method == 'none' else 1 if cfg.method == 'uniform' else cfg.max_rounds+1
    for level in range(rounds):
        if progress: progress('refine_select_'+str(level))
        edges, inv, _ = mesh_edges(mesh.triangles)
        if cfg.method == 'uniform':
            selected = np.ones(len(mesh.triangles), bool)
        else:
            if vertex_values is None:
                vertex_values = evaluate_batched(provider, mesh.vertices, cfg.query_batch)
            centers = mesh.vertices[mesh.triangles].mean(axis=1)
            cv = evaluate_batched(provider, centers, cfg.query_batch)
            selected, near, length = local_selection(mesh, vertex_values, cv, provider.presence(), cfg)
            if not selected.any():
                history.append(dict(level=level, triangles=len(mesh.triangles), selected=0,
                                    local_max_edge=float(length[near].max()) if near.any() else 0.))
                break
            require(level < cfg.max_rounds, 'refinement_target_unmet', 'Local target remains unmet after maximum rounds',
                    remaining=int(selected.sum()), target=cfg.target_edge, history=history)
        marked = np.zeros(len(edges), bool)
        marked[inv[selected].ravel()] = True
        old_vertices = len(mesh.vertices)
        refined, parent_local, info = split_edges(mesh, marked, cfg.max_triangles)
        parent = parent[parent_local]
        history.append(dict(level=level, before=len(mesh.triangles), after=len(refined.triangles),
                            selected=int(selected.sum()), transition_faces=int((marked[inv].any(1)&~selected).sum()), **info))
        mesh = refined
        if vertex_values is not None:
            new_values = evaluate_batched(provider, mesh.vertices[old_vertices:], cfg.query_batch)
            vertex_values = np.vstack((vertex_values, new_values))
        if progress: progress('refine_finished_'+str(level))
    final = mesh_metrics(mesh)
    require((original['components'], original['euler']) == (final['components'], final['euler']),
            'refinement_topology_changed', 'Linear refinement changed mesh topology', original=original, refined=final)
    for key in ('area', 'volume'):
        require(abs(final[key]-original[key]) <= 1e-10+abs(original[key])*1e-9,
                'refinement_geometry_changed', 'Linear refinement changed area/volume', metric=key)
    report = dict(config=asdict(cfg), original=original, refined=final, history=history,
                  elapsed_seconds=time.monotonic()-started, geometry='unchanged_piecewise_linear_surface')
    mesh.report.update(refinement=report)
    return mesh, parent, report
