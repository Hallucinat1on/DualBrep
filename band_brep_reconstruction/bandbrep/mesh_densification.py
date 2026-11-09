"""Conforming linear refinement of an existing surface; only scalar fields are queried."""
import time
from .deadline import checkpoint
import numpy as np
from .schemas import SurfaceMesh, require
from .surface_mesh import mesh_edges, face_adjacency, components, evaluate_batched
from .field_provider import CenterFields


def areas(mesh):
    p = mesh.vertices[mesh.triangles]
    return .5*np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)


def mesh_metrics(mesh):
    checkpoint('mesh_metrics')
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
    nc, _ = components(face_adjacency(f, (edges, inv, counts)), len(f))
    return dict(vertices=len(v), triangles=len(f), edges=len(edges), components=int(nc),
                euler=int(len(v)-len(edges)+len(f)), area=float(a.sum()), volume=volume,
                edge_quantiles=np.quantile(lengths, [.5, .95, .99, 1]).tolist(),
                area_quantiles=np.quantile(a, [.5, .95, .99, 1]).tolist())


def split_edges(mesh, marked, edge_index):
    """Split globally shared marked edges, triangulating every incident face (red/green)."""
    checkpoint("refinement_split")
    v, f = mesh.vertices, mesh.triangles
    edges, inv, _ = edge_index
    require(np.shape(marked) == (len(edges),), 'refinement_mask', 'Edge mask has wrong length')
    cuts = marked[inv]
    counts = cuts.sum(axis=1)
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
    # Necessary vertices lie on target edges; refine from E without relying on V.
    if presence[0] == 'present':
        minimum = vertex_values[mesh.triangles, 1].min(axis=1)
        near = ((minimum-length <= cfg.refinement_rho_edge)
                | (center_values[:, 1] <= cfg.refinement_rho_edge))
    return near & (length > cfg.refinement_target_edge*(1+1e-10)), near, length


def marked_long_edges(mesh, edge_index, selected, target):
    edges, inv, _ = edge_index
    marked = np.zeros(len(edges), bool)
    marked[inv[selected].ravel()] = True
    length = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    return marked & (length > target * (1 + 1e-10))


def refinement_report(mesh, original, history, settings, started):
    checkpoint('refinement_final_validation')
    final = mesh_metrics(mesh)
    require((original['components'], original['euler']) == (final['components'], final['euler']),
            'refinement_topology_changed', 'Linear refinement changed mesh topology')
    for key in ('area', 'volume'):
        require(abs(final[key] - original[key]) <= 1e-10 + abs(original[key]) * 1e-9,
                'refinement_geometry_changed', 'Linear refinement changed area/volume', metric=key)
    return dict(config=settings, original=original, refined=final, history=history,
                elapsed_seconds=time.monotonic() - started, geometry='unchanged_piecewise_linear_surface')


def emit_progress(progress, stage, event, mesh, history, started, **details):
    checkpoint(stage + '_' + event)
    if progress:
        progress(dict(stage=stage, event=event, level=len(history), vertices=len(mesh.vertices),
                      triangles=len(mesh.triangles), history=history,
                      elapsed_seconds=time.monotonic() - started, **details))


def coarse_refine_surface(mesh, target_edge, progress=None):
    """Geometry only: conformingly split long edges until the global target is met."""
    require(not isinstance(target_edge,bool) and np.isfinite(target_edge) and target_edge > 0, 'refinement_config', 'Invalid coarse edge target')
    started = time.monotonic()
    original = mesh_metrics(mesh)
    parent = np.arange(len(mesh.triangles), dtype=np.int64)
    history = []
    while True:
        emit_progress(progress, 'coarse_refinement', 'select', mesh, history, started)
        edge_index = mesh_edges(mesh.triangles)
        marked = marked_long_edges(mesh, edge_index, np.ones(len(mesh.triangles), bool), target_edge)
        emit_progress(progress, 'coarse_refinement', 'selected', mesh, history, started,
                      remaining=int(marked.sum()))
        if not marked.any():break
        before = len(mesh.triangles)
        mesh, parent_local, info = split_edges(mesh, marked, edge_index)
        parent = parent[parent_local]
        history.append(dict(level=len(history), before=before, after=len(mesh.triangles), **info))
        emit_progress(progress, 'coarse_refinement', 'split_finished', mesh, history, started)
    report = refinement_report(mesh, original, history, dict(target_edge=target_edge), started)
    report['target_reached'] = True
    return mesh, parent, report


def fine_refine_surface(mesh, provider, cfg, progress=None):
    """E-only near-field selection; split overlong edges, returning final queries."""
    cfg.validate()
    started = time.monotonic()
    original = mesh_metrics(mesh)
    parent = np.arange(len(mesh.triangles), dtype=np.int64)
    history = []
    vertex_values = None
    while True:
        emit_progress(progress, 'fine_refinement', 'select', mesh, history, started)
        edge_index = mesh_edges(mesh.triangles)
        if vertex_values is None:
            vertex_values = evaluate_batched(provider, mesh.vertices, cfg.query_batch_size)
        centers = mesh.vertices[mesh.triangles].mean(axis=1)
        values = evaluate_batched(provider, centers, cfg.query_batch_size)
        selected, near, length = local_selection(mesh, vertex_values, values, provider.presence(), cfg)
        local_max = float(length[near].max()) if near.any() else 0.
        emit_progress(progress, 'fine_refinement', 'selected', mesh, history, started,
                      remaining=int(selected.sum()), local_max_edge=local_max)
        if not selected.any():break
        marked = marked_long_edges(mesh, edge_index, selected, cfg.refinement_target_edge)
        old_vertices, before = len(mesh.vertices), len(mesh.triangles)
        transition = int((marked[edge_index[1]].any(1) & ~selected).sum())
        mesh, parent_local, info = split_edges(mesh, marked, edge_index)
        parent = parent[parent_local]
        history.append(dict(level=len(history), before=before, after=len(mesh.triangles),
                            selected=int(selected.sum()), transition_faces=transition, **info))
        emit_progress(progress, 'fine_refinement', 'split_finished', mesh, history, started)
        vertex_values = np.vstack((vertex_values,
            evaluate_batched(provider, mesh.vertices[old_vertices:], cfg.query_batch_size)))
    settings = dict(target_edge=cfg.refinement_target_edge, rho_edge=cfg.refinement_rho_edge,
                    selection='edge_udf_only', query_batch=cfg.query_batch_size,
                    pruning='long_edges_only')
    report = refinement_report(mesh, original, history, settings, started)
    report.update(target_reached=True, local_max_edge=local_max)
    return mesh, parent, report, CenterFields(centers, values, provider.fingerprint)
