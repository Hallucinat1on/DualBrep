"""Recover vertex-edge topology and fit curves from 3D voxel thinning."""
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .curve_geometry import closest_parameters, projected_seed
from .curve_fitting import CurveRecord, fit_circle, fit_fixed_arc
from .deadline import checkpoint
from .edge_first_grid import sample_local_tube
from .schemas import EdgeInstance, LogicalGraph, ReconstructionError, VertexNode, require
from .shell_geometry import shell_vertices
from .skeleton_paths import (
    Path,
    recover_cut_ball_short_paths,
    refine_voxel_centers,
    trace_paths,
)
from .voxel_skeleton import thin_volume


def add_short_paths(paths, vertices, grid, cfg):
    """Keep short segments if cached off-axis E observations support them."""
    if len(vertices) < 2:
        return
    h = cfg.fine_spacing
    positions = np.array([v.position for v in vertices])
    connected = {tuple(sorted(p.nodes)) for p in paths if len(p.nodes) == 2}
    valid = grid.valid[:, 1] & (grid.values[:, 1] <= cfg.shell_radius)
    if not valid.any():
        return
    q, d = grid.points[valid], grid.values[valid, 1]
    tree = cKDTree(q)
    for a, b in sorted(cKDTree(positions).query_pairs(5 * h)):
        if (a, b) in connected:
            continue
        p0, p1 = positions[[a, b]]
        axis = p1 - p0
        length = np.linalg.norm(axis)
        if length <= 1e-10:
            continue
        ids = tree.query_ball_point((p0 + p1) * .5, .5 * length + h)
        if not ids:
            continue
        points = q[ids]
        t = (points - p0) @ axis / (axis @ axis)
        radial = np.linalg.norm(points - p0 - t[:, None] * axis, axis=1)
        keep = (t > .12) & (t < .88) & (radial <= h)
        if keep.sum() < 4 or np.ptp(t[keep]) < .35:
            continue
        residual = radial[keep] - d[ids][keep]
        bias = np.median(residual)
        if abs(bias) > .5 * h or np.quantile(abs(residual - bias), .9) > .25 * h:
            continue
        occupied = p0 + np.linspace(.2, .8, 7)[:, None] * axis
        if any(cKDTree(p.points).query(occupied)[0].max() < .3 * length for p in paths):
            continue
        paths.append(Path([a, b], np.vstack((p0, occupied, p1)), False))
        connected.add((a, b))


def _path_near_surface(path, provider, limit):
    if provider is None or not hasattr(provider, 'oracle') or not len(path.points):
        return True
    sq = provider.oracle.closest(path.points)[0]
    return float(np.sqrt(sq.max())) <= limit


def _resolve_local_junctions(unresolved, vertices, provider, cfg, existing_paths):
    """Re-thin unresolved merged-tube components in a sparse local tube at finer spacing."""
    if not unresolved or provider is None or cfg.local_refine_levels <= 0:
        for item in unresolved:
            item.pop('_support_points', None)
            item.pop('_junction_points', None)
        return [], unresolved
    resolved_paths = []
    remaining = []
    existing_pairs = {tuple(sorted(p.nodes)) for p in existing_paths if len(p.nodes) == 2}
    v_positions = np.asarray([v.position for v in vertices]).reshape(-1, 3)

    for item in unresolved:
        support_pts = item.pop('_support_points', None)
        if item.get('reason') != 'skeleton_junction_unresolved' or support_pts is None or not len(support_pts):
            remaining.append(item)
            continue
        lo, hi = np.asarray(item['bbox'][0], float), np.asarray(item['bbox'][1], float)
        pad = max(3 * cfg.fine_spacing, cfg.shell_radius * 2)
        local_v_ids = (np.flatnonzero(np.all((v_positions >= lo - pad) & (v_positions <= hi + pad), axis=1)).tolist()
                       if len(v_positions) else [])
        tube_centers = (np.vstack((support_pts, v_positions[local_v_ids]))
                        if local_v_ids else support_pts)
        success = False
        best_partial = None
        for lvl in range(1, cfg.local_refine_levels + 1):
            checkpoint('local_junction_refine')
            h_sub = cfg.fine_spacing / (2 ** lvl)
            r_sub = max(1.35 * h_sub, cfg.shell_radius / (2 ** lvl))
            tube_radius = max(2.2 * cfg.shell_radius, 3.5 * r_sub) if lvl == 1 else max(1.6 * cfg.shell_radius, 3.5 * r_sub)
            local_grid = sample_local_tube(provider, tube_centers, tube_radius, h_sub, cfg)
            if not len(local_grid.points):
                break
            local_pos = v_positions[local_v_ids] if local_v_ids else np.empty((0, 3))
            if len(local_pos) > 1:
                lgaps = cKDTree(local_pos).query(local_pos, k=2)[0][:, 1]
            else:
                lgaps = np.full(len(local_pos), np.inf)
            local_verts = [VertexNode(i, vertices[gid].position, vertices[gid].residual,
                                      min(2 * h_sub, 0.4 * float(lgaps[i])))
                           for i, gid in enumerate(local_v_ids)]
            l_pts, l_links, _ = thin_volume(local_grid, local_verts, cfg, spacing=h_sub, level=r_sub)
            l_paths, l_unres = trace_paths(
                l_pts, l_links, local_verts, shell_radius=r_sub, spacing=h_sub,
                allow_collapse=cfg.allow_feature_collapse, collapse_tolerance=cfg.collapse_tolerance,
            )
            l_short = recover_cut_ball_short_paths(l_pts, l_links, local_verts, l_paths, h_sub)
            l_paths.extend(l_short)
            for u in l_unres:
                u.pop('_support_points', None)
                u.pop('_junction_points', None)
            # Filter out cross-space bogus jumps before accepting any local paths.
            l_paths = [lp for lp in l_paths if _path_near_surface(lp, provider, cfg.feature_sdf_limit)]
            if not l_unres and l_paths:
                refine_voxel_centers(l_paths, local_grid, cfg, spacing=h_sub, level=r_sub)
                for lp in l_paths:
                    global_nodes = [local_v_ids[i] for i in lp.nodes]
                    key = tuple(sorted(global_nodes)) if len(global_nodes) == 2 else None
                    if key is not None and key in existing_pairs:
                        continue
                    lp.nodes = global_nodes
                    resolved_paths.append(lp)
                    if key is not None:
                        existing_pairs.add(key)
                success = True
                break
            if l_paths and (best_partial is None or len(l_unres) < len(best_partial[1])):
                best_partial = (l_paths, l_unres, local_grid, h_sub, r_sub)
        if not success:
            if best_partial is not None:
                l_paths, l_unres, local_grid, h_sub, r_sub = best_partial
                refine_voxel_centers(l_paths, local_grid, cfg, spacing=h_sub, level=r_sub)
                for lp in l_paths:
                    global_nodes = [local_v_ids[i] for i in lp.nodes]
                    key = tuple(sorted(global_nodes)) if len(global_nodes) == 2 else None
                    if key is not None and key in existing_pairs:
                        continue
                    lp.nodes = global_nodes
                    resolved_paths.append(lp)
                    if key is not None:
                        existing_pairs.add(key)
                for u in l_unres:
                    u['vertices'] = [local_v_ids[i] for i in u.get('vertices', [])]
                    remaining.append(u)
            else:
                remaining.append(item)
    return resolved_paths, remaining


def _collapse_close_vertices(vertices, paths, cfg):
    """Contract very short connected vertex pairs and merge near-duplicate parallel edges."""
    if not cfg.allow_feature_collapse or len(vertices) < 2:
        return vertices, paths, []
    tol = cfg.collapse_tolerance
    positions = np.asarray([v.position for v in vertices])
    parent = list(range(len(vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    collapsed = []
    for p in paths:
        if len(p.nodes) == 2 and p.nodes[0] != p.nodes[1]:
            a, b = find(p.nodes[0]), find(p.nodes[1])
            seg_len = float(np.linalg.norm(np.diff(p.points, axis=0), axis=1).sum()) if len(p.points) >= 2 else 0.0
            if a != b and np.linalg.norm(positions[a] - positions[b]) <= tol and seg_len <= 1.6 * tol:
                keep, drop = min(a, b), max(a, b)
                parent[drop] = keep
                collapsed.append(dict(kept=keep, merged=drop, distance=float(np.linalg.norm(positions[a] - positions[b]))))
    if collapsed:
        groups = {}
        for i in range(len(vertices)):
            groups.setdefault(find(i), []).append(i)
        remap = {}
        new_vertices = []
        collapsed_roots = {find(c['kept']) for c in collapsed}
        collapsed_new_v = set()
        for root in sorted(groups):
            members = groups[root]
            nid = len(new_vertices)
            for m in members:
                remap[m] = nid
            if root in collapsed_roots:
                collapsed_new_v.add(nid)
            pos = np.mean([vertices[m].position for m in members], axis=0)
            res = float(max(vertices[m].residual for m in members))
            rad = float(max(vertices[m].support_radius for m in members))
            new_vertices.append(VertexNode(nid, pos, res, rad))
        vertices = new_vertices
        remapped_paths = []
        for p in paths:
            mapped = [remap[i] for i in p.nodes]
            if len(mapped) == 2 and mapped[0] == mapped[1]:
                if np.linalg.norm(np.diff(p.points, axis=0), axis=1).sum() <= 2.5 * tol:
                    continue
            if any(v in collapsed_new_v for v in mapped):
                p.collapsed = True
            p.nodes = mapped
            remapped_paths.append(p)
        paths = remapped_paths

    # Merge near-duplicate parallel paths sharing the same endpoints within collapse_tolerance.
    kept_paths = []
    by_ends = {}
    for p in paths:
        if len(p.nodes) == 2 and p.nodes[0] != p.nodes[1]:
            key = (min(p.nodes), max(p.nodes))
            merged = False
            for prev in by_ends.get(key, []):
                if len(prev.points) >= 2 and len(p.points) >= 2:
                    d1 = cKDTree(prev.points).query(p.points)[0].max()
                    d2 = cKDTree(p.points).query(prev.points)[0].max()
                    if max(d1, d2) <= tol:
                        prev.collapsed = True
                        merged = True
                        break
            if merged:
                continue
            by_ends.setdefault(key, []).append(p)
        kept_paths.append(p)
    return vertices, kept_paths, collapsed


def with_endpoints(points, ends, closed, mask=None):
    n_orig = len(points)
    m = np.ones(n_orig, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if len(m) != n_orig:
        m = np.ones(n_orig, dtype=bool)
    if len(ends) == 1:
        ends = np.repeat(ends, 2, axis=0)
    if len(ends) == 2:
        points = np.vstack((ends[0], points, ends[1]))
        m = np.r_[True, m, True]
    elif closed and np.linalg.norm(points[0] - points[-1]) > 1e-10:
        points = np.vstack((points, points[0]))
        m = np.r_[m, m[0]]
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]
    return points[keep], ends, m[keep]


def _curve_accepted(errors, refined_mask, tol, h, analytic=False):
    if errors.max() <= tol:
        return True
    ref_ok = refined_mask.sum() >= 6
    if ref_ok:
        ref_err = errors[refined_mask]
        unref_err = errors[~refined_mask]
        ref_pass = (ref_err.max() <= tol) or (len(ref_err) >= 20 and np.quantile(ref_err, 0.95) <= tol and ref_err.max() <= 1.5 * tol)
        unref_pass = (not len(unref_err)) or (unref_err.max() <= max(tol, 0.75 * h))
        return bool(ref_pass and unref_pass)
    # All raw unrefined lattice points on an analytic primitive: lattice quantization
    # error is up to ~0.5*sqrt(2)*h even when the least-squares primitive is exact.
    if analytic and len(errors) >= 12:
        return bool(np.quantile(errors, 0.95) <= max(tol, 0.6 * h) and errors.max() <= max(tol, 0.75 * h))
    return False


def _densify_chord_gaps(points, max_gap):
    """Linearly subdivide large gaps so cubic Bezier splines cannot overshoot empty spans."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        d = float(np.linalg.norm(b - a))
        if d > max_gap:
            steps = int(np.ceil(d / max_gap))
            for k in range(1, steps):
                out.append(a + (k / steps) * (b - a))
        out.append(b)
    return np.asarray(out)


def _curve_near_surface(record, provider, limit):
    if provider is None or not hasattr(provider, 'oracle'):
        return True
    pts = record.evaluate(np.linspace(0, 1, 65))
    return float(np.sqrt(provider.oracle.closest(pts)[0].max())) <= limit


def fit_geometry(edge_id, path, ends, cfg, *, provider=None):
    """Line/circle/C1 Bézier with fixed endpoints, refinement-aware acceptance, and overshoot guard."""
    points, ends, mask = with_endpoints(path.points, ends, path.closed, path.refined_mask)
    tol = cfg.curve_tolerance
    h = cfg.fine_spacing
    fit_pts = points[mask] if mask.sum() >= 6 else points

    if len(ends) == 2 and np.linalg.norm(ends[1] - ends[0]) > 1e-10:
        line = CurveRecord(edge_id, 'line', dict(p0=ends[0], p1=ends[1]), 0.)
        errors = closest_parameters(line, points)[1]
        if _curve_accepted(errors, mask, tol, h, analytic=True):
            line.residual = float(errors[mask].max() if mask.sum() >= 4 else errors.max())
            return line
    if len(points) >= 6:
        data, _ = fit_circle(fit_pts)
        if path.closed and abs(data['sweep']) > 5.5:
            data['sweep'] = np.sign(data['sweep']) * 2 * np.pi
            cand = CurveRecord(edge_id, 'circle', data, 0.)
            errors = closest_parameters(cand, points)[1]
            if _curve_accepted(errors, mask, tol, h, analytic=True):
                cand.residual = float(errors[mask].max() if mask.sum() >= 6 else errors.max())
                return cand
        if len(ends) == 2 and np.linalg.norm(ends[1] - ends[0]) > 1e-10:
            candidate = fit_fixed_arc(fit_pts, *ends)
            if candidate:
                cand = CurveRecord(edge_id, 'circle', candidate[0], 0.)
                errors = closest_parameters(cand, points)[1]
                if _curve_accepted(errors, mask, tol, h, analytic=True):
                    cand.residual = float(errors[mask].max() if mask.sum() >= 6 else errors.max())
                    return cand
    periodic = path.closed or (len(ends) == 2 and np.linalg.norm(ends[1] - ends[0]) <= 1e-10)
    dense_pts = _densify_chord_gaps(points, 2.0 * h)
    for fit_source in (points, dense_pts):
        for spans in range(4 if periodic else 1, cfg.max_segments + 1):
            if 3 * spans + 1 > 2 * len(fit_source) + 4:
                break
            checkpoint('geometry_fit')
            breaks = np.linspace(0, 1, spans + 1)
            record = projected_seed(fit_source, ends, breaks, edge_id, periodic)
            errors = closest_parameters(record, points)[1]
            if _curve_accepted(errors, mask, tol, h, analytic=False):
                if _curve_near_surface(record, provider, cfg.feature_sdf_limit):
                    record.residual = float(errors[mask].max() if mask.sum() >= 6 else errors.max())
                    return record
                # Overshot the surface on a sparse gap; retry with chord-densified points.
                break
    require(False, 'curve_geometry_failed', 'Skeleton centers do not support a curve within tolerance', edge=edge_id)


def _promote_unresolved_junctions(cloud, links, vertices, unresolved, grid, cfg):
    """Promote remaining deg>2 junction clusters to vertices (or absorb into nearby vertices)."""
    if not cfg.allow_feature_collapse or not unresolved or not len(links):
        return vertices, [], unresolved, 0
    j_pts = []
    for u in unresolved:
        jp = u.get('_junction_points')
        if u.get('reason') == 'skeleton_junction_unresolved' and jp is not None and len(jp):
            j_pts.append(np.asarray(jp, float).reshape(-1, 3))
    if not j_pts:
        return vertices, [], unresolved, 0
    all_j = np.vstack(j_pts)
    tol = max(cfg.collapse_tolerance, 2.0 * cfg.fine_spacing)
    # Greedy cluster nearby deg>2 nodes.
    clusters = []
    for p in all_j:
        placed = False
        for c in clusters:
            if np.linalg.norm(p - c[0]) <= tol:
                c[1].append(p)
                c[0] = np.mean(c[1], axis=0)
                placed = True
                break
        if not placed:
            clusters.append([p.copy(), [p]])

    new_verts = [VertexNode(v.id, v.position.copy(), v.residual, v.support_radius) for v in vertices]
    v_pos = np.asarray([v.position for v in new_verts]).reshape(-1, 3)
    promoted_ids = set()
    changed = False
    for center, members in clusters:
        if len(v_pos):
            dists = np.linalg.norm(v_pos - center, axis=1)
            near_id = int(np.argmin(dists))
            if dists[near_id] <= 1.5 * tol:
                # Absorb nearby junction into existing vertex by expanding its cut ball safely.
                other = np.delete(dists, near_id)
                max_rad = 0.45 * float(other.min()) if len(other) else 3.0 * cfg.shell_radius
                needed = float(np.linalg.norm(np.asarray(members) - v_pos[near_id], axis=1).max()) + 0.6 * cfg.fine_spacing
                if needed <= max_rad and needed > new_verts[near_id].support_radius:
                    new_verts[near_id].support_radius = needed
                    changed = True
                continue
        nid = len(new_verts)
        gap = float(np.linalg.norm(v_pos - center, axis=1).min()) if len(v_pos) else np.inf
        rad = min(2.0 * cfg.fine_spacing, 0.4 * gap)
        new_verts.append(VertexNode(nid, center, cfg.fine_spacing * 0.5, rad))
        v_pos = np.asarray([v.position for v in new_verts]).reshape(-1, 3)
        promoted_ids.add(nid)
        changed = True

    if not changed:
        return vertices, [], unresolved, 0
    new_paths, new_unres = trace_paths(
        cloud, links, new_verts,
        shell_radius=cfg.shell_radius, spacing=cfg.fine_spacing,
        allow_collapse=True, collapse_tolerance=cfg.collapse_tolerance,
    )
    new_paths.extend(recover_cut_ball_short_paths(cloud, links, new_verts, new_paths, cfg.fine_spacing))
    add_short_paths(new_paths, new_verts, grid, cfg)
    refine_voxel_centers(new_paths, grid, cfg)
    for p in new_paths:
        if any(v in promoted_ids for v in p.nodes):
            p.collapsed = True
    return new_verts, new_paths, new_unres, len(promoted_ids)


def recover_wireframe(grid, provider, cfg, v_shell, v_faces):
    started = perf_counter()
    vertices, rejected = shell_vertices(v_shell, v_faces, grid, cfg, provider=provider)
    if provider.presence()[1] == 'present' and not vertices:
        rejected.append(dict(reason='vertex_shell_empty'))
    vertices_done = perf_counter()
    warnings = []

    cloud, links, statistics = thin_volume(grid, vertices, cfg)
    paths, unresolved = trace_paths(
        cloud, links, vertices,
        shell_radius=cfg.shell_radius, spacing=cfg.fine_spacing,
        allow_collapse=cfg.allow_feature_collapse, collapse_tolerance=cfg.collapse_tolerance,
    )
    short_from_skel = recover_cut_ball_short_paths(cloud, links, vertices, paths, cfg.fine_spacing)
    paths.extend(short_from_skel)
    add_short_paths(paths, vertices, grid, cfg)
    refine_voxel_centers(paths, grid, cfg)

    local_paths, unresolved = _resolve_local_junctions(unresolved, vertices, provider, cfg, paths)
    paths.extend(local_paths)

    promoted_count = 0
    if cfg.allow_feature_collapse and any(u.get('reason') == 'skeleton_junction_unresolved' for u in unresolved):
        p_verts, p_paths, p_unres, promoted_count = _promote_unresolved_junctions(
            cloud, links, vertices, unresolved, grid, cfg)
        if p_paths and len(p_paths) > len(paths):
            vertices, paths, unresolved = p_verts, p_paths, p_unres

    for u in unresolved:
        u.pop('_support_points', None)
        u.pop('_junction_points', None)

    vertices, paths, collapsed = _collapse_close_vertices(vertices, paths, cfg)

    statistics['paths'] = [dict(vertices=path.nodes, closed=path.closed, ports=path.ports,
                                samples=len(path.points)) for path in paths]
    statistics['local_resolved_paths'] = len(local_paths)
    statistics['promoted_vertices'] = promoted_count
    statistics['collapsed_vertices'] = collapsed
    skeleton = cloud, links, statistics
    if statistics['unknown_interface']:
        warnings.append(dict(reason='skeleton_unknown_interface',
                             points=statistics['unknown_interface']))
    if len(cloud) and not len(links):
        unresolved.append(dict(reason='skeleton_no_links', points=len(cloud)))
    extraction_done = perf_counter()

    rejected.extend(unresolved)
    edges, curves, ports, collapsed_edges = [], {}, {}, []
    for path in paths:
        eid = len(edges)
        ends = np.array([vertices[v].position for v in path.nodes]).reshape(-1, 3)
        try:
            curve = fit_geometry(eid, path, ends, cfg, provider=provider)
        except (ReconstructionError, np.linalg.LinAlgError) as exc:
            rejected.append(dict(reason=getattr(exc, 'code', 'curve_solve_failed'), vertices=path.nodes))
            continue
        ids = path.nodes if len(path.nodes) != 1 else path.nodes * 2
        edges.append(EdgeInstance(eid, (), path.points, path.closed, ids, -1, cfg.support_radius))
        curves[eid] = curve
        if path.collapsed:
            collapsed_edges.append(eid)
        if path.ports:
            ports[eid] = path.ports
    used = {v for e in edges for v in e.vertices}
    rejected.extend(dict(reason='isolated_vertex', vertex=v.id) for v in vertices if v.id not in used)
    miss = 0
    if curves:
        samples = []
        for curve in curves.values():
            length = np.linalg.norm(np.diff(curve.evaluate(np.linspace(0, 1, 129)), axis=0), axis=1).sum()
            samples.append(curve.evaluate(np.linspace(0, 1, max(129, int(2 * length / cfg.fine_spacing) + 1))))
        tree = cKDTree(np.vstack(samples))
        miss = int((tree.query(cloud)[0] > 2 * cfg.fine_spacing).sum()) if len(cloud) else 0
        valid = grid.valid[:, 1] & (grid.values[:, 1] <= min(cfg.fine_spacing, cfg.shell_radius * .6))
        if valid.any():
            gap = tree.query(grid.points[valid])[0]
            miss += int((gap > grid.values[valid, 1] + 3 * cfg.fine_spacing).sum())
    report = dict(vertices=len(vertices), edges=len(edges), support_points=len(cloud),
                  unresolved=rejected, warnings=warnings, uncovered_support=miss,
                  sphere_ports=ports, collapsed_vertices=collapsed, collapsed_edges=collapsed_edges,
                  timings=dict(vertices=vertices_done - started,
                               centerline_extraction=extraction_done - vertices_done,
                               fit_and_coverage=perf_counter() - extraction_done),
                  skeleton=statistics)
    return LogicalGraph(0, vertices, edges, report={'wireframe': report}), curves, cloud, skeleton
