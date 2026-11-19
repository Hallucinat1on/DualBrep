"""Recover a vertex/edge graph from scalar observations and ordered traces."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .curve_distance_fitting import closest_parameters, fit_edge_distances
from .curve_fitting import CurveRecord, fit_circle
from .deadline import checkpoint
from .edge_first_grid import query
from .edge_geometry import circle_order, order_general
from .schemas import EdgeInstance, LogicalGraph, ReconstructionError, VertexNode
from .surface_mesh import components
from .vertex_localization import distance_center


@dataclass
class Path:
    nodes: list
    points: np.ndarray
    closed: bool
    observations: tuple[np.ndarray, np.ndarray] | None = None


def unique_points(points, spacing):
    _, ids = np.unique(np.round(points / spacing).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(ids)]


def project_low_set(provider, points, channel, radius, cfg):
    """Project valid samples to a local low set without accepting band exits."""
    q = points.copy()
    tolerance = cfg.curve_tolerance * .2
    samples = np.empty((0, 3))
    for _ in range(5):
        values, valid = query(provider, q, cfg.query_batch_size)
        accepted = valid[:, channel] & (values[:, channel] <= tolerance)
        samples = np.vstack((samples, q[accepted]))
        active = valid[:, channel] & ~accepted
        q, d = q[active], values[active, channel]
        if not len(q):
            break
        gradient, good = provider.gradient(q, channel)
        norms = np.linalg.norm(gradient, axis=1)
        good &= norms > 1e-8
        q = q[good] - np.minimum(d[good], radius)[:, None] * gradient[good] / norms[good, None]
    values, valid = query(provider, q, cfg.query_batch_size)
    return np.vstack((samples, q[valid[:, channel] & (values[:, channel] <= tolerance)]))


def recover_vertices(grid, provider, cfg):
    if provider.presence()[1] == 'absent':
        return []
    mask = grid.valid[:, 2] & (grid.values[:, 2] < cfg.shell_radius)
    votes = project_low_set(provider, grid.points[mask], 2, cfg.shell_radius, cfg)
    if not len(votes):
        return []
    pairs = cKDTree(votes).query_pairs(.5 * cfg.localization_tolerance, output_type='ndarray')
    _, labels = components(pairs, len(votes))
    centers = np.array([np.median(votes[labels == i], axis=0) for i in np.unique(labels)])
    gap = (cKDTree(centers).query(centers, k=2)[0][:, 1]
           if len(centers) > 1 else np.full(len(centers), np.inf))
    seed_points, distances = grid.points[mask], grid.values[mask, 2]
    tree = cKDTree(seed_points)
    vertices = []
    for i, center in enumerate(centers):
        radius = min(2 * cfg.fine_spacing, .4 * gap[i])
        indices = tree.query_ball_point(center, max(radius, cfg.fine_spacing))
        if len(indices) >= 4:
            point, residual, rank = distance_center(seed_points[indices], distances[indices], center)
            if rank == 3 and residual < cfg.localization_tolerance and np.linalg.norm(point - center) < radius:
                center = point
        values, valid = query(provider, center[None], cfg.query_batch_size)
        if valid[0, 1:3].all() and values[0, 1:3].max() <= cfg.localization_tolerance:
            vertices.append(VertexNode(len(vertices), center, float(values[0, 2]), radius))
    return vertices


def local_tangents(points, tree, radius):
    tangents = []
    for point, neighbors in zip(points, tree.query_ball_point(points, radius)):
        local = points[neighbors] - point
        _, vec = np.linalg.eigh(local.T @ local)
        tangents.append(vec[:, -1])
    return np.asarray(tangents)


def trace_both_directions(points, ends, closed, cfg):
    """Predict along a tangent and correct using samples in its transverse neighborhood."""
    step = cfg.fine_spacing
    tree = cKDTree(points)
    tangents = local_tangents(points, tree, 2.5 * step)
    start = int(tree.query(np.mean(ends, axis=0))[1]) if len(ends) else 0
    seen = {start}

    def walk(sign):
        path, index = [], start
        direction = sign * tangents[start]
        for _ in range(len(points)):
            checkpoint('edge_first_trace')
            nearest = np.asarray(tree.query_ball_point(points[index], 2.5 * step), int)
            delta = points[nearest] - points[index]
            advance = delta @ direction
            perpendicular = np.linalg.norm(delta - advance[:, None] * direction, axis=1)
            candidates = ((advance > .2 * step) & (perpendicular <= .7 * step)
                          & (np.abs(tangents[nearest] @ direction) > .45)
                          & np.array([int(j) not in seen for j in nearest]))
            if not candidates.any():
                break
            choice = np.where(candidates, abs(advance - step) + 2 * perpendicular, np.inf).argmin()
            index = int(nearest[choice])
            seen.add(index)
            path.append(index)
            tangent = tangents[index]
            direction = tangent if tangent @ direction > 0 else -tangent
        return path

    forward = walk(1)
    reverse = walk(-1)
    path = points[reverse[::-1] + [start] + forward]
    if len(path) < 2:
        return None
    if cKDTree(path).query(points)[0].max() > 2 * step:
        return None
    if closed:
        if np.linalg.norm(path[0] - path[-1]) > 2.5 * step:
            return None
        return np.vstack((path, path[0]))
    if len(ends):
        if np.linalg.norm(path[0] - ends[0]) > np.linalg.norm(path[-1] - ends[0]):
            path = path[::-1]
        if np.max(np.linalg.norm(path[[0, -1]] - ends[[0, -1]], axis=1)) > 4 * step:
            return None
        return np.vstack((ends[0], path, ends[-1]))
    return path


def paths_from_cloud(cloud, vertices, cfg, method):
    """Cut junction neighborhoods before grouping branch support."""
    h = cfg.fine_spacing
    if not len(cloud):
        return [], []
    positions = np.asarray([v.position for v in vertices]).reshape(-1, 3)
    radii = np.array([v.support_radius for v in vertices])
    if len(vertices):
        distance, nearest = cKDTree(positions).query(cloud)
        cloud = cloud[distance > radii[nearest]]
    if not len(cloud):
        return [], []
    tree = cKDTree(cloud)
    pairs = tree.query_pairs(1.8 * h, output_type='ndarray')
    tangents = local_tangents(cloud, tree, 2.5 * h)
    d = cloud[pairs[:, 1]] - cloud[pairs[:, 0]]
    unit = d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-15)
    projected = np.abs((tangents[pairs] * unit[:, None, :]).sum(2))
    keep = (projected > .4).all(1) | (np.linalg.norm(d, axis=1) < .38 * h)
    _, labels = components(pairs[keep], len(cloud))
    paths, unresolved = [], []
    for label in np.unique(labels):
        points = cloud[labels == label]
        if len(points) < 3:
            continue  # Short-edge candidates explicitly cover tiny components.
        if len(vertices):
            gaps = cKDTree(points).query(positions)[0]
            ids = np.flatnonzero(gaps <= radii + 1.6 * h).tolist()
        else:
            ids = []
        if len(ids) > 2:
            unresolved.append(dict(reason='ambiguous_branch_endpoints', vertices=ids, points=len(points)))
            continue
        ends = positions[ids]
        closed = not ids
        if method == 'trace':
            path = trace_both_directions(points, ends, closed, cfg)
        else:
            try:
                circle, error = fit_circle(points)
                path = (circle_order(points, circle, ends, cfg) if error <= cfg.curve_tolerance
                        else order_general(points, ends, cfg))
            except (ReconstructionError, np.linalg.LinAlgError):
                path = None
        # One observed endpoint can denote a self-loop only if both ends
        # approach it, not an open branch with a missing second vertex.
        if path is not None and len(ids) == 1:
            local = points[np.linalg.norm(points - ends[0], axis=1) <= radii[ids[0]] + h]
            local_tree = cKDTree(local)
            if len(local) < 2 or len(local_tree.query_pairs(1.6 * h)) == len(local) * (len(local) - 1) // 2:
                path = None
        if path is None:
            unresolved.append(dict(reason='discontinuous_branch', vertices=ids, points=len(points)))
        else:
            paths.append(Path(ids, path, closed))
    return paths, unresolved


def add_short_paths(paths, vertices, provider, cfg):
    """Propose short chords and require independent side-distance evidence."""
    if len(vertices) < 2:
        return
    h = cfg.fine_spacing
    positions = np.asarray([v.position for v in vertices])
    connected = {tuple(sorted(p.nodes)) for p in paths if len(p.nodes) == 2}
    for a, b in sorted(cKDTree(positions).query_pairs(6 * h)):
        if (a, b) in connected:
            continue
        start, end = positions[[a, b]]
        axis = end - start
        length = np.linalg.norm(axis)
        if length <= 1e-10:
            continue
        _, _, basis = np.linalg.svd(axis[None], full_matrices=True)
        t = np.linspace(.15, .85, 7)
        middle = start + t[:, None] * axis
        samples = np.vstack((middle, *(middle + s * min(.2 * length, h) * v
                                      for v in basis[1:] for s in (-1, 1))))
        values, valid = query(provider, samples, cfg.query_batch_size)
        chord = CurveRecord(-1, 'line', dict(p0=start, p1=end), 0.)
        _, expected = closest_parameters(chord, samples)
        tol = min(cfg.curve_tolerance, .1 * length)
        evidence = valid[:, 1] & (np.abs(values[:, 1] - expected) <= tol)
        if not evidence[:7].all() or evidence[7:].sum() < 14:
            continue
        # A competing recovered branch must not already explain the chord.
        if any(cKDTree(p.points).query(middle)[0].max() < .4 * length for p in paths):
            continue
        paths.append(Path([a, b], np.vstack((start, middle, end)), False,
                          (samples[evidence], values[evidence, 1])))


def fit_paths(paths, vertices, grid, provider, cfg, unresolved):
    near = grid.valid[:, 1] & (grid.values[:, 1] <= min(cfg.shell_radius, cfg.support_radius))
    points, distances = grid.points[near], grid.values[near, 1]
    records, edges = {}, []
    if not paths:
        return LogicalGraph(0, vertices, edges), records
    support_gap, support_index = cKDTree(np.vstack([p.points for p in paths])).query(points)
    owners = np.repeat(np.arange(len(paths)), [len(p.points) for p in paths])[support_index]
    for i, path in enumerate(paths):
        selected = (owners == i) & (support_gap <= cfg.support_radius)
        # Keep only observations consistent with this branch being nearest.
        selected &= support_gap <= distances + 1.6 * cfg.fine_spacing
        ends = np.asarray([vertices[v].position for v in path.nodes]).reshape(-1, 3)
        for node in set(path.nodes):
            vertex = vertices[node]
            # Global E can name a competing branch even for points assigned
            # to this path by the nearest sampled point. Exclude the junction
            # from fit observations, but keep the endpoint as a hard constraint.
            radius = vertex.support_radius + cfg.fine_spacing
            selected &= np.linalg.norm(points - vertex.position, axis=1) > radius
        p, de = points[selected], distances[selected]
        if path.observations is not None:
            p, de = path.observations
        if len(p) < 3 and len(ends) == 2:
            p = path.points
            v, valid = query(provider, p, cfg.query_batch_size)
            p, de = p[valid[:, 1]], v[valid[:, 1], 1]
        eid = len(edges)
        try:
            record, _ = fit_edge_distances(eid, p, de, ends, path.closed, provider, cfg, path=path.points)
        except (ReconstructionError, np.linalg.LinAlgError) as exc:
            unresolved.append(dict(reason=getattr(exc, 'code', 'linear_solve_failed'), vertices=path.nodes,
                                   details=getattr(exc, 'details', {})))
            continue
        ids = path.nodes if len(path.nodes) != 1 else path.nodes * 2
        records[eid] = record
        edges.append(EdgeInstance(eid, (), path.points, path.closed, ids, -1, cfg.support_radius))
    return LogicalGraph(0, vertices, edges), records


def recover_wireframe(grid, provider, cfg, shell_points=None):
    started = perf_counter()
    vertices = recover_vertices(grid, provider, cfg)
    projected_start = perf_counter()
    near = grid.valid[:, 1] & (grid.values[:, 1] <= cfg.shell_radius)
    seeds = grid.points[near] if shell_points is None else shell_points
    if shell_points is not None:
        # Add sparse seeds only at the acquisition boundary where the shell
        # can be clipped. Interior skeleton samples come from the MC shell.
        outer = near & (np.abs(grid.values[:, 0]) > provider.band_width - cfg.shell_radius)
        seeds = np.vstack((seeds, grid.points[outer]))
    cloud = unique_points(project_low_set(provider, seeds, 1, cfg.shell_radius, cfg),
                          .25 * cfg.fine_spacing)
    paths_start = perf_counter()
    paths, unresolved = paths_from_cloud(cloud, vertices, cfg, cfg.extraction)
    add_short_paths(paths, vertices, provider, cfg)
    fit_start = perf_counter()
    graph, curves = fit_paths(paths, vertices, grid, provider, cfg, unresolved)
    coverage_start = perf_counter()
    used = {v for edge in graph.edges for v in edge.vertices}
    for vertex in vertices:
        if vertex.id not in used:
            unresolved.append(dict(reason='isolated_vertex', vertex=vertex.id))
    report = dict(vertices=len(vertices), edges=len(curves), seeds=len(seeds), skeleton_points=len(cloud),
                  unresolved=unresolved)
    if curves:
        samples = []
        for c in curves.values():
            length = np.linalg.norm(np.diff(c.evaluate(np.linspace(0, 1, 129)), axis=0), axis=1).sum()
            count = max(129, int(np.ceil(length / (.5 * cfg.fine_spacing))) + 1)
            samples.append(c.evaluate(np.linspace(0, 1, count)))
        tree = cKDTree(np.vstack(samples))
        report['uncovered_support'] = int((tree.query(cloud)[0] > 2 * cfg.fine_spacing).sum())
        # MC can miss a sub-grid shell entirely. Independently verify low-E
        # grid probes where their observed distance cannot be explained.
        idx = np.flatnonzero(grid.valid[:, 1] & (grid.values[:, 1] <= cfg.fine_spacing))
        gap = tree.query(grid.points[idx])[0]
        missed = idx[gap > grid.values[idx, 1] + cfg.fine_spacing]
        if len(missed):
            uncovered = project_low_set(provider, grid.points[missed], 1, cfg.shell_radius, cfg)
            report['uncovered_support'] += int((tree.query(uncovered)[0] > 2 * cfg.fine_spacing).sum())
    report['timings'] = dict(vertices=projected_start-started, project=paths_start-projected_start,
                             paths=fit_start-paths_start, fit=coverage_start-fit_start,
                             coverage=perf_counter()-coverage_start)
    graph.report['wireframe'] = report
    return graph, curves, cloud
