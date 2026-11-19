"""Coarse wireframe projection and planar fitting on the original triangles."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import Polygon, contains_xy

from .deadline import checkpoint
from .edge_first_embedding import label_barriers
from .schemas import ReconstructionError, require
from .surface_fitting import FacePatch, frame, sample_face_regions
from .surface_mesh import mesh_edges


def sample_path(points, spacing):
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    distances = np.linspace(0., lengths[-1], max(2, int(np.ceil(lengths[-1] / spacing)) + 1))
    return np.column_stack([np.interp(distances, lengths, axis) for axis in points.T])


def route_path(vertices, vertex_tree, neighbors, points, start, end, radius, spacing, edge):
    """Find a continuous path of existing mesh edges inside the curve corridor."""
    samples = sample_path(points, spacing)
    curve_tree = cKDTree(samples)
    stride = max(1, int(radius / spacing))
    probes = np.vstack((samples[::stride], samples[-1]))
    hits = vertex_tree.query_ball_point(probes, radius + stride * spacing / 2)
    candidates = np.unique(np.concatenate(hits).astype(np.int64))
    distances = curve_tree.query(vertices[candidates])[0]
    use = distances <= radius
    candidates, distances = candidates[use], distances[use]

    endpoints = np.searchsorted(candidates, [start, end])
    require(start != end and np.all(endpoints < len(candidates)) and
            np.array_equal(candidates[endpoints], [start, end]), 'coarse_projection_collapsed',
            'The projected endpoints have no distinct anchors inside the corridor', edge=edge)
    a, b = endpoints
    links = neighbors[candidates][:, candidates].tocoo()
    i, j = links.row, links.col
    delta = vertices[candidates]
    middle = curve_tree.query((delta[i] + delta[j]) / 2)[0]
    valid = middle <= radius
    # Short routes close to the curve are preferred over detours onto another sheet.
    lengths = np.linalg.norm(delta[i] - delta[j], axis=1)
    offset = (distances[i] + distances[j] + 2 * middle) / 4
    weights = lengths * (1 + (offset / spacing) ** 2)
    network = csr_matrix((weights[valid], (i[valid], j[valid])), shape=links.shape)
    cost, previous = dijkstra(network, indices=a, return_predecessors=True)
    require(np.isfinite(cost[b]), 'coarse_projection_disconnected',
            'No continuous path of mesh edges follows the curve', edge=edge)
    nodes = [b]
    while nodes[-1] != a:
        nodes.append(previous[nodes[-1]])
    route = candidates[nodes[::-1]]

    trace = sample_path(vertices[route], spacing)
    error = float(max(curve_tree.query(trace)[0].max(), cKDTree(trace).query(samples)[0].max()))
    require(error <= radius, 'coarse_projection_coverage',
            'Projected path does not cover the curve in both directions', edge=edge, error=error)
    return route, error


def coarse_curve_path(mesh, tree, neighbors, edge, curve, anchors, radius, spacing):
    if edge.vertices and edge.vertices[0] != edge.vertices[-1]:
        start, end = anchors[edge.vertices]
        points = curve.evaluate(np.linspace(0, 1, 129))
        path, error = route_path(mesh.vertices, tree, neighbors, points,
                                 start, end, radius, spacing, edge.id)
    else:
        # Route both halves so a closed edge cannot collapse to its starting node.
        first, halfway = tree.query(curve.evaluate(np.array([0., .5])))[1]
        start = anchors[edge.vertices[0]] if edge.vertices else first
        path1, error1 = route_path(mesh.vertices, tree, neighbors,
                                   curve.evaluate(np.linspace(0, .5, 129)),
                                   start, halfway, radius, spacing, edge.id)
        path2, error2 = route_path(mesh.vertices, tree, neighbors,
                                   curve.evaluate(np.linspace(.5, 1, 129)),
                                   halfway, start, radius, spacing, edge.id)
        path, error = np.r_[path1[:-1], path2], max(error1, error2)
    return path, error


def coarse_paths(mesh, graph, curves, radius, spacing, edge_index, *, allow_partial=False,
                 anchor_tolerance=None):
    """Route reliable boundaries on the original mesh edges."""
    a, b = edge_index[0].T
    rows, columns = np.r_[a, b], np.r_[b, a]
    size = len(mesh.vertices)
    neighbors = csr_matrix((np.ones(len(rows), dtype=bool), (rows, columns)), shape=(size, size))
    vertex_tree = cKDTree(mesh.vertices)
    spacing = min(spacing, radius) / 2
    positions = np.asarray([v.position for v in graph.vertices]).reshape(-1, 3)
    distance, anchors = vertex_tree.query(positions)
    _, inverse, counts = np.unique(anchors, return_inverse=True, return_counts=True)
    valid = (distance <= (radius if anchor_tolerance is None else anchor_tolerance)) & (counts[inverse] == 1)
    require(allow_partial or np.all(valid), 'coarse_projection_vertices',
            'Shared vertices have no distinct nearby mesh anchors')
    reserved = set(anchors[distance <= radius])
    claimed = set()
    barriers = {}
    paths, rejected = {}, {}
    offsets = []
    for edge in graph.edges:
        checkpoint('coarse_projection')
        try:
            require(not edge.vertices or np.all(valid[edge.vertices]), 'coarse_projection_vertices',
                    'Shared vertices have no distinct nearby mesh anchors', edge=edge.id)
            path, error = coarse_curve_path(mesh, vertex_tree, neighbors, edge, curves[edge.id],
                                            anchors, radius, spacing)
            interior = path[1:-1] if edge.vertices else path[:-1]
            interior_nodes = set(interior)
            require(len(interior_nodes) == len(interior)
                    and interior_nodes.isdisjoint(reserved) and interior_nodes.isdisjoint(claimed),
                    'coarse_projection_crossing',
                    'Projected boundaries meet away from shared endpoints', edge=edge.id)
            entries = {(min(u, v), max(u, v)): (edge.id, u, v) for u, v in zip(path, path[1:])}
            require(len(entries) == len(path) - 1 and entries.keys().isdisjoint(barriers),
                    'coarse_projection_overlap', 'Two boundary segments use the same mesh edge', edge=edge.id)
        except ReconstructionError as exc:
            if not allow_partial:
                raise
            rejected[edge.id] = exc.code
            continue
        barriers.update(entries)
        claimed.update(interior_nodes)
        paths[edge.id] = path
        offsets.append(error)
    return paths, barriers, dict(barrier_edges=len(barriers), max_offset=max(offsets, default=0.),
                                 rejected_edges=rejected)


def project_wireframe(mesh, graph, curves, cfg):
    """Project onto existing mesh edges and validate the induced face topology."""
    require(graph.edges, 'fast_face_no_edges', 'A boundary-free face needs a support model')
    edge_index = mesh_edges(mesh.triangles)
    _, barriers, report = coarse_paths(mesh, graph, curves, cfg.band_width, cfg.fine_spacing, edge_index)
    result = deepcopy(graph)
    labels = label_barriers(mesh.triangles, result, barriers, edge_index)
    require(all(result.cycles.values()), 'coarse_projection_missing_boundary',
            'A mesh component has no projected boundary')
    result.report['coarse_projection'] = report
    return result, labels


def face_loops(cycles, curves):
    loops = []
    for cycle in cycles:
        xyz = []
        for use in cycle:
            t = np.linspace(0, 1, 129) if use.forward else np.linspace(1, 0, 129)
            xyz.extend(curves[use.edge].evaluate(t)[:-1])
        loops.append(np.asarray(xyz))
    return loops


def face_polygon(patch, loops):
    uv = [patch.uv(loop) for loop in loops]
    largest = max(range(len(uv)), key=lambda i: Polygon(uv[i]).area)
    polygon = Polygon(uv[largest], [loop for i, loop in enumerate(uv) if i != largest])
    require(polygon.is_valid and polygon.area > 1e-12,
            'fast_face_invalid_loop', 'Planar outer/inner rings ambiguous', face=patch.id)
    return polygon


def fit_plane_ransac(points, normal, tolerance, rng):
    best = np.empty(0, int)
    for _ in range(64):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length <= 1e-12:
            continue
        n /= length
        if abs(n @ normal) < .9:
            continue
        ids = np.flatnonzero(abs((points - a) @ n) <= tolerance)
        if len(ids) > len(best):
            best = ids
    require(len(best) >= .85 * len(points), 'fast_face_plane_inliers', 'No dominant observed plane')
    p = points[best]
    center = p.mean(0)
    n = np.linalg.svd(p - center, full_matrices=False)[2][-1]
    return center, n * np.sign(n @ normal)


def recover_planar_faces(mesh, graph, curves, cfg):
    """Fit planes to points sampled from connected regions of the input mesh."""
    result, labels = project_wireframe(mesh, graph, curves, cfg)
    sample_cfg = replace(cfg, face_samples=min(256, max(48, cfg.face_samples // 4)))
    regions = sample_face_regions(mesh, labels, sample_cfg)
    triangles = mesh.vertices[mesh.triangles]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    region_areas = np.bincount(labels, weights=areas, minlength=result.face_count)

    patches = {}
    rng = np.random.default_rng(cfg.seed)
    for fid, (points, normals) in regions.items():
        checkpoint('fast_face_sampling')
        normal = normals.mean(axis=0)
        length = np.linalg.norm(normal)
        require(length > .9, 'fast_face_region_normals',
                'Projected region has inconsistent plane normals', face=fid)
        normal /= length
        x, y = frame(normal)
        patch = FacePatch(fid, 'plane', dict(origin=points.mean(0), x=x, y=y, axis=normal), normal, {})
        loops = face_loops(result.cycles[fid], curves)
        polygon = face_polygon(patch, loops)
        uv = patch.uv(points)
        aligned = normals @ normal >= .9
        core = aligned & contains_xy(polygon.buffer(-cfg.fine_spacing), *uv.T)
        if np.count_nonzero(core) < 12:
            core = aligned & contains_xy(polygon, *uv.T)
        require(np.count_nonzero(core) >= 12, 'fast_face_region_interior',
                'Too few observed samples inside the face loop', face=fid)

        center, n = fit_plane_ransac(points[core], normal, cfg.surface_tolerance, rng)
        x, y = frame(n)
        patch.data.update(origin=center, x=x, y=y, axis=n)
        patch.normal = n
        boundaries = np.vstack(loops)
        error = float(abs((boundaries - center) @ n).max())
        require(error <= cfg.boundary_tolerance, 'fast_face_boundary',
                'Fitted plane misses a fixed boundary', face=fid, error=error)
        polygon = face_polygon(patch, loops)

        uv = patch.uv(points)
        inliers = (abs((points - center) @ n) <= cfg.surface_tolerance) & (normals @ n >= .9)
        within = contains_xy(polygon.buffer(cfg.fine_spacing), *uv.T)
        supported = inliers & within
        # Reject regions that leak into adjacent faces or leave much of this face unsupported.
        area_ratio = region_areas[fid] / polygon.area
        require(supported.mean() >= .85 and .75 <= area_ratio <= 1.25, 'fast_face_region_coverage',
                'The projected region does not support the bounded plane', face=fid,
                support=float(supported.mean()), area_ratio=float(area_ratio))
        interior = points[inliers & contains_xy(polygon, *uv.T)]
        require(len(interior), 'fast_face_region_interior', 'No observed sample inside the face loop', face=fid)
        patch.data['orientation_point'] = interior[0]
        patch.report = dict(boundary_max_error=error, region_triangles=int(np.count_nonzero(labels == fid)),
                            mesh_samples=len(points), fit_samples=int(core.sum()),
                            region_support=float(supported.mean()),
                            region_area_ratio=float(area_ratio), method='coarse_projection')
        patches[fid] = patch
    return result, patches, labels
