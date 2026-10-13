import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
from OCC.Core.gp import gp_Pnt, gp_Vec

from .cad import PreprocessError, xyz


QUERY_SOURCES = {"uniform": 0, "near_surface": 1, "on_surface": 2,
                 "near_edge": 3, "near_vertex": 4, "edge_anchor": 5, "vertex_anchor": 6}


def allocate_counts(weights, total, minimum=0):
    weights = np.asarray(weights, dtype=np.float64)
    if not len(weights):
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Allocation needs nonnegative finite weights with positive total")
    total = max(total, minimum * len(weights))
    quota = weights / weights.sum() * (total - minimum * len(weights))
    result = np.floor(quota).astype(np.int64) + minimum
    remaining = total - int(result.sum())
    order = np.argsort(-(quota - np.floor(quota)), kind="stable")
    result[order[:remaining]] += 1
    return result


def sample_surface(mesh, cfg, rng):
    v, f, face_ids = mesh
    triangles = v[f]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    unique_ids = np.unique(face_ids)
    face_areas = [twice_area[face_ids == fid].sum() for fid in unique_ids]
    if min(face_areas) <= 0:
        raise PreprocessError("meshing_failed", "A face has no nondegenerate triangles")
    counts = allocate_counts(face_areas, cfg.surface_count, cfg.min_face_samples)
    chosen = []
    for fid, count in zip(unique_ids, counts):
        ids = np.flatnonzero((face_ids == fid) & (twice_area > 0))
        chosen.extend(rng.choice(ids, int(count), p=twice_area[ids] / twice_area[ids].sum()))
    chosen = np.asarray(chosen, dtype=np.int64)
    u, w = rng.random((2, len(chosen)))
    root = np.sqrt(u)
    bary = np.column_stack((1 - root, root * (1 - w), root * w))
    points = np.sum(triangles[chosen] * bary[:, :, None], axis=1)
    normals = cross[chosen] / twice_area[chosen, None]
    return np.column_stack((points, normals)).astype(np.float32), face_ids[chosen]


def sample_edges(targets, cfg):
    if not targets.edges:
        return np.empty((0, 6), np.float32), np.empty(0, np.int32), np.empty(0, bool), np.empty((0, 3))
    curves = [BRepAdaptor_Curve(edge) for edge in targets.edges]
    lengths = [GCPnts_AbscissaPoint.Length(c, cfg.solver_tolerance) for c in curves]
    if not np.isfinite(lengths).all() or min(lengths) <= 0:
        raise PreprocessError("missing_curve", "Target curve has invalid arc length")
    counts = allocate_counts(lengths, cfg.edge_count, cfg.min_edge_samples)
    points, ids, valid, anchors = [], [], [], []
    for tid, (curve, count) in enumerate(zip(curves, counts)):
        closed = targets.topology["edges"][tid]["closed"]
        requested = int(count) + int(closed)
        sampler = GCPnts_UniformAbscissa(curve, requested, cfg.solver_tolerance)
        if not sampler.IsDone() or sampler.NbPoints() != requested:
            raise PreprocessError("sampling_failed", f"Arc-length sampling failed for edge {tid}")
        parameters = [sampler.Parameter(i + 1) for i in range(int(count))]
        anchor_parameters = np.linspace(curve.FirstParameter(), curve.LastParameter(), cfg.zero_checks_per_edge,
                                        endpoint=not closed)
        anchors.extend(xyz(curve.Value(float(t))) for t in anchor_parameters)
        for parameter in parameters:
            point = gp_Pnt()
            tangent = gp_Vec()
            curve.D1(parameter, point, tangent)
            direction = xyz(tangent)
            magnitude = np.linalg.norm(direction)
            is_valid = bool(np.isfinite(direction).all() and magnitude > 1e-14)
            direction = direction / magnitude if is_valid else np.zeros(3)
            points.append(np.r_[xyz(point), direction])
            ids.append(tid)
            valid.append(is_valid)
    return (np.asarray(points, np.float32), np.asarray(ids, np.int32),
            np.asarray(valid, bool), np.asarray(anchors, np.float64).reshape(-1, 3))


def perturb(base, count, scales, rng):
    if not count:
        return np.empty((0, 3))
    selected = base[rng.integers(len(base), size=count), :3].astype(np.float64)
    sigma = rng.choice(scales, size=count)
    points = selected + rng.normal(size=(count, 3)) * sigma[:, None]
    outside = np.any(np.abs(points) > 1, axis=1)
    attempts = 0
    while outside.any():
        attempts += 1
        if attempts > 1000:
            raise PreprocessError("sampling_failed", "Noise scales produce excessive out-of-domain queries")
        points[outside] = selected[outside] + rng.normal(size=(int(outside.sum()), 3)) * sigma[outside, None]
        outside = np.any(np.abs(points) > 1, axis=1)
    return points


def sample_all(targets, mesh, cfg, rng):
    surface, face_id = sample_surface(mesh, cfg, rng)
    edge, edge_id, tangent_valid, edge_anchors = sample_edges(targets, cfg)
    weights = np.array(cfg.query_weights, dtype=np.float64)
    if not len(targets.vertices):
        weights[1] += weights[4]
        weights[4] = 0
    if not len(targets.edges):
        weights[1] += weights[3]
        weights[3] = 0
    counts = allocate_counts(weights, cfg.query_count)
    sources = [rng.uniform(-1, 1, (counts[0], 3)),
               perturb(surface, int(counts[1]), cfg.noise_scales, rng),
               surface[rng.integers(len(surface), size=counts[2]), :3],
               perturb(edge, int(counts[3]), cfg.noise_scales, rng),
               perturb(targets.vertices, int(counts[4]), cfg.noise_scales, rng),
               edge_anchors, targets.vertices]
    # Quantize BEFORE querying: labels correspond to the stored xyz positions.
    query = np.concatenate(sources).astype(np.float32)
    query_source = np.concatenate([np.full(len(p), i, np.uint8) for i, p in enumerate(sources)])
    order = rng.permutation(len(query))
    arrays = dict(surface_points=surface, surface_face_id=face_id,
                  edge_points=edge, edge_point_id=edge_id, edge_tangent_valid=tangent_valid,
                  vertex_points=targets.vertices.astype(np.float32),
                  query_points=query[order], query_source=query_source[order],
                  has_edges=np.array(bool(len(targets.edges))),
                  has_vertices=np.array(bool(len(targets.vertices))))
    return arrays
