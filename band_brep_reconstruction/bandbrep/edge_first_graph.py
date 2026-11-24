"""Recover vertex-edge topology and fit curves from the 8D vector-field wireframe."""
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .curve_fitting import CurveRecord, fit_circle, fit_fixed_arc
from .curve_geometry import closest_parameters, projected_seed
from .deadline import checkpoint
from .schemas import EdgeInstance, LogicalGraph, ReconstructionError, require
from .vector_wireframe import Path, extract_vector_wireframe


def with_endpoints(points, ends, closed, refined_mask=None):
    mask = (np.ones(len(points), dtype=bool) if refined_mask is None
            else np.asarray(refined_mask, dtype=bool).copy())
    if closed or not len(ends):
        return points, np.empty((0, 3)), mask
    if len(ends) == 1:
        ends = np.vstack((ends[0], ends[0]))
    if len(points) >= 2:
        pts = points.copy()
        d_fwd = np.linalg.norm(pts[0] - ends[0]) + np.linalg.norm(pts[-1] - ends[1])
        d_rev = np.linalg.norm(pts[0] - ends[1]) + np.linalg.norm(pts[-1] - ends[0])
        if d_rev < d_fwd:
            pts = pts[::-1]
            mask = mask[::-1]
        if np.linalg.norm(pts[0] - ends[0]) > 1e-9:
            pts = np.vstack((ends[0], pts))
            mask = np.r_[True, mask]
        else:
            pts[0] = ends[0]
            mask[0] = True
        if np.linalg.norm(pts[-1] - ends[1]) > 1e-9:
            pts = np.vstack((pts, ends[1]))
            mask = np.r_[mask, True]
        else:
            pts[-1] = ends[1]
            mask[-1] = True
        return pts, ends, mask
    return ends, ends, np.ones(len(ends), dtype=bool)


def _curve_accepted(errors, refined_mask, tol, h, analytic=False):
    if errors.max() <= tol:
        return True
    ref_ok = refined_mask.sum() >= 6
    if ref_ok:
        ref_err = errors[refined_mask]
        unref_err = errors[~refined_mask]
        ref_pass = (ref_err.max() <= tol) or (
            len(ref_err) >= 20 and np.quantile(ref_err, 0.95) <= tol and ref_err.max() <= 1.5 * tol
        )
        unref_pass = (not len(unref_err)) or (unref_err.max() <= max(tol, 0.75 * h))
        return bool(ref_pass and unref_pass)
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
    """Line/circle/C1 Bezier with fixed endpoints, refinement-aware acceptance, and overshoot guard."""
    points, ends, mask = with_endpoints(path.points, ends, path.closed, path.refined_mask)
    tol = cfg.curve_tolerance
    h = cfg.fine_spacing
    fit_pts = points[mask] if mask.sum() >= 6 else points

    if len(ends) == 2 and np.linalg.norm(ends[1] - ends[0]) > 1e-10:
        line = CurveRecord(edge_id, 'line', dict(p0=ends[0], p1=ends[1]), 0.0)
        errors = closest_parameters(line, points)[1]
        if _curve_accepted(errors, mask, tol, h, analytic=True):
            line.residual = float(errors[mask].max() if mask.sum() >= 4 else errors.max())
            return line
    if len(points) >= 6:
        data, _ = fit_circle(fit_pts)
        if path.closed and abs(data['sweep']) > 5.5:
            data['sweep'] = np.sign(data['sweep']) * 2 * np.pi
            cand = CurveRecord(edge_id, 'circle', data, 0.0)
            errors = closest_parameters(cand, points)[1]
            if _curve_accepted(errors, mask, tol, h, analytic=True):
                cand.residual = float(errors[mask].max() if mask.sum() >= 6 else errors.max())
                return cand
        if len(ends) == 2 and np.linalg.norm(ends[1] - ends[0]) > 1e-10:
            candidate = fit_fixed_arc(fit_pts, *ends)
            if candidate:
                cand = CurveRecord(edge_id, 'circle', candidate[0], 0.0)
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
                break
    require(False, 'curve_geometry_failed', 'Extracted centerline does not support a curve within tolerance', edge=edge_id)


def recover_wireframe(grid, provider, cfg):
    """Extract vertices and curves from the 8D joint field and build the wireframe graph."""
    started = perf_counter()
    vertices, paths, rejected, cloud, statistics = extract_vector_wireframe(grid, provider, cfg)
    extraction_done = perf_counter()

    edges, curves, collapsed_edges = [], {}, []
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

    used = {v for e in edges for v in e.vertices}
    rejected.extend(dict(reason='isolated_vertex', vertex=v.id) for v in vertices if v.id not in used)

    miss = 0
    if curves:
        samples = []
        for curve in curves.values():
            length = np.linalg.norm(np.diff(curve.evaluate(np.linspace(0, 1, 129)), axis=0), axis=1).sum()
            samples.append(curve.evaluate(np.linspace(0, 1, max(129, int(2 * length / cfg.fine_spacing) + 1))))
        tree = cKDTree(np.vstack(samples))
        if len(cloud):
            miss = int((tree.query(cloud)[0] > 2 * cfg.fine_spacing).sum())

    report = dict(
        vertices=len(vertices),
        edges=len(edges),
        support_points=len(cloud),
        unresolved=rejected,
        warnings=[],
        uncovered_support=miss,
        collapsed_edges=collapsed_edges,
        timings=dict(
            vector_extraction=extraction_done - started,
            fit_and_coverage=perf_counter() - extraction_done,
        ),
        skeleton=statistics,
    )
    return LogicalGraph(0, vertices, edges, report={'wireframe': report}), curves, cloud
