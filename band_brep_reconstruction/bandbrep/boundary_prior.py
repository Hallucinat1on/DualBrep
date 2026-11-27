"""Learning-free boundary-first surface prior (BrepForge §4.2) for narrow or rounded faces.

Used as a fallback inside surface_fitting when boundary-eroded core samples are too sparse
(e.g., narrow chamfers or MC/Poisson-rounded strips).
"""
import numpy as np
from shapely import Polygon, contains_xy


def _orthonormal_basis(normal):
    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    x = np.cross(n, ref)
    x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(n, x)
    return x, y, n


def newell_normal(loop_pts):
    """Estimate robust surface normal from a closed 3D boundary loop via Newell's algorithm."""
    p = np.asarray(loop_pts, dtype=float)
    q = np.roll(p, -1, axis=0)
    nx = float(np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])))
    ny = float(np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])))
    nz = float(np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])))
    n = np.array([nx, ny, nz], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm > 1e-10:
        return n / norm
    c = p - p.mean(axis=0)
    _, evecs = np.linalg.eigh(c.T @ c)
    return evecs[:, 0]


def _ordered_loop_points(cycle, curves, samples_per_edge=32):
    parts = []
    for use in cycle:
        ts = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        if not use.forward:
            ts = 1.0 - ts
        parts.append(curves[use.edge].evaluate(ts))
    return np.vstack(parts)


def boundary_prior_samples(face_cycles, curves, cfg, *, ref_normals=None):
    """Synthesize clean interior (points, normals) from a face's outer 3D boundary loop.

    Constructs Newell normal -> 2D PCA local basis -> planar/quadratic surface fit over the
    outer loop, then samples interior grid points inside the 2D loop polygon (or medial chord
    for ultra-narrow strips).
    """
    if not face_cycles:
        return np.empty((0, 3)), np.empty((0, 3))

    outer = max(face_cycles, key=len)
    bnd = _ordered_loop_points(outer, curves, samples_per_edge=32)
    if len(bnd) < 3:
        return np.empty((0, 3)), np.empty((0, 3))

    origin = bnd.mean(axis=0)
    n_est = newell_normal(bnd)
    bx, by, n_axis = _orthonormal_basis(n_est)

    rel = bnd - origin
    uv_raw = np.column_stack((rel @ bx, rel @ by))
    cov2 = uv_raw.T @ uv_raw
    _, ev2 = np.linalg.eigh(cov2)
    bx = ev2[0, 1] * bx + ev2[1, 1] * by
    by = np.cross(n_axis, bx)

    u_b = rel @ bx
    v_b = rel @ by
    z_b = rel @ n_axis
    poly2d = Polygon(np.column_stack((u_b, v_b)))
    if not poly2d.is_valid:
        poly2d = poly2d.buffer(0.0)

    is_planar = float(np.abs(z_b).max()) <= cfg.boundary_tolerance
    scale = max(float(np.ptp(u_b)), float(np.ptp(v_b)), 1e-4)
    if is_planar:
        coef = np.zeros(6, dtype=float)
    else:
        un, vn = u_b / scale, v_b / scale
        A = np.column_stack((np.ones_like(un), un, vn, un * un, un * vn, vn * vn))
        reg = np.diag([0.0, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3])
        coef = np.linalg.lstsq(np.vstack((A, reg)), np.r_[z_b, np.zeros(6)], rcond=None)[0]

    def eval_prior(u_arr, v_arr):
        if is_planar:
            z_arr = np.zeros_like(u_arr)
            dz_du = np.zeros_like(u_arr)
            dz_dv = np.zeros_like(u_arr)
        else:
            un, vn = u_arr / scale, v_arr / scale
            z_arr = (coef[0] + coef[1] * un + coef[2] * vn
                     + coef[3] * un * un + coef[4] * un * vn + coef[5] * vn * vn)
            dz_du = (coef[1] + 2.0 * coef[3] * un + coef[4] * vn) / scale
            dz_dv = (coef[2] + coef[4] * un + 2.0 * coef[5] * vn) / scale
        p3 = origin + u_arr[:, None] * bx + v_arr[:, None] * by + z_arr[:, None] * n_axis
        n3 = n_axis[None, :] - dz_du[:, None] * bx - dz_dv[:, None] * by
        n3 /= np.maximum(np.linalg.norm(n3, axis=1, keepdims=True), 1e-12)
        return p3, n3

    u_lo, u_hi = float(u_b.min()), float(u_b.max())
    v_lo, v_hi = float(v_b.min()), float(v_b.max())
    uu, vv = np.meshgrid(np.linspace(u_lo, u_hi, 18), np.linspace(v_lo, v_hi, 18))
    u_grid, v_grid = uu.ravel(), vv.ravel()
    inside = (contains_xy(poly2d, u_grid, v_grid)
              if (not poly2d.is_empty and poly2d.area > 1e-12) else np.ones(len(u_grid), bool))
    if inside.sum() < 4:
        u_mid = np.linspace(u_lo, u_hi, 24)
        v_mid = np.full_like(u_mid, 0.5 * (v_lo + v_hi))
        prior_pts, prior_nrms = eval_prior(u_mid, v_mid)
    else:
        prior_pts, prior_nrms = eval_prior(u_grid[inside], v_grid[inside])

    if ref_normals is not None and len(ref_normals):
        ref_mean = np.asarray(ref_normals, float).mean(axis=0)
        if float(np.dot(prior_nrms.mean(axis=0), ref_mean)) < 0.0:
            prior_nrms = -prior_nrms
    return prior_pts, prior_nrms
