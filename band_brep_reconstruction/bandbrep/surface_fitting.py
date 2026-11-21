"""Analytic initialization and smooth boundary-weighted tensor-spline fitting."""
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .schemas import ReconstructionError, require
from .surface_mesh import normals


@dataclass
class FacePatch:
    id: int
    kind: str
    data: dict
    normal: np.ndarray
    report: dict

    def uv(self, points, ordered=False):
        p = np.asarray(points, float)
        d = self.data
        if self.kind in ('plane', 'bspline'):
            if self.kind == 'bspline' and 'mesh_uv' in d:
                tree = cKDTree(d['mesh_pts'])
                idx = tree.query(p)[1]
                uv = d['mesh_uv'][idx].copy()
            else:
                uv = np.column_stack(((p - d['origin']) @ d['x'], (p - d['origin']) @ d['y']))
                if self.kind == 'bspline':
                    uv = (uv - d['uv_lo']) / d['uv_extent']
            if self.kind == 'bspline':
                uv = np.clip(uv, 0, 1)
                def _newton(uv_in, pts_in, iters=10):
                    u = uv_in.copy()
                    for _ in range(iters):
                        base = self.evaluate(u)
                        h = 1e-4
                        du = (self.evaluate(u + [h, 0]) - base) / h
                        dv = (self.evaluate(u + [0, h]) - base) / h
                        J = np.stack((du, dv), axis=2)
                        lhs = np.einsum('nki,nkj->nij', J, J) + np.eye(2) * 1e-10
                        rhs = np.einsum('nki,nk->ni', J, pts_in - base)
                        delta = np.linalg.solve(lhs, rhs[..., None])[..., 0]
                        u = np.clip(u + np.clip(delta, -.15, .15), 0, 1)
                        if np.max(np.abs(delta)) < 1e-8:
                            break
                    return u
                uv = _newton(uv, p, iters=10)
                if ordered and len(p) >= 2:
                    # Enforce UV continuity along ordered curves by marching from uv[0]
                    # whenever a jump occurs between consecutive samples.
                    duv = np.linalg.norm(np.diff(uv, axis=0), axis=1)
                    jumps = np.flatnonzero(duv > max(0.015, 3.0 * float(np.median(duv))))
                    if len(jumps):
                        cur = uv[0:1].copy()
                        fixed = [cur[0]]
                        for k in range(1, len(p)):
                            cand = _newton(cur, p[k:k + 1], iters=12)
                            e_cand = np.linalg.norm(self.evaluate(cand) - p[k:k + 1])
                            e_orig = np.linalg.norm(self.evaluate(uv[k:k + 1]) - p[k:k + 1])
                            if e_cand <= max(e_orig * 2.0, 1e-3):
                                cur = cand
                            else:
                                cur = uv[k:k + 1]
                            fixed.append(cur[0])
                        uv = np.asarray(fixed)
            return uv
        q = p - d['origin']
        z = q @ d['axis']
        x = q @ d['x']
        y = q @ d['y']
        u = np.arctan2(y, x)
        if ordered:
            u = np.unwrap(u)
        if self.kind == 'cylinder':
            return np.column_stack((u, z))
        if self.kind == 'sphere':
            return np.column_stack((u, np.arctan2(z, np.sqrt(x * x + y * y))))
        if self.kind == 'cone':
            return np.column_stack((u, z / np.cos(d['angle'])))
        v = np.arctan2(z, np.sqrt(x * x + y * y) - d['major'])
        return np.column_stack((u, np.unwrap(v) if ordered else v))

    def evaluate(self, uv):
        uv = np.asarray(uv, float)
        d = self.data
        u, v = uv.T
        if self.kind == 'plane':
            return d['origin'] + u[:, None] * d['x'] + v[:, None] * d['y']
        if self.kind == 'bspline':
            return tensor_basis(uv, d['knots'], d['degree']) @ d['controls'].reshape(-1, 3)
        radial = np.cos(u)[:, None] * d['x'] + np.sin(u)[:, None] * d['y']
        if self.kind == 'cylinder':
            return d['origin'] + d['radius'] * radial + v[:, None] * d['axis']
        if self.kind == 'sphere':
            return d['origin'] + d['radius'] * (np.cos(v)[:, None] * radial + np.sin(v)[:, None] * d['axis'])
        if self.kind == 'cone':
            return d['origin'] + (d['radius'] + v * np.sin(d['angle']))[:, None] * radial + (v * np.cos(d['angle']))[:, None] * d['axis']
        return d['origin'] + (d['major'] + d['minor'] * np.cos(v))[:, None] * radial + (d['minor'] * np.sin(v))[:, None] * d['axis']


def frame(axis):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x = np.eye(3)[np.argmin(np.abs(axis))]
    x -= np.dot(x, axis) * axis
    x /= np.linalg.norm(x)
    return x, np.cross(axis, x)


def tensor_basis(uv, knots, degree=3):
    n = len(knots) - degree - 1
    b = BSpline(knots, np.eye(n), degree, extrapolate=True)
    a = b(uv[:, 0])
    c = b(uv[:, 1])
    return np.einsum('ni,nj->nij', a, c).reshape(len(uv), n * n)


def _thin_plate_matrix(n):
    """Second-difference bending energy on an n x n control net."""
    rows = []
    for i in range(n):
        for j in range(n - 2):
            r = np.zeros(n * n)
            r[i * n + j] = 1.
            r[i * n + j + 1] = -2.
            r[i * n + j + 2] = 1.
            rows.append(r)
    for i in range(n - 2):
        for j in range(n):
            r = np.zeros(n * n)
            r[i * n + j] = 1.
            r[(i + 1) * n + j] = -2.
            r[(i + 2) * n + j] = 1.
            rows.append(r)
    for i in range(n - 1):
        for j in range(n - 1):
            r = np.zeros(n * n)
            r[i * n + j] = 1.
            r[i * n + j + 1] = -1.
            r[(i + 1) * n + j] = -1.
            r[(i + 1) * n + j + 1] = 1.
            rows.append(r * np.sqrt(2.0))
    return np.asarray(rows)


def _lscm_parameterize(mesh, face_tris):
    """Intrinsic Least Squares Conformal Map for curved disk-topology faces."""
    if len(face_tris) < 2:
        return None
    vids, inv = np.unique(face_tris, return_inverse=True)
    local_tris = inv.reshape(-1, 3)
    pts = mesh.vertices[vids]
    n_v = len(vids)
    # Pick two farthest vertices as conformal pins to fix translation/scale/rotation.
    p0 = 0
    d0 = np.linalg.norm(pts - pts[p0], axis=1)
    pin_a = int(np.argmax(d0))
    d1 = np.linalg.norm(pts - pts[pin_a], axis=1)
    pin_b = int(np.argmax(d1))
    if pin_a == pin_b or d1[pin_b] <= 1e-12:
        return None
    free = [i for i in range(n_v) if i not in (pin_a, pin_b)]
    free_idx = {v: k for k, v in enumerate(free)}
    n_f = len(free)
    if n_f == 0:
        return None
    # Build LSCM sparse matrix M * z = rhs (complex u + i*v represented as 2 real vars).
    rows, cols, vals = [], [], []
    rhs_u = np.zeros(2 * len(local_tris))
    row_id = 0
    for a, b, c in local_tris:
        pa, pb, pc = pts[a], pts[b], pts[c]
        e1 = pb - pa
        l1 = np.linalg.norm(e1)
        if l1 <= 1e-14:
            row_id += 2
            continue
        x_ax = e1 / l1
        nrm = np.cross(e1, pc - pa)
        area2 = np.linalg.norm(nrm)
        if area2 <= 1e-14:
            row_id += 2
            continue
        y_ax = np.cross(nrm / area2, x_ax)
        x0, y0 = 0.0, 0.0
        x1, y1 = l1, 0.0
        x2, y2 = float((pc - pa) @ x_ax), float((pc - pa) @ y_ax)
        scale = 1.0 / np.sqrt(area2)
        # Complex coefficients for (a, b, c)
        coeffs = [(a, x2 - x1, y2 - y1), (b, x0 - x2, y0 - y2), (c, x1 - x0, y1 - y0)]
        for vid, cr, ci in coeffs:
            cr *= scale
            ci *= scale
            if vid == pin_a:
                # pinned to (0, 0)
                continue
            elif vid == pin_b:
                # pinned to (1, 0)
                rhs_u[row_id] -= cr
                rhs_u[row_id + 1] -= ci
            else:
                k = free_idx[vid]
                # Real part: cr * u - ci * v
                rows.extend([row_id, row_id])
                cols.extend([2 * k, 2 * k + 1])
                vals.extend([cr, -ci])
                # Imag part: ci * u + cr * v
                rows.extend([row_id + 1, row_id + 1])
                cols.extend([2 * k, 2 * k + 1])
                vals.extend([ci, cr])
        row_id += 2
    M = coo_matrix((vals, (rows, cols)), shape=(2 * len(local_tris), 2 * n_f)).tocsr()
    A = M.T @ M
    b = M.T @ rhs_u
    try:
        sol = spsolve(A, b)
    except Exception:
        return None
    if not np.all(np.isfinite(sol)):
        return None
    uv = np.zeros((n_v, 2))
    uv[pin_b] = [1.0, 0.0]
    for k, vid in enumerate(free):
        uv[vid] = [sol[2 * k], sol[2 * k + 1]]
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    ext = np.maximum(hi - lo, 1e-8)
    uv = (uv - lo) / ext
    return pts, uv


def sample_face_regions(mesh, labels, cfg):
    """Area-weighted interior sampling per face region."""
    rng = np.random.default_rng(cfg.seed)
    output = {}
    nrm = normals(mesh)
    for face in sorted(np.unique(labels[labels >= 0])):
        ids = np.flatnonzero(labels == face)
        tri = mesh.vertices[mesh.triangles[ids]]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if area.sum() <= 1e-18:
            continue
        chosen = rng.choice(len(ids), cfg.face_samples, p=area / area.sum())
        root = np.sqrt(rng.random(cfg.face_samples))
        v = rng.random(cfg.face_samples)
        b = np.column_stack((1 - root, root * (1 - v), root * v))
        pts = np.einsum('ni,nij->nj', b, tri[chosen])
        output[int(face)] = (pts, nrm[ids[chosen]])
    return output


def _core_samples(points, normal, boundaries, cfg):
    """Exclude samples very close to boundaries when enough interior samples exist."""
    if not len(boundaries) or len(points) < 40:
        return points, normal
    margin = min(cfg.fine_spacing, 0.15 * float(np.linalg.norm(np.ptp(points, axis=0))))
    dist = cKDTree(boundaries).query(points)[0]
    core = dist >= margin
    if core.sum() >= max(32, int(0.35 * len(points))):
        return points[core], normal[core]
    return points, normal


def analytic_candidates(points, normal, boundaries):
    allpts = np.vstack((points, boundaries)) if len(boundaries) else points
    mean = allpts.mean(axis=0)
    _, _, vt = np.linalg.svd(allpts - mean, full_matrices=False)
    axis = vt[-1]
    x, y = frame(axis)
    if len(boundaries) >= 3:
        bm = boundaries.mean(axis=0)
        _, _, bvt = np.linalg.svd(boundaries - bm, full_matrices=False)
        if np.max(np.abs((boundaries - bm) @ bvt[-1])) < 1e-5:
            mean = bm
            axis = bvt[-1]
            x, y = frame(axis)
    candidates = [FacePatch(-1, 'plane', dict(origin=mean, x=x, y=y, axis=axis), normal.mean(axis=0), {})]
    # Robust sphere fit (soft_l1 resists scan outliers/rounding).
    A = np.column_stack((2 * allpts, np.ones(len(allpts))))
    fit = np.linalg.lstsq(A, np.sum(allpts**2, axis=1), rcond=None)[0]
    radius = np.sqrt(max(0, fit[3] + fit[:3] @ fit[:3]))
    if np.isfinite(radius) and 1e-5 < radius < 100:
        sphere = least_squares(lambda z: np.linalg.norm(allpts - z[:3], axis=1) - z[3],
                               np.r_[fit[:3], radius], loss='soft_l1', f_scale=1e-3, max_nfev=40)
        x, y = frame(np.array([0., 0., 1.]))
        candidates.append(FacePatch(-1, 'sphere', dict(origin=sphere.x[:3], radius=sphere.x[3],
                                                       axis=np.array([0., 0., 1.]), x=x, y=y), normal.mean(axis=0), {}))
    _, _, vn = np.linalg.svd(normal, full_matrices=False)
    axes = [vn[-1], vt[0], vt[-1]]
    for ax in axes:
        x, y = frame(ax)
        uv = np.column_stack(((allpts - mean) @ x, (allpts - mean) @ y))
        fit = np.linalg.lstsq(np.column_stack((2 * uv, np.ones(len(uv)))), np.sum(uv * uv, axis=1), rcond=None)[0]
        rad = np.sqrt(max(0, fit[2] + fit[0]**2 + fit[1]**2))
        if rad <= 1e-6:
            continue
        c = mean + x * fit[0] + y * fit[1]
        initial = np.r_[c, ax, rad]

        def residual(z):
            a = z[3:6] / max(np.linalg.norm(z[3:6]), 1e-10)
            q = allpts - z[:3]
            h = q @ a
            return np.r_[np.linalg.norm(q - h[:, None] * a, axis=1) - z[6], (np.linalg.norm(z[3:6]) - 1) * .1]

        res = least_squares(residual, initial, loss='soft_l1', f_scale=1e-3, max_nfev=60)
        a = res.x[3:6]
        a /= np.linalg.norm(a)
        x, y = frame(a)
        if res.x[6] > 0:
            candidates.append(FacePatch(-1, 'cylinder', dict(origin=res.x[:3], axis=a, x=x, y=y, radius=res.x[6]),
                                        normal.mean(axis=0), {}))
    return candidates


def _axis_point(points, normals_in, axis):
    m = np.cross(normals_in, axis)
    rhs = np.einsum('ij,ij->i', m, points)
    ok = np.linalg.norm(m, axis=1) > 1e-3
    if ok.sum() < 3:
        return points.mean(axis=0)
    c = np.linalg.lstsq(m[ok], rhs[ok], rcond=None)[0]
    return c - ((c - points.mean(axis=0)) @ axis) * axis


def _profile(points, origin, axis):
    q = points - origin
    z = q @ axis
    rho = np.linalg.norm(q - z[:, None] * axis, axis=1)
    return rho, z


def revolution_candidates(points, normal, related, boundaries):
    """Cylinder / cone / torus by robust joint axis + meridian-profile least squares."""
    guesses = []
    circles = [c for c in related if c.kind == 'circle']
    for c in circles:
        guesses.append((np.asarray(c.data['center'], float), np.asarray(c.data['normal'], float)))
    if len(circles) >= 2:
        d = circles[1].data['center'] - circles[0].data['center']
        if np.linalg.norm(d) > 1e-4:
            guesses.append((np.asarray(circles[0].data['center'], float), d / np.linalg.norm(d)))
    for centered in (True, False):
        n = normal - normal.mean(axis=0) if centered else normal
        a = np.linalg.svd(n, full_matrices=False)[2][-1]
        guesses.append((_axis_point(points, normal, a), a))
    data = np.vstack((points, boundaries)) if len(boundaries) else points
    weight = np.r_[np.ones(len(points)), np.full(len(boundaries), 3.)]
    out = []
    for kind in ('cylinder', 'cone', 'torus'):
        best = None
        for c0, a0 in guesses:
            a0 = a0 / np.linalg.norm(a0)
            x0, y0 = frame(a0)
            rho, z = _profile(data, c0, a0)
            if kind == 'cylinder':
                p0 = [np.median(rho)]
            elif kind == 'cone':
                k, r0 = np.polyfit(z, rho, 1)
                if abs(k) < 1e-3:
                    continue
                p0 = [r0, k]
            else:
                A = np.column_stack((2 * rho, 2 * z, np.ones(len(z))))
                sol = np.linalg.lstsq(A, rho * rho + z * z, rcond=None)[0]
                R, zc = sol[:2]
                r = np.sqrt(max(sol[2] + R * R + zc * zc, 0))
                if not (r > 1e-4 and R > r * .5):
                    continue
                p0 = [R, zc, r]

            def residual(s):
                a = a0 + s[0] * x0 + s[1] * y0
                a /= np.linalg.norm(a)
                c = c0 + s[2] * x0 + s[3] * y0
                rho, z = _profile(data, c, a)
                p = s[4:]
                if kind == 'cylinder':
                    r = rho - p[0]
                elif kind == 'cone':
                    r = (rho - p[0] - p[1] * z) / np.sqrt(1 + p[1]**2)
                else:
                    r = np.hypot(rho - p[0], z - p[1]) - p[2]
                return weight * r

            try:
                fit = least_squares(residual, np.r_[0, 0, 0, 0, p0], loss='soft_l1', f_scale=1e-3,
                                    max_nfev=200, x_scale='jac')
            except ValueError:
                continue
            if not np.all(np.isfinite(fit.x)):
                continue
            score = float(np.abs(fit.fun).max())
            if best is None or score < best[0]:
                best = (score, fit.x, a0, x0, y0, c0)
        if best is None:
            continue
        _, s, a0, x0, y0, c0 = best
        a = a0 + s[0] * x0 + s[1] * y0
        a /= np.linalg.norm(a)
        c = c0 + s[2] * x0 + s[3] * y0
        x, y = frame(a)
        p = s[4:]
        if kind == 'cylinder' and p[0] > 0:
            out.append(FacePatch(-1, 'cylinder', dict(origin=c, axis=a, x=x, y=y, radius=float(p[0])),
                                 normal.mean(axis=0), {}))
        elif kind == 'cone':
            _, z = _profile(data, c, a)
            z0 = float(np.median(z))
            radius = float(p[0] + p[1] * z0)
            angle = float(np.arctan(p[1]))
            if radius > 0 and 1e-3 < abs(angle) < np.pi / 2 - 1e-3:
                out.append(FacePatch(-1, 'cone', dict(origin=c + z0 * a, axis=a, x=x, y=y, radius=radius, angle=angle),
                                     normal.mean(axis=0), {}))
        elif kind == 'torus' and p[0] > 0 and p[2] > 0:
            out.append(FacePatch(-1, 'torus', dict(origin=c + p[1] * a, axis=a, x=x, y=y,
                                                   major=float(p[0]), minor=float(p[2])), normal.mean(axis=0), {}))
    return out


def _solve_smooth_spline(A, points, B, boundaries, n, w_bnd=35.0, w_reg=1e-3):
    """Weighted least squares with high boundary weight and thin-plate smoothness."""
    reg = _thin_plate_matrix(n)
    lhs = np.vstack((A, w_bnd * B, w_reg * reg))
    rhs = np.vstack((points, w_bnd * boundaries, np.zeros((len(reg), 3))))
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def fit_general_spline(face, points, boundaries, normal, cfg, *, mesh=None, face_tris=None):
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
    x, y = vt[:2]
    axis = np.cross(x, y)

    # Choose between PCA UV and intrinsic LSCM UV based on normal turning.
    n_mean = normal.mean(axis=0)
    n_len = np.linalg.norm(n_mean)
    min_dot = float((normal @ (n_mean / max(n_len, 1e-12))).min()) if n_len > 1e-6 else -1.0
    uv_modes = ['pca']
    if mesh is not None and face_tris is not None and len(face_tris) >= 2:
        if min_dot < 0.25:
            uv_modes = ['lscm', 'pca']
        else:
            uv_modes = ['pca', 'lscm']

    last_err = None
    for mode in uv_modes:
        extra_data = {}
        if mode == 'lscm':
            lscm = _lscm_parameterize(mesh, face_tris)
            if lscm is None:
                continue
            m_pts, m_uv = lscm
            tree = cKDTree(m_pts)
            uv = m_uv[tree.query(points)[1]].copy()
            bu = m_uv[tree.query(boundaries)[1]].copy()
            lo, extent = np.zeros(2), np.ones(2)
            extra_data = dict(mesh_pts=m_pts, mesh_uv=m_uv)
        else:
            uv = np.column_stack(((points - mean) @ x, (points - mean) @ y))
            bu = np.column_stack(((boundaries - mean) @ x, (boundaries - mean) @ y))
            lo = np.minimum(uv.min(axis=0), bu.min(axis=0))
            hi = np.maximum(uv.max(axis=0), bu.max(axis=0))
            extent = np.maximum(hi - lo, 1e-8)
            uv = (uv - lo) / extent
            bu = (bu - lo) / extent

        for n in range(cfg.spline_controls, cfg.max_spline_controls + 1, 2):
            knots = np.r_[np.zeros(4), np.linspace(0, 1, n - 2)[1:-1], np.ones(4)]
            data = dict(origin=mean, x=x, y=y, axis=axis, uv_lo=lo, uv_extent=extent,
                        knots=knots, degree=3, **extra_data)
            cur_uv, cur_bu = uv.copy(), bu.copy()
            patch = None
            for _ in range(cfg.uv_iterations):
                A = tensor_basis(cur_uv, knots)
                B = tensor_basis(cur_bu, knots)
                P = _solve_smooth_spline(A, points, B, boundaries, n)
                data['controls'] = P.reshape(n, n, 3)
                patch = FacePatch(face, 'bspline', data, normal.mean(axis=0), {})
                cur_uv = patch.uv(points)
                cur_bu = patch.uv(boundaries)
            # Final re-solve at converged UVs.
            P = _solve_smooth_spline(tensor_basis(cur_uv, knots), points, tensor_basis(cur_bu, knots), boundaries, n)
            data['controls'] = P.reshape(n, n, 3)
            patch = FacePatch(face, 'bspline', data, normal.mean(axis=0), {})
            b_err = float(np.linalg.norm(patch.evaluate(cur_bu) - boundaries, axis=1).max())
            i_err = float(np.quantile(np.linalg.norm(patch.evaluate(cur_uv) - points, axis=1), 0.95))
            if b_err <= cfg.boundary_tolerance and i_err <= cfg.surface_tolerance * 2:
                return patch
            last_err = f'mode={mode} n={n} b_err={b_err:.5f} i_err={i_err:.5f}'
    raise ReconstructionError('surface_boundary_incompatible', 'Spline budget cannot satisfy fixed boundary',
                              {'face': face, 'cause': str(last_err)})


def _refit_collapsed_edges(patches, graph, curves, collapsed_edges, cfg):
    """Replace collapsed edges between analytic faces with their exact surface-surface intersection."""
    if not collapsed_edges:
        return
    from OCC.Core.GeomAPI import GeomAPI_IntSS
    from .edge_first_graph import Path, fit_geometry
    from .occ_assembly import occ_surface

    analytic_kinds = ('plane', 'cylinder', 'cone', 'sphere', 'torus')
    for eid in sorted(collapsed_edges):
        if eid not in curves:
            continue
        edge = graph.edges[eid]
        if len(edge.faces) != 2:
            continue
        p0, p1 = patches.get(edge.faces[0]), patches.get(edge.faces[1])
        if p0 is None or p1 is None or p0.kind not in analytic_kinds or p1.kind not in analytic_kinds:
            continue
        try:
            inter = GeomAPI_IntSS(occ_surface(p0), occ_surface(p1), 1e-6)
            if not inter.IsDone() or inter.NbLines() == 0:
                continue
        except Exception:
            continue
        old_pts = curves[eid].evaluate(np.linspace(0, 1, 65))
        best_pts, best_dist = None, np.inf
        for k in range(1, inter.NbLines() + 1):
            c = inter.Line(k)
            f_u, l_u = c.FirstParameter(), c.LastParameter()
            if not (np.isfinite(f_u) and np.isfinite(l_u)) or l_u - f_u > 1e4:
                continue
            ts = np.linspace(f_u, l_u, 129)
            cand = np.array([[c.Value(float(t)).X(), c.Value(float(t)).Y(), c.Value(float(t)).Z()] for t in ts])
            d, idx = cKDTree(cand).query(old_pts)
            score = float(d.max())
            if score < best_dist:
                best_dist = score
                best_pts = cand[idx]
        if best_pts is None or best_dist > max(cfg.collapse_tolerance * 2.0, 0.01):
            continue
        ends = np.array([graph.vertices[v].position for v in edge.vertices]).reshape(-1, 3) if edge.vertices else np.empty((0, 3))
        if len(ends) == 2:
            # Project endpoints onto both analytic surfaces iteratively if moved.
            for j in range(2):
                pt = best_pts[0 if j == 0 else -1]
                if np.linalg.norm(pt - ends[j]) <= cfg.collapse_tolerance:
                    ends[j] = pt
                    graph.vertices[edge.vertices[j]].position = pt.copy()
        try:
            curves[eid] = fit_geometry(eid, Path(list(edge.vertices), best_pts, edge.closed), ends, cfg)
        except Exception:
            pass


def initialize_face_surfaces(mesh, face_labels, graph, curves, provider, cfg):
    regions = sample_face_regions(mesh, face_labels, cfg)
    patches = {}
    scan_limit = max(cfg.surface_tolerance * 2, cfg.scan_offset_tolerance)
    collapsed_edges = set(graph.report.get('wireframe', {}).get('collapsed_edges', []))
    for face in range(graph.face_count):
        require(face in regions, 'missing_face_region', 'No sampled region for face', face=face)
        p, n = regions[face]
        inc_edges = [e for e in graph.edges if face in e.faces]
        related = [curves[e.id] for e in inc_edges]
        has_collapsed = any(e.id in collapsed_edges for e in inc_edges)
        clean_related = [curves[e.id] for e in inc_edges if e.id not in collapsed_edges]
        boundary = np.vstack([c.evaluate(np.linspace(0, 1, 65)) for c in related]) if related else np.empty((0, 3))
        fit_boundary = (np.vstack([c.evaluate(np.linspace(0, 1, 65)) for c in clean_related])
                        if (has_collapsed and clean_related) else boundary)
        bnd_screen_tol = max(cfg.boundary_tolerance, cfg.collapse_tolerance) if has_collapsed else cfg.boundary_tolerance
        p_core, n_core = _core_samples(p, n, boundary, cfg)
        candidates = analytic_candidates(p_core, n_core, fit_boundary)
        if len(related) == 2 and all(c.kind == 'circle' for c in related):
            c0, c1 = related
            axis = c1.data['center'] - c0.data['center']
            if np.linalg.norm(axis) > 1e-6 and abs(c0.data['radius'] - c1.data['radius']) < bnd_screen_tol:
                axis /= np.linalg.norm(axis)
                x, y = frame(axis)
                candidates.insert(0, FacePatch(face, 'cylinder', dict(origin=c0.data['center'], axis=axis, x=x, y=y,
                                                                      radius=(c0.data['radius'] + c1.data['radius']) / 2), n[0], {}))
        accepted = []

        def screen(options):
            for patch in options:
                err = np.linalg.norm(patch.evaluate(patch.uv(p_core)) - p_core, axis=1)
                be = np.linalg.norm(patch.evaluate(patch.uv(boundary)) - boundary, axis=1) if len(boundary) else np.zeros(1)
                if np.quantile(err, 0.90) <= cfg.surface_tolerance * 2 and err.max() <= scan_limit and be.max() <= bnd_screen_tol:
                    accepted.append((float(err.mean() + be.max()), patch, float(be.max())))

        screen(candidates)
        accepted = [a for a in accepted if a[1].kind != 'plane' or a[2] <= bnd_screen_tol * .25] or accepted
        if not accepted:
            screen(revolution_candidates(p_core, n_core, related, fit_boundary))
            rank = {'cylinder': 0, 'cone': 1, 'torus': 2}
            accepted.sort(key=lambda item: (rank.get(item[1].kind, 3), item[0]))
            if accepted:
                accepted = accepted[:1]
        if accepted:
            patch = next((v for _, v, _ in accepted if v.kind == 'plane'), min(accepted, key=lambda z: z[0])[1])
            patch.id = face
        else:
            require(len(boundary) > 0, 'surface_initialization_failed',
                    'Unbounded nonanalytic face needs explicit UV initialization')
            face_tris = mesh.triangles[face_labels == face]
            patch = fit_general_spline(face, p_core, boundary, n_core, cfg, mesh=mesh, face_tris=face_tris)
        patch.normal = n_core[0].copy()
        patch.data['orientation_point'] = p_core[0].copy()
        patches[face] = patch

    if collapsed_edges:
        _refit_collapsed_edges(patches, graph, curves, collapsed_edges, cfg)

    for face in range(graph.face_count):
        patch = patches[face]
        p, n = regions[face]
        inc_edges = [e for e in graph.edges if face in e.faces]
        related = [curves[e.id] for e in inc_edges]
        boundary = np.vstack([c.evaluate(np.linspace(0, 1, 65)) for c in related]) if related else np.empty((0, 3))
        p_core, _ = _core_samples(p, n, boundary, cfg)
        maximum = 0.
        for e_inst, curve in zip(inc_edges, related):
            tol_e = max(cfg.boundary_tolerance, cfg.cad_tolerance) if e_inst.id in collapsed_edges else cfg.boundary_tolerance
            e_max = 0.
            for size in (129, 257, 513):
                q = curve.evaluate(np.linspace(0, 1, size))
                uv = patch.uv(q, ordered=True)
                e = np.linalg.norm(patch.evaluate(uv) - q, axis=1)
                e_max = max(e_max, float(e.max()))
            require(e_max <= tol_e, 'surface_boundary_incompatible',
                    'Fixed curve does not lie on support surface', face=face, error=e_max)
            maximum = max(maximum, e_max)
        uv = patch.uv(p_core)
        q = patch.evaluate(uv)
        errors = np.linalg.norm(q - p_core, axis=1)
        sdf = np.abs(provider.sdf(q))
        require(np.quantile(errors, 0.90) <= cfg.surface_tolerance * 2 and errors.max() <= scan_limit
                and np.quantile(sdf, 0.90) <= cfg.surface_tolerance * 2 and sdf.max() <= scan_limit,
                'surface_fit_failed', 'Interior fit exceeds error budget', face=face)
        h = 1e-5
        du = (patch.evaluate(uv + [h, 0]) - q) / h
        dv = (patch.evaluate(uv + [0, h]) - q) / h
        jac = np.linalg.norm(np.cross(du, dv), axis=1)
        # Exclude analytic cone apex / sphere poles where Jacobian vanishes naturally.
        jac_check = jac
        if patch.kind == 'sphere':
            jac_check = jac[np.abs(np.abs(uv[:, 1]) - np.pi / 2) > 1e-3]
        elif patch.kind == 'cone':
            v_apex = -patch.data['radius'] / np.sin(patch.data['angle'])
            jac_check = jac[np.abs(uv[:, 1] - v_apex) > 1e-3]
        require(not len(jac_check) or np.all(jac_check > 1e-10), 'surface_fold',
                'Degenerate local parameterization', face=face)
        patch.report = {
            'boundary_max_error': maximum,
            'interior_max_error': float(errors.max()),
            'interior_p90_error': float(np.quantile(errors, 0.90)),
            'sdf_max_error': float(sdf.max()),
            'jacobian_min': float(jac_check.min()) if len(jac_check) else 0.0,
            'continuous_boundary_guarantee': False,
            'global_self_intersection_certified': False,
            'boundary_check_samples_per_edge': 513,
            'periodic_u': patch.kind in ('cylinder', 'sphere', 'cone', 'torus'),
        }
    return patches
