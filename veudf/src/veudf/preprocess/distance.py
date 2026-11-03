"""Global finite CAD-edge and necessary-vertex distances on surface samples."""

import time

import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Line
from OCC.Core.gp import gp_Pnt

from veudf.schema import EMPTY_DISTANCE
from veudf.preprocess.cad import PreprocessError, bounds, xyz


def make_tool(shape, tolerance):
    tool = BRepExtrema_DistShapeShape()
    tool.LoadS2(shape)
    tool.SetDeflection(tolerance)
    return tool


def solve(tool, point_vertex):
    tool.LoadS1(point_vertex)
    tool.Perform()
    if not tool.IsDone() or tool.NbSolution() < 1:
        raise PreprocessError(
            "distance_solver_failed", "OCC distance query did not converge"
        )
    distance = float(tool.Value())
    if not np.isfinite(distance) or distance < 0:
        raise PreprocessError(
            "distance_solver_failed", f"Invalid CAD distance: {distance}"
        )
    closest = xyz(tool.PointOnShape2(1))
    if not np.isfinite(closest).all():
        raise PreprocessError("distance_solver_failed", "Nonfinite closest point")
    return distance, closest, tool.NbSolution() > 1


def empty_result(count):
    """Empty target set: placeholder distance, no nearest entity."""
    return dict(
        distance=np.full(count, EMPTY_DISTANCE, dtype=np.float64),
        nearest_id=np.full(count, -1, dtype=np.int32),
        tie=np.zeros(count, dtype=bool),
    )


class EdgeOracle:
    """Global finite-edge distances, batched by edge and cached mesh triangle.

    Analytic segments/arcs avoid OCC point construction entirely. Conservative
    CAD boxes may remove candidates; face ownership never removes an edge.
    ``cad_reference`` deliberately uses all-edge OCC queries for independent audits.
    Instances own mutable OCC tools and must not be shared between worker threads.
    """

    def __init__(
        self,
        edges,
        cfg,
        backend="surface",
        *,
        mesh=None,
        edge_faces=None,
        triangle_face_ids=None,
    ):
        if backend not in ("surface", "cad_reference"):
            raise ValueError("Unsupported edge distance backend")
        self.cfg, self.backend, self.edges = cfg, backend, edges
        self.tools = [make_tool(edge, cfg.solver_tolerance) for edge in edges]
        self.boxes = np.asarray(
            [bounds(e, cfg.geometry_tolerance) for e in edges], float
        ).reshape(-1, 2, 3)
        self.analytic = []
        for edge in edges:
            c = BRepAdaptor_Curve(edge)
            first, last = c.FirstParameter(), c.LastParameter()
            if c.GetType() == GeomAbs_Line:
                self.analytic.append(("line", xyz(c.Value(first)), xyz(c.Value(last))))
            elif c.GetType() == GeomAbs_Circle:
                circle = c.Circle()
                self.analytic.append(
                    (
                        "circle",
                        xyz(circle.Location()),
                        circle.Radius(),
                        xyz(circle.XAxis().Direction()),
                        xyz(circle.YAxis().Direction()),
                        first,
                        last,
                    )
                )
            else:
                self.analytic.append(None)
        self.mesh = mesh
        if mesh is not None:
            tri = mesh[0][mesh[1]]
            self.triangle_lo, self.triangle_hi = tri.min(1), tri.max(1)
            self.centers = tri.mean(1)
            self.radii = np.linalg.norm(tri - self.centers[:, None], axis=2).max(1)
        self.triangle_face_ids = triangle_face_ids
        self.face_edges = {}
        for eid, faces in enumerate(edge_faces or []):
            for fid in faces:
                self.face_edges.setdefault(int(fid), []).append(eid)
        self.cache = {}
        self.calls = 0
        self.stats = dict(
            query_rows=0,
            analytic_rows=0,
            candidate_pairs=0,
            bbox_tests=0,
            cache_hits=0,
            cache_misses=0,
            index_seconds=0.0,
            query_seconds=0.0,
        )

    def _distance(self, eid, points):
        data = self.analytic[eid]
        if self.backend == "cad_reference" or data is None:
            distance, tie = np.empty(len(points)), np.zeros(len(points), bool)
            for i, p in enumerate(points):
                d, closest, ambiguous = solve(
                    self.tools[eid], BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex()
                )
                self.calls += 1
                if abs(np.linalg.norm(p - closest) - d) > self.cfg.label_tolerance:
                    raise PreprocessError(
                        "distance_solver_failed",
                        "Closest point disagrees with CAD distance",
                    )
                distance[i], tie[i] = d, ambiguous
            # For analytic primitives use geometric ambiguity, not the number
            # of near-duplicate numerical extrema returned by OCC.
            if data is not None:
                tie = self._analytic_distance(data, points)[1]
            return distance, tie
        self.stats["analytic_rows"] += len(points)
        return self._analytic_distance(data, points)

    def _analytic_distance(self, data, points):
        if data[0] == "line":
            _, a, b = data
            direction = b - a
            t = np.clip((points - a) @ direction / np.dot(direction, direction), 0, 1)
            return np.linalg.norm(
                points - a - t[:, None] * direction, axis=1
            ), np.zeros(len(points), bool)
        _, center, radius, x, y, first, last = data
        delta = points - center
        xx, yy = delta @ x, delta @ y
        angle = np.arctan2(yy, xx)
        parameter = first + np.mod(angle - first, 2 * np.pi)
        # A trimmed arc has two endpoint candidates in addition to its radial foot.
        parameter = np.minimum(parameter, last)
        foot = center + radius * (
            np.cos(parameter)[:, None] * x + np.sin(parameter)[:, None] * y
        )
        a = center + radius * (np.cos(first) * x + np.sin(first) * y)
        b = center + radius * (np.cos(last) * x + np.sin(last) * y)
        d = np.minimum(
            np.linalg.norm(points - foot, axis=1),
            np.minimum(
                np.linalg.norm(points - a, axis=1), np.linalg.norm(points - b, axis=1)
            ),
        )
        tie = np.hypot(xx, yy) <= self.cfg.solver_tolerance
        # Outside a trimmed arc both distinct endpoints may realize the minimum.
        da, db = np.linalg.norm(points - a, axis=1), np.linalg.norm(points - b, axis=1)
        if np.linalg.norm(a - b) > self.cfg.geometry_tolerance:
            tie |= (np.abs(da - db) <= self.cfg.tie_tolerance) & (
                np.abs(da - d) <= self.cfg.tie_tolerance
            )
        return d, tie

    def _lower(self, points, eid):
        lo, hi = self.boxes[eid]
        self.stats["bbox_tests"] += len(points)
        return np.linalg.norm(
            np.maximum(np.maximum(lo - points, points - hi), 0), axis=1
        )

    def _prepare_triangles(self, triangle_ids):
        unique = np.unique(triangle_ids)
        missing = np.array([t for t in unique if int(t) not in self.cache], np.int64)
        self.stats["cache_hits"] += len(unique) - len(missing)
        self.stats["cache_misses"] += len(missing)
        start = time.monotonic()
        for offset in range(0, len(missing), 512):
            tids = missing[offset : offset + 512]
            centers = self.centers[tids]
            lower = np.linalg.norm(
                np.maximum(
                    np.maximum(
                        self.boxes[None, :, 0] - centers[:, None],
                        centers[:, None] - self.boxes[None, :, 1],
                    ),
                    0,
                ),
                axis=2,
            )
            seed = lower.argmin(1)
            if self.triangle_face_ids is not None:
                face_ids = self.triangle_face_ids[tids]
                for fid in np.unique(face_ids):
                    hints = self.face_edges.get(int(fid), [])
                    if hints:
                        rows = np.flatnonzero(face_ids == fid)
                        seed[rows] = np.asarray(hints)[
                            lower[np.ix_(rows, hints)].argmin(1)
                        ]
            upper = np.empty(len(tids))
            for eid in np.unique(seed):
                rows = np.flatnonzero(seed == eid)
                upper[rows] = self._distance(int(eid), centers[rows])[0]
            upper += self.radii[tids]
            gap = np.maximum(
                np.maximum(
                    self.boxes[None, :, 0] - self.triangle_hi[tids, None],
                    self.triangle_lo[tids, None] - self.boxes[None, :, 1],
                ),
                0,
            )
            slack = self.cfg.tie_tolerance + 2 * self.cfg.solver_tolerance + 2e-7
            keep = np.linalg.norm(gap, axis=2) <= upper[:, None] + slack
            order = np.argsort(lower, axis=1, kind="stable")
            for i, tid in enumerate(tids):
                self.cache[int(tid)] = order[i, keep[i, order[i]]]
        self.stats["index_seconds"] += time.monotonic() - start

    def query(self, points, *, triangle_ids=None, source_ids=None):
        start = time.monotonic()
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite edge query")
        result = empty_result(len(points))
        self.stats["query_rows"] += len(points)
        if not self.edges or not len(points):
            return result
        if self.backend == "cad_reference":
            groups = [(np.arange(len(points)), np.arange(len(self.edges)))]
        elif triangle_ids is not None:
            if self.mesh is None:
                raise ValueError("Triangle queries require a mesh")
            triangle_ids = np.asarray(triangle_ids)
            if (
                triangle_ids.shape != (len(points),)
                or triangle_ids.dtype.kind not in "iu"
            ):
                raise ValueError("Invalid triangle IDs")
            if (triangle_ids < 0).any() or (triangle_ids >= len(self.mesh[1])).any():
                raise ValueError("Invalid triangle IDs")
            if np.any(
                np.linalg.norm(points - self.centers[triangle_ids], axis=1)
                > self.radii[triangle_ids] + 2e-7
            ):
                raise ValueError("Query outside its triangle's candidate bound")
            self._prepare_triangles(triangle_ids)
            # Group once rather than repeatedly scanning the whole batch per triangle.
            order = np.argsort(triangle_ids, kind="stable")
            parts = np.split(order, np.flatnonzero(np.diff(triangle_ids[order])) + 1)
            groups = [(ids, self.cache[int(triangle_ids[ids[0]])]) for ids in parts]
        else:
            # Generic queries (audits, postprocessing): group-vectorized CAD-box sweep.
            center = points.mean(0)
            lower = np.linalg.norm(
                np.maximum(
                    np.maximum(self.boxes[:, 0] - center, center - self.boxes[:, 1]), 0
                ),
                axis=1,
            )
            groups = [(np.arange(len(points)), np.argsort(lower, kind="stable"))]
        best = np.full(len(points), np.inf)
        second = best.copy()
        local_tie = np.zeros(len(points), bool)
        nearest = np.full(len(points), -1, np.int32)
        slack = self.cfg.tie_tolerance + 2 * self.cfg.solver_tolerance

        def evaluate(eid, ids):
            # Avoid counting an explicitly seeded edge twice for the same point.
            ids = ids[nearest[ids] != eid]
            if not len(ids):
                return
            if self.backend != "cad_reference":
                ids = ids[self._lower(points[ids], eid) <= best[ids] + slack]
            if not len(ids):
                return
            self.stats["candidate_pairs"] += len(ids)
            distance, ambiguous = self._distance(int(eid), points[ids])
            improved = distance < best[ids]
            yes, no = ids[improved], ids[~improved]
            second[yes] = np.minimum(second[yes], best[yes])
            best[yes], nearest[yes], local_tie[yes] = (
                distance[improved],
                eid,
                ambiguous[improved],
            )
            second[no] = np.minimum(second[no], distance[~improved])

        if source_ids is not None and self.backend != "cad_reference":
            source_ids = np.asarray(source_ids)
            if source_ids.shape != (len(points),):
                raise ValueError("Invalid edge hints")
            for eid in np.unique(source_ids):
                if 0 <= eid < len(self.edges):
                    evaluate(int(eid), np.flatnonzero(source_ids == eid))
        # Merge triangle groups by candidate edge so NumPy works on large batches,
        # not thousands of one-point analytic solves on a highly tessellated mesh.
        by_edge, seeds = {}, {}
        for ids, candidates in groups:
            if len(candidates):
                seeds.setdefault(int(candidates[0]), []).append(ids)
            for eid in candidates:
                by_edge.setdefault(int(eid), []).append(ids)
        for eid, rows in seeds.items():
            evaluate(eid, np.concatenate(rows))
        for eid, rows in by_edge.items():
            evaluate(eid, np.concatenate(rows))
        if not np.isfinite(best).all():
            raise PreprocessError("distance_solver_failed", "No finite nearest edge")
        result.update(
            distance=best,
            nearest_id=nearest,
            tie=local_tie | (second - best <= self.cfg.tie_tolerance),
        )
        self.stats["query_seconds"] += time.monotonic() - start
        return result

    def report(self):
        return dict(self.stats, occ_calls=self.calls, cached_triangles=len(self.cache))


class VertexOracle:
    def __init__(self, vertices, cfg):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.cfg = cfg
        self.tree = cKDTree(self.vertices) if len(self.vertices) else None

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = empty_result(len(points))
        if self.tree is None or not len(points):
            return result
        k = min(2, len(self.vertices))
        distance, index = self.tree.query(points, k=k, eps=0)
        if k == 2:
            result["tie"] = distance[:, 1] - distance[:, 0] <= self.cfg.tie_tolerance
            distance, index = distance[:, 0], index[:, 0]
        result["distance"] = distance
        result["nearest_id"] = index.astype(np.int32)
        return result
