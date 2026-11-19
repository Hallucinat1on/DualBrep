"""Feature-first finite-CAD UDFs, clipped before far-field distance solving."""

import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Core.gp import gp_Pnt

from veudf.preprocess.cad import PreprocessError, bounds, xyz
from veudf.preprocess.reference import make_tool, solve
from veudf.preprocess.sampling import curve_points


def reduce_min(result, rows, distances, instance, local_ties, tolerance):
    """Strict minima, with stable first-ID selection for exactly equal values."""
    previous = result["distance"][rows]
    known = result["nearest_id"][rows] >= 0
    improved = distances < previous
    tied = known & (np.abs(distances - previous) <= tolerance)
    ambiguity = np.where(
        improved, (result["tie"][rows] & tied) | tied | local_ties,
        result["tie"][rows] | tied,
    )
    changed = rows[improved]
    result["distance"][changed] = distances[improved]
    result["nearest_id"][changed] = instance
    result["tie"][rows] = ambiguity


class FeatureUDFOracle:
    """One geometry preparation per asset, one shared point KDTree per batch.

    Edge seed balls conservatively cover the finite curve's r-tube, including
    CAD endpoint tolerance. They only choose candidates: final labels always
    come from the original CAD edge. Vertex balls use exact Euclidean distance.
    Saturated points have no claimed nearest instance. Empty targets also use
    the training radius, but remain absent and are not marked saturated.
    """

    def __init__(self, targets, cfg):
        self.cfg, self.targets, self.radius = cfg, targets, cfg.udf_truncation
        self.tools = [make_tool(e, cfg.solver_tolerance) for e in targets.edges]
        self.covers, self.boxes, self.circles = [], [], {}
        padding = cfg.geometry_tolerance + 4 * cfg.solver_tolerance
        for index, edge in enumerate(targets.edges):
            curve = BRepAdaptor_Curve(edge)
            length = GCPnts_AbscissaPoint.Length(curve, cfg.solver_tolerance)
            segments = max(2, int(np.ceil(length / self.radius)))
            seeds = curve_points(edge, segments + 1, cfg.solver_tolerance)
            self.covers.append((seeds, self.radius + length / segments / 2 + padding))
            self.boxes.append(bounds(edge, cfg.geometry_tolerance))
            if curve.GetType() == GeomAbs_Circle:
                circle = curve.Circle()
                self.circles[index] = (xyz(circle.Location()), xyz(circle.Axis().Direction()))
        self.calls = self.batches = 0

    def _empty(self, count):
        return dict(
            distance=np.full(count, self.radius),
            nearest_id=np.full(count, -1, np.int32), tie=np.zeros(count, bool),
            saturated=np.zeros(count, bool),
        )

    def _finish(self, result, present):
        if present:
            saturated = result["distance"].astype(np.float32) >= np.float32(self.radius)
            result["saturated"] = saturated
            result["distance"][saturated] = self.radius
            result["nearest_id"][saturated] = -1
            result["tie"][saturated] = False
        return result

    def query(self, points):
        q = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite feature UDF query")
        edge = self._empty(len(q))
        vertex = self._empty(len(q))
        self.batches += 1
        if not len(q):
            return edge, vertex
        tree = cKDTree(q)
        query_vertices = [None] * len(q)
        slack = self.cfg.tie_tolerance + 2 * self.cfg.solver_tolerance
        for index, ((seeds, cover), (lo, hi)) in enumerate(zip(self.covers, self.boxes)):
            groups = tree.query_ball_point(seeds, cover, eps=0, workers=1, return_sorted=False)
            groups = [g for g in groups if len(g)]
            if not groups:
                continue
            rows = np.unique(np.concatenate(groups)).astype(np.int64)
            delta = np.maximum(np.maximum(lo - q[rows], q[rows] - hi), 0)
            lower = np.einsum("ij,ij->i", delta, delta)
            rows = rows[lower <= (edge["distance"][rows] + slack) ** 2]
            distances, local_ties = np.empty(len(rows)), np.zeros(len(rows), bool)
            for j, row in enumerate(rows):
                if query_vertices[row] is None:
                    query_vertices[row] = BRepBuilderAPI_MakeVertex(gp_Pnt(*q[row])).Vertex()
                distance, projected, local_tie = solve(self.tools[index], query_vertices[row])
                self.calls += 1
                if abs(np.linalg.norm(q[row] - projected) - distance) > self.cfg.label_tolerance:
                    raise PreprocessError("distance_solver_failed", "Closest-point residual")
                distances[j], local_ties[j] = distance, local_tie
            if index in self.circles:
                center, normal = self.circles[index]
                displacement = q[rows] - center
                radial = displacement - (displacement @ normal)[:, None] * normal
                local_ties |= np.linalg.norm(radial, axis=1) <= self.cfg.solver_tolerance
            reduce_min(edge, rows, distances, index, local_ties, self.cfg.tie_tolerance)
        for index, point in enumerate(self.targets.vertices):
            rows = np.asarray(tree.query_ball_point(
                point, self.radius + slack, eps=0, workers=1
            ), np.int64)
            distances = np.linalg.norm(q[rows] - point, axis=1)
            reduce_min(vertex, rows, distances, index, False, self.cfg.tie_tolerance)
        return self._finish(edge, bool(self.targets.edges)), self._finish(vertex, bool(len(self.targets.vertices)))
