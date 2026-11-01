"""Exact finite CAD-edge, necessary-vertex and boundary SDF distance queries."""

from dataclasses import dataclass
import heapq
import itertools

import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON
from OCC.Core.gp import gp_Pnt

from veudf.schema import EMPTY_DISTANCE
from veudf.preprocess.cad import PreprocessError, bounds, compound, xyz


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


@dataclass
class Node:
    lo: np.ndarray
    hi: np.ndarray
    indices: tuple = ()
    children: tuple = ()


def build_bvh(boxes, indices):
    lo = np.min([boxes[i][0] for i in indices], axis=0)
    hi = np.max([boxes[i][1] for i in indices], axis=0)
    if len(indices) <= 2:
        return Node(lo, hi, tuple(indices))
    centers = np.array([(boxes[i][0] + boxes[i][1]) / 2 for i in indices])
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    ordered = [indices[i] for i in np.argsort(centers[:, axis], kind="stable")]
    mid = len(ordered) // 2
    return Node(
        lo,
        hi,
        children=(build_bvh(boxes, ordered[:mid]), build_bvh(boxes, ordered[mid:])),
    )


def box_distance(point, node):
    return float(
        np.linalg.norm(np.maximum(np.maximum(node.lo - point, point - node.hi), 0))
    )


def empty_result(count):
    """Empty target set: placeholder distance, no nearest entity."""
    return dict(
        distance=np.full(count, EMPTY_DISTANCE, dtype=np.float64),
        nearest_id=np.full(count, -1, dtype=np.int32),
        tie=np.zeros(count, dtype=bool),
    )


class EdgeOracle:
    """``cad_bvh`` (labels) or ``cad_reference`` (all-edge checks)."""

    def __init__(self, edges, cfg, backend="cad_bvh"):
        if backend not in ("cad_reference", "cad_bvh"):
            raise ValueError("Unsupported edge distance backend")
        self.cfg, self.backend, self.edges = cfg, backend, edges
        self.tools = [make_tool(edge, cfg.solver_tolerance) for edge in edges]
        self.circles = {}
        for index, edge in enumerate(edges):
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() == GeomAbs_Circle:
                circle = curve.Circle()
                self.circles[index] = (
                    xyz(circle.Location()),
                    xyz(circle.Axis().Direction()),
                )
        self.boxes = [bounds(edge, cfg.geometry_tolerance) for edge in edges]
        self.root = build_bvh(self.boxes, list(range(len(edges)))) if edges else None
        self.calls = 0

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = empty_result(len(points))
        if not self.edges:
            return result
        slack = self.cfg.tie_tolerance + 2 * self.cfg.solver_tolerance
        for row, point in enumerate(points):
            query_vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
            best, nearest, ambiguous = float("inf"), -1, False
            candidates, heap = [], []
            if self.backend == "cad_reference":
                candidates = list(range(len(self.edges)))
            else:
                counter = itertools.count()
                heap = [(box_distance(point, self.root), next(counter), self.root)]
            while candidates or heap:
                if candidates:
                    index = candidates.pop()
                else:
                    lower, _, node = heapq.heappop(heap)
                    if lower > best + slack:
                        continue
                    if node.children:
                        for child in node.children:
                            lower = box_distance(point, child)
                            if lower <= best + slack:
                                heapq.heappush(heap, (lower, next(counter), child))
                    else:
                        candidates.extend(node.indices)
                    continue
                distance, projected, local_tie = solve(self.tools[index], query_vertex)
                self.calls += 1
                if index in self.circles:
                    center, normal = self.circles[index]
                    displacement = point - center
                    radial = displacement - np.dot(displacement, normal) * normal
                    # The whole arc is equidistant from an axis point.
                    local_tie |= np.linalg.norm(radial) <= self.cfg.solver_tolerance
                if (
                    abs(np.linalg.norm(point - projected) - distance)
                    > self.cfg.label_tolerance
                ):
                    raise PreprocessError(
                        "distance_solver_failed",
                        "Closest-point distance disagrees with CAD value",
                    )
                if distance < best:
                    is_tie = best - distance <= self.cfg.tie_tolerance
                    ambiguous = (ambiguous and is_tie) or is_tie or local_tie
                    best, nearest = distance, index
                elif distance - best <= self.cfg.tie_tolerance:
                    ambiguous = True
            result["distance"][row] = best
            result["nearest_id"][row] = nearest
            result["tie"][row] = ambiguous
        return result


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


class CADSDFOracle:
    """Exact CAD boundary distance and solid sign, used only for reference audits."""

    def __init__(self, solid, faces, cfg):
        self.cfg = cfg
        self.classifier = BRepClass3d_SolidClassifier(solid)
        self.tool = make_tool(compound(faces), cfg.solver_tolerance)

    def classify(self, point):
        self.classifier.Perform(gp_Pnt(*point), self.cfg.solver_tolerance)
        state = self.classifier.State()
        if state == TopAbs_IN:
            return -1.0
        if state == TopAbs_OUT:
            return 1.0
        if state == TopAbs_ON:
            return 0.0
        raise PreprocessError("sdf_sign_failed", "Solid classifier returned UNKNOWN")

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        values = np.empty(len(points), dtype=np.float64)
        for i, point in enumerate(points):
            distance = solve(
                self.tool, BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
            )[0]
            values[i] = self.classify(point) * distance
        if not np.isfinite(values).all():
            raise PreprocessError("sdf_sign_failed", "Nonfinite CAD SDF")
        return values
