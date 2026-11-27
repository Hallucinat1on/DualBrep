"""Prepare a normalized mesh and exact 8D E/V field observations from an input STEP."""
import numpy as np

from .field_provider import CallableUDFProvider
from .schemas import SurfaceMesh
from .veudf_contract import require_veudf_v5


class ExactField8D:
    """Query finite CAD edges and necessary vertices in normalized coordinates (8D: dE, dV, uE, uV)."""

    def __init__(self, targets, cfg):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Line
        from veudf.preprocess.cad import xyz
        from veudf.preprocess.reference import ReferenceEdgeOracle, ReferenceVertexOracle

        adaptors = [BRepAdaptor_Curve(edge) for edge in targets.edges]
        self.lines = None
        self.edge = None
        if adaptors and all(curve.GetType() == GeomAbs_Line for curve in adaptors):
            self.lines = [(np.asarray(xyz(curve.Value(curve.FirstParameter()))),
                           np.asarray(xyz(curve.Value(curve.LastParameter())))) for curve in adaptors]
        else:
            self.edge = ReferenceEdgeOracle(targets.edges, cfg)
        self.vertex = ReferenceVertexOracle(targets.vertices, cfg)
        self.last_points = None
        self.last_output = None

    def values(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.last_points is None or not np.array_equal(points, self.last_points):
            if self.lines is None:
                eres = self.edge.query(points)
                edge_dist = eres["distance"]
                edge_offset = eres["offset"]
            else:
                d2 = np.full(len(points), np.inf)
                edge_offset = np.zeros((len(points), 3), dtype=np.float64)
                for start, end in self.lines:
                    direction = end - start
                    denom = float(direction @ direction)
                    t = np.clip((points - start) @ direction / max(denom, 1e-30), 0.0, 1.0)
                    offset = start + t[:, None] * direction - points
                    cand_d2 = np.einsum('ij,ij->i', offset, offset)
                    better = cand_d2 < d2
                    d2[better] = cand_d2[better]
                    edge_offset[better] = offset[better]
                edge_dist = np.sqrt(d2)
            vres = self.vertex.query(points)
            vertex_dist = vres["distance"]
            vertex_offset = vres["offset"]
            self.last_points = points.copy()
            self.last_output = np.column_stack((edge_dist, vertex_dist, edge_offset, vertex_offset))
        return self.last_output


def load_cad_source(path, fingerprint):
    require_veudf_v5()

    from veudf.preprocess.config import Config
    from veudf.preprocess.cad import read_normalized, collect_targets, triangulate
    from veudf.preprocess.condition import weld
    from veudf.preprocess.filtering import check_solid_faces, audit_shape
    from veudf.preprocess.pipeline import code_fingerprint, validate_geometry

    cfg = Config().validate()
    solid, normalization = read_normalized(str(path), cfg)
    check_solid_faces(solid, cfg)
    targets = collect_targets(solid, cfg)
    validate_geometry(targets, cfg)
    audit_shape(solid, targets, cfg)
    (vertices, triangles), _ = weld(triangulate(targets, cfg), cfg)
    mesh = SurfaceMesh(vertices, triangles, {"normalization": normalization, "source_step_sha256": fingerprint})
    oracle = ExactField8D(targets, cfg)
    presence = ("present", "present") if targets.edges else ("absent", "absent")
    provider = CallableUDFProvider(
        oracle.values,
        fingerprint=f"production-cad-8d:{fingerprint}:{code_fingerprint()}",
        presence=presence,
    )
    return mesh, provider
