"""Prepare a normalized mesh and scalar E/V observations from an input STEP."""
import numpy as np

from .field_provider import CallableUDFProvider
from .schemas import SurfaceMesh
from .veudf_contract import require_veudf_v5


class ExactUDF:
    """Query finite CAD edges and necessary vertices in normalized coordinates."""

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
        self.last_distances = None

    def values(self, points):
        if self.last_points is None or not np.array_equal(points, self.last_points):
            if self.lines is None:
                edge = self.edge.query(points)["distance"]
            else:
                # Exact finite-segment distance in a bounded vectorized batch.
                d2 = np.full(len(points), np.inf)
                for start, end in self.lines:
                    direction = end - start
                    t = np.clip((points - start) @ direction / (direction @ direction), 0, 1)
                    delta = points - start - t[:, None] * direction
                    np.minimum(d2, np.einsum('ij,ij->i', delta, delta), out=d2)
                edge = np.sqrt(d2)
            vertex = self.vertex.query(points)["distance"]
            self.last_points = points.copy()
            self.last_distances = np.column_stack((edge, vertex))
        return self.last_distances


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
    oracle = ExactUDF(targets, cfg)
    presence = (
        "present" if targets.edges else "absent",
        "present" if len(targets.vertices) else "absent",
    )
    provider = CallableUDFProvider(
        oracle.values,
        fingerprint=f"production-cad:{fingerprint}:{code_fingerprint()}",
        presence=presence,
    )
    return mesh, provider
