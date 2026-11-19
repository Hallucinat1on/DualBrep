"""Prepare a normalized mesh and scalar E/V observations from an input STEP."""
import numpy as np

from .field_provider import CallableUDFProvider
from .schemas import SurfaceMesh
from .veudf_contract import require_veudf_v3


class ExactUDF:
    """Query finite CAD edges and necessary vertices in normalized coordinates."""

    def __init__(self, targets, cfg):
        from veudf.preprocess.reference import ReferenceEdgeOracle, ReferenceVertexOracle

        self.edge = ReferenceEdgeOracle(targets.edges, cfg)
        self.vertex = ReferenceVertexOracle(targets.vertices, cfg)
        self.last_points = None
        self.last_distances = None

    def values(self, points):
        if self.last_points is None or not np.array_equal(points, self.last_points):
            edge = self.edge.query(points)["distance"]
            vertex = self.vertex.query(points)["distance"]
            self.last_points = points.copy()
            self.last_distances = np.column_stack((edge, vertex))
        return self.last_distances


def load_cad_source(path, fingerprint):
    require_veudf_v3()

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
    mesh = SurfaceMesh(vertices, triangles, {"normalization": normalization})
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
