"""Prepare a normalized mesh and scalar E/V observations from an input STEP."""
import numpy as np

from .field_provider import CallableUDFProvider
from .schemas import SurfaceMesh


class ExactUDF:
    """Query finite CAD edges and necessary vertices in normalized coordinates."""

    def __init__(self, targets, cfg):
        from veudf.preprocess.distance import EdgeOracle, VertexOracle

        self.edge = EdgeOracle(targets.edges, cfg)
        self.vertex = VertexOracle(targets.vertices, cfg)
        self.last_points = None
        self.last_distances = None
        self.counts = dict(evaluation_rows=0, edge_cache_rows=0, actual_edge_rows=0)

    def values(self, points):
        self.counts["evaluation_rows"] += len(points)
        if self.last_points is not None and np.array_equal(points, self.last_points):
            self.counts["edge_cache_rows"] += len(points)
        else:
            self.counts["actual_edge_rows"] += len(points)
            edge = self.edge.query(points)["distance"]
            vertex = self.vertex.query(points)["distance"]
            self.last_points = points.copy()
            self.last_distances = np.column_stack((edge, vertex))
        return self.last_distances

    def report(self):
        return dict(**self.counts, cad_distance_solver_calls=self.edge.calls)


def load_cad_source(path, fingerprint):
    from veudf.preprocess.config import Config
    from veudf.preprocess.cad import read_normalized, collect_targets, triangulate
    from veudf.preprocess.condition import weld
    from veudf.preprocess.filtering import check_solid_faces, audit_shape
    from veudf.preprocess.pipeline import validate_geometry

    cfg = Config().validate()
    solid, normalization = read_normalized(str(path), cfg)
    check_solid_faces(solid, cfg)
    targets = collect_targets(solid, cfg)
    validate_geometry(targets, cfg)
    audit_shape(solid, targets, cfg)
    (vertices, triangles), _ = weld(triangulate(targets, cfg), cfg)
    mesh = SurfaceMesh(vertices, triangles, {"normalization": normalization})
    return mesh, cad_udf(targets, cfg, fingerprint)[0]


def cad_udf(targets, cfg, fingerprint):
    oracle = ExactUDF(targets, cfg)
    presence = (
        "present" if targets.edges else "absent",
        "present" if len(targets.vertices) else "absent",
    )
    provider = CallableUDFProvider(
        oracle.values, fingerprint="production-cad:" + fingerprint, presence=presence
    )
    return provider, oracle
