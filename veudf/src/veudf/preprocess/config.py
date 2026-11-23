"""One configuration for filtering, labeling, condition construction and deduplication."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Normalization / CAD tolerances
    bounding: float = 0.9
    mesh_deflection: float = 0.0002
    mesh_angle: float = 0.15
    solver_tolerance: float = 1e-7
    geometry_tolerance: float = 2e-5
    label_tolerance: float = 1e-4
    tie_tolerance: float = 1e-7
    unify_same_domain: bool = True
    # Dataset eligibility (whole model rejected on any violation)
    min_faces: int = 2
    max_faces: int = 100
    max_face_edges: int = 50
    min_edge_length: float = 0.001
    # Final CAD mesh; optional volume supervision is separate from the four pools.
    volume_pool_count: int = 0
    evaluation_volume_count: int = 4096
    feature_curve_spacing: float = 0.002
    geometry_surface_pool_count: int = 262144
    pool_count: int = 1179648
    pool_weights: tuple = (2 / 9, 1 / 3, 5 / 18, 1 / 6)
    feature_surface_ratios: tuple = (0.0, 0.0)
    band_max: float = 0.04
    feature_radii: tuple = (0.002, 0.005, 0.01, 0.02, 0.04)
    feature_weights: tuple = (0.20, 0.28, 0.27, 0.17, 0.08)
    feature_min_samples: int = 32
    sampling_rounds: int = 32
    evaluation_points_per_pool: int = 4096
    zero_edge_count: int = 65536
    zero_edge_min_samples: int = 8
    evaluation_zero_edge_count: int = 4096
    batch_size: int = 4096
    label_batch_size: int = 32768
    udf_truncation: float = 0.05
    reference_checks: int = 128
    seed: int = 0
    # Geometric near-duplicate policy (project thresholds, not DualBrep's)
    dedup_surface_points: int = 16384
    dedup_edge_points: int = 4096
    # Fixed arc-length pool, independent of the supervised surface pools.
    dedup_edge_pool_count: int = 65536
    dedup_surface_rms: float = 0.001
    dedup_surface_p95: float = 0.002
    dedup_edge_rms: float = 0.001
    dedup_vertex_max: float = 0.001

    def validate(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is int or f.type == "int":
                if type(value) is not int or value < (
                    0 if f.name in ("seed", "volume_pool_count") else 1
                ):
                    raise ValueError(f"{f.name} must be a positive integer")
            elif f.type is float or f.type == "float":
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{f.name} must be finite and positive")
            elif f.type is bool or f.type == "bool":
                if type(value) is not bool:
                    raise ValueError(f"{f.name} must be a boolean")
        if self.bounding >= 1:
            raise ValueError("bounding must leave padding inside [-1,1]^3")
        if self.bounding + self.band_max >= 1:
            raise ValueError("band_max must keep near-surface queries inside [-1,1]^3")
        if self.band_max < 0.01:
            raise ValueError(
                "band_max must be at least 0.01 for the near-surface strata"
            )
        if self.max_faces < self.min_faces:
            raise ValueError("max_faces < min_faces")
        if self.udf_truncation <= max(self.feature_radii) + self.label_tolerance:
            raise ValueError("UDF truncation must cover all feature sampling layers")
        if len(self.feature_surface_ratios) != 2 or any(
            not math.isfinite(r) or r != 0 for r in self.feature_surface_ratios
        ):
            raise ValueError(
                "feature_surface_ratios must be (0.0, 0.0) for pure 3D feature pools"
            )
        from veudf.utils import allocate

        if (
            len(self.pool_weights) != 4
            or min(self.pool_weights) < 0
            or abs(sum(self.pool_weights) - 1) > 1e-8
        ):
            raise ValueError("pool_weights must sum to one and have four entries")
        if (
            allocate(self.pool_count, self.pool_weights)[0]
            < self.geometry_surface_pool_count
        ):
            raise ValueError("Surface pool must contain the fixed geometry prefix")
        if (
            not self.feature_radii
            or len(self.feature_radii) != len(self.feature_weights)
            or any(not math.isfinite(r) or r <= 0 for r in self.feature_radii)
            or list(self.feature_radii) != sorted(set(self.feature_radii))
            or self.feature_radii[-1] > self.band_max
            or any(not math.isfinite(w) for w in self.feature_weights)
            or min(self.feature_weights) < 0
            or abs(sum(self.feature_weights) - 1) > 1e-8
        ):
            raise ValueError("Invalid feature distance strata")
        return self

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


def load_config(path=None, overrides=None):
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for key in (
        "pool_weights",
        "feature_surface_ratios",
        "feature_radii",
        "feature_weights",
    ):
        if key in data:
            data[key] = tuple(data[key])
    return Config(**data).validate()
