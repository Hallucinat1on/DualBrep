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
    # Dataset eligibility (whole model rejected on any violation)
    min_faces: int = 2
    max_faces: int = 100
    max_face_edges: int = 50
    min_edge_length: float = 0.001
    # Final CAD mesh; optional volume supervision is separate from the four pools.
    volume_pool_count: int = 0
    evaluation_volume_count: int = 4096
    feature_curve_spacing: float = 0.002
    geometry_surface_pool_count: int = 393216
    pool_count: int = 1572864
    pool_weights: tuple = (0.25, 0.35, 0.25, 0.15)
    band_max: float = 0.03
    feature_radii: tuple = (0.002, 0.005, 0.01, 0.02, 0.03)
    feature_weights: tuple = (0.2, 0.3, 0.3, 0.2, 0.0)
    feature_min_samples: int = 32
    sampling_rounds: int = 32
    evaluation_points_per_pool: int = 4096
    zero_checks_per_edge: int = 3
    batch_size: int = 4096
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
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{f.name} must be finite and positive")
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
    for key in ("pool_weights", "feature_radii", "feature_weights"):
        if key in data:
            data[key] = tuple(data[key])
    return Config(**data).validate()
