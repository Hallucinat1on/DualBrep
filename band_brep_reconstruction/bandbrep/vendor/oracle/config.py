from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    bounding: float = 0.9
    query_count: int = 200000
    surface_count: int = 10000
    edge_count: int = 10000
    min_face_samples: int = 16
    min_edge_samples: int = 32
    edge_backend: str = "cad_bvh"
    sdf_backend: str = "cad"
    mesh_deflection: float = 0.0002
    mesh_angle: float = 0.15
    solver_tolerance: float = 1e-7
    geometry_tolerance: float = 2e-5
    label_tolerance: float = 1e-4
    tie_tolerance: float = 1e-7
    clip_sdf: float = 0.02
    clip_edge: float = 0.02
    clip_vertex: float = 0.02
    noise_scales: tuple = (0.0002, 0.001, 0.005, 0.02)
    # uniform, near-surface, on-surface, near-edge, near-vertex
    query_weights: tuple = (0.16, 0.24, 0.16, 0.32, 0.12)
    batch_size: int = 4096
    reference_checks: int = 128
    zero_checks_per_edge: int = 3
    seed: int = 0
    save_mesh: bool = True

    def validate(self):
        integer_names = ("query_count", "surface_count", "edge_count", "min_face_samples",
                         "min_edge_samples", "batch_size", "reference_checks", "zero_checks_per_edge")
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("bounding", "mesh_deflection", "mesh_angle", "solver_tolerance",
                     "geometry_tolerance", "label_tolerance", "tie_tolerance",
                     "clip_sdf", "clip_edge", "clip_vertex"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.bounding >= 1:
            raise ValueError("bounding must leave padding inside [-1,1]^3")
        if self.edge_backend not in ("cad_reference", "cad_bvh"):
            raise ValueError("edge_backend must be cad_reference or cad_bvh")
        if self.sdf_backend not in ("cad", "mesh", "mesh_fast"):
            raise ValueError("sdf_backend must be cad, mesh or mesh_fast")
        if len(self.query_weights) != 5 or any(not math.isfinite(v) or v < 0 for v in self.query_weights):
            raise ValueError("query_weights needs five finite, nonnegative weights")
        if sum(self.query_weights) <= 0:
            raise ValueError("query_weights must have positive total")
        if not self.noise_scales or any(not math.isfinite(v) or v <= 0 for v in self.noise_scales):
            raise ValueError("noise_scales must be finite and positive")
        if type(self.save_mesh) is not bool:
            raise ValueError("save_mesh must be boolean")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path=None, overrides=None):
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for key in ("noise_scales", "query_weights"):
        if key in data:
            data[key] = tuple(data[key])
    return Config(**data).validate()
