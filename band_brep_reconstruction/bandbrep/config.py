from dataclasses import dataclass, asdict, fields
import json
import math
from pathlib import Path

@dataclass
class Config:
    query_batch_size: int = 65536
    refinement_coarse_edge: float = 0.03
    refinement_target_edge: float = 0.003
    refinement_rho_edge: float = 0.02
    sample_timeout_seconds: float = 10800.0
    tau_edge: float = 0.025
    tau_vertex: float = 0.02
    face_threshold1: float = 0.003
    vertex_width_fallback: float = 0.01
    face_filter_size: int = 10
    face_avg_udf_threshold: float = 0.003
    field_error: float = 0.0002
    surface_tolerance: float = 0.003
    localization_tolerance: float = 0.001
    curve_tolerance: float = 0.001
    boundary_tolerance: float = 0.001
    cad_tolerance: float = 0.002
    max_segments: int = 64
    face_samples: int = 1200
    spline_controls: int = 8
    max_spline_controls: int = 16
    uv_iterations: int = 4
    seed: int = 0
    stop_after: str = 'solid'
    vertex_radius_fraction: float = 0.45
    vertex_cut_factor: float = 1.5
    vertex_min_votes: int = 3
    curve_init_samples: int = 256
    curve_fit_samples: int = 512
    curve_refine_samples: int = 2048
    curve_distance_iterations: int = 15
    feature_sdf_tolerance: float = 0.004

    @property
    def feature_sdf_limit(self):
        return self.feature_sdf_tolerance

    def validate(self):
        if type(self.vertex_min_votes) is not int or self.vertex_min_votes < 1:
            raise ValueError('vertex_min_votes must be a positive integer')
        if not (0 < self.vertex_radius_fraction < .5):
            raise ValueError('vertex_radius_fraction must lie in (0, 0.5)')
        if not (math.isfinite(self.vertex_cut_factor) and self.vertex_cut_factor > 0):
            raise ValueError('invalid vertex_cut_factor')
        for key in ('query_batch_size','max_segments',
                    'face_samples','spline_controls','max_spline_controls','uv_iterations','face_filter_size',
                    'curve_init_samples','curve_fit_samples','curve_refine_samples','curve_distance_iterations'):
            v = getattr(self,key)
            if type(v) is not int or v < 1: raise ValueError(f'{key} must be a positive integer')
        if self.spline_controls < 4 or self.max_spline_controls < self.spline_controls:
            raise ValueError('invalid spline controls')
        if not (8 <= self.curve_init_samples <= self.curve_fit_samples <= self.curve_refine_samples):
            raise ValueError('Require 8 <= curve_init_samples <= curve_fit_samples <= curve_refine_samples')
        for key in ('sample_timeout_seconds','refinement_coarse_edge','refinement_target_edge','refinement_rho_edge','tau_edge','tau_vertex','face_threshold1','vertex_width_fallback','face_avg_udf_threshold','field_error','surface_tolerance',
                    'localization_tolerance','curve_tolerance','feature_sdf_tolerance','boundary_tolerance','cad_tolerance'):
            v = getattr(self,key)
            if isinstance(v,bool) or not math.isfinite(v) or v <= 0: raise ValueError(f'invalid {key}')
        if self.refinement_target_edge > self.refinement_coarse_edge:
            raise ValueError('refinement_target_edge must not exceed refinement_coarse_edge')
        if self.stop_after not in ('seeds','graph','vertices','curves','surfaces','solid'): raise ValueError('invalid stop_after')
        return self

    def to_dict(self): return asdict(self)

def load_config(path=None, overrides=None):
    """Post1109 defaults < JSON < explicit overrides."""
    d = {}
    d.update(json.loads(Path(path).read_text()) if path else {})
    d.update({k:v for k,v in (overrides or {}).items() if v is not None})
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown: raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    return Config(**d).validate()
