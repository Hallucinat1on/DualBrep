from dataclasses import dataclass, asdict, fields
import json
import math
from pathlib import Path

@dataclass
class Config:
    query_batch_size: int = 8192
    refinement_coarse_edge: float = .02
    refinement_target_edge: float = .005
    refinement_rho_edge: float = .02
    refinement_rho_vertex: float = .03  # Legacy configuration compatibility; selection uses E only.
    sample_timeout_seconds: float = 10800.
    presence_samples: int = 393216
    presence_tau: float = .005
    presence_min_points: int = 16
    tau_edge: float = .025
    tau_vertex: float = .02
    face_threshold1: float = .005
    face_threshold2: float = .01
    face_second_pass: bool = True
    face_filter_size: int = 10
    face_avg_udf_threshold: float = .003
    field_error: float = .0002
    surface_tolerance: float = .003
    localization_tolerance: float = .001
    curve_tolerance: float = .001
    boundary_tolerance: float = .001
    cad_tolerance: float = .002
    max_segments: int = 64
    face_samples: int = 1200
    spline_controls: int = 8
    max_spline_controls: int = 16
    uv_iterations: int = 4
    seed: int = 0
    stop_after: str = 'solid'
    save_queries: bool = True
    band_connectivity: str = 'vertex'
    band_propagation: str = 'noncore'
    vertex_domains: str = 'local'
    vertex_radius_fraction: float = .45
    vertex_cut_factor: float = 1.5
    vertex_min_votes: int = 3
    infer_short_edges: bool = True
    surface_candidates: str = 'revolution'
    curve_candidates: str = 'arcs'
    periodic_frame_planning: bool = True
    feature_sdf_tolerance: float = None

    @property
    def feature_sdf_limit(self):
        return self.surface_tolerance if self.feature_sdf_tolerance is None else self.feature_sdf_tolerance

    def validate(self):
        if isinstance(self.sample_timeout_seconds,bool):raise ValueError("invalid sample_timeout_seconds")
        if self.feature_sdf_tolerance is not None:
            if isinstance(self.feature_sdf_tolerance, bool) or not math.isfinite(self.feature_sdf_tolerance) or self.feature_sdf_tolerance <= 0:
                raise ValueError('invalid feature_sdf_tolerance')
        if self.band_connectivity not in ('edge', 'vertex'):
            raise ValueError('invalid band_connectivity')
        if self.curve_candidates not in ('basic', 'arcs'):
            raise ValueError('invalid curve_candidates')
        if self.vertex_domains not in ('threshold', 'local'):
            raise ValueError('invalid vertex_domains')
        if self.surface_candidates not in ('basic', 'revolution'):
            raise ValueError('invalid surface_candidates')
        for key in ('infer_short_edges', 'periodic_frame_planning', 'face_second_pass'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(self.vertex_min_votes) is not int or self.vertex_min_votes < 1:
            raise ValueError('vertex_min_votes must be a positive integer')
        if not (0 < self.vertex_radius_fraction < .5):
            raise ValueError('vertex_radius_fraction must lie in (0, 0.5)')
        if not (math.isfinite(self.vertex_cut_factor) and self.vertex_cut_factor > 0):
            raise ValueError('invalid vertex_cut_factor')
        if self.band_propagation not in ('distance_only', 'noncore'):
            raise ValueError('invalid band_propagation')
        for key in ('presence_samples','presence_min_points','query_batch_size','max_segments',
                    'face_samples','spline_controls','max_spline_controls','uv_iterations','face_filter_size'):
            v = getattr(self,key)
            if type(v) is not int or v < 1: raise ValueError(f'{key} must be a positive integer')
        if self.spline_controls < 4 or self.max_spline_controls < self.spline_controls:
            raise ValueError('invalid spline controls')
        if self.face_threshold2 < self.face_threshold1:
            raise ValueError('face_threshold2 must be >= face_threshold1')
        for key in ('sample_timeout_seconds','refinement_coarse_edge','refinement_target_edge','refinement_rho_edge','refinement_rho_vertex','presence_tau','tau_edge','tau_vertex','face_threshold1','face_threshold2','face_avg_udf_threshold','field_error','surface_tolerance',
                    'localization_tolerance','curve_tolerance','boundary_tolerance','cad_tolerance'):
            v = getattr(self,key)
            if isinstance(v,bool) or not math.isfinite(v) or v <= 0: raise ValueError(f'invalid {key}')
        if self.refinement_target_edge > self.refinement_coarse_edge:
            raise ValueError('refinement_target_edge must not exceed refinement_coarse_edge')
        if self.stop_after not in ('seeds','graph','vertices','curves','surfaces','solid'): raise ValueError('invalid stop_after')
        return self

    def to_dict(self): return asdict(self)

def load_config(path=None, overrides=None):
    """Production mesh/L5 defaults < JSON < explicit overrides."""
    d = {}
    d.update(json.loads(Path(path).read_text()) if path else {})
    d.update({k:v for k,v in (overrides or {}).items() if v is not None})
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown: raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    return Config(**d).validate()
