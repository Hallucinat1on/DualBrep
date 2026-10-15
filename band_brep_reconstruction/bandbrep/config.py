from dataclasses import dataclass, asdict, fields
import json
import math
from pathlib import Path

@dataclass
class Config:
    resolution: int = 65
    max_resolution: int = 129
    bounds: tuple = (-1., 1.)
    query_batch_size: int = 8192
    tau_edge: float = .025
    tau_vertex: float = .04
    face_threshold1: float = .005
    face_threshold2: float = .01
    face_filter_size: int = 10
    face_avg_udf_threshold: float = .003
    max_triangles: int = 1500000
    field_error: float = .0002
    surface_tolerance: float = .003
    allow_surface_residual_exceedance: bool = False
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

    def validate(self):
        if type(self.allow_surface_residual_exceedance) is not bool:
            raise ValueError('allow_surface_residual_exceedance must be a boolean')
        for key in ('resolution','max_resolution','query_batch_size','max_triangles','max_segments',
                    'face_samples','spline_controls','max_spline_controls','uv_iterations','face_filter_size'):
            v = getattr(self,key)
            if type(v) is not int or v < 1: raise ValueError(f'{key} must be a positive integer')
        if self.max_resolution < self.resolution: raise ValueError('max_resolution must be >= resolution')
        if self.resolution < 9 or self.spline_controls < 4 or self.max_spline_controls < self.spline_controls:
            raise ValueError('invalid resolution / spline controls')
        if self.face_threshold2 < self.face_threshold1:
            raise ValueError('face_threshold2 must be >= face_threshold1')
        for key in ('tau_edge','tau_vertex','face_threshold1','face_threshold2','face_avg_udf_threshold','field_error','surface_tolerance',
                    'localization_tolerance','curve_tolerance','boundary_tolerance','cad_tolerance'):
            v = getattr(self,key)
            if not math.isfinite(v) or v <= 0: raise ValueError(f'invalid {key}')
        if self.stop_after not in ('seeds','graph','vertices','curves','surfaces','solid'): raise ValueError('invalid stop_after')
        if len(self.bounds) != 2 or self.bounds[0] >= self.bounds[1]: raise ValueError('invalid bounds')
        return self

    def to_dict(self): return asdict(self)

def load_config(path=None, overrides=None):
    d = json.loads(Path(path).read_text()) if path else {}
    d.update({k:v for k,v in (overrides or {}).items() if v is not None})
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown: raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    return Config(**d).validate()
