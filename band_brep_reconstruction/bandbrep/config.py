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
    max_band_edge: float = .0125
    max_refinements: int = 7
    max_triangles: int = 1500000
    stability_factors: tuple = (.85, 1., 1.15)
    require_stability: bool = True
    verify_band_stability: bool = True
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

    def validate(self):
        for key in ('resolution','max_resolution','query_batch_size','max_refinements','max_triangles','max_segments',
                    'face_samples','spline_controls','max_spline_controls','uv_iterations'):
            v = getattr(self,key)
            if type(v) is not int or v < 1: raise ValueError(f'{key} must be a positive integer')
        if self.max_resolution < self.resolution: raise ValueError('max_resolution must be >= resolution')
        if self.resolution < 9 or self.spline_controls < 4 or self.max_spline_controls < self.spline_controls:
            raise ValueError('invalid resolution / spline controls')
        for key in ('tau_edge','tau_vertex','max_band_edge','field_error','surface_tolerance',
                    'localization_tolerance','curve_tolerance','boundary_tolerance','cad_tolerance'):
            v = getattr(self,key)
            if not math.isfinite(v) or v <= 0: raise ValueError(f'invalid {key}')
        if self.max_band_edge > self.tau_edge / 2 + 1e-12:
            raise ValueError('max_band_edge must be <= tau_edge/2')
        if self.stop_after not in ('seeds','graph','curves','surfaces','solid'): raise ValueError('invalid stop_after')
        if len(self.bounds) != 2 or self.bounds[0] >= self.bounds[1]: raise ValueError('invalid bounds')
        if not self.stability_factors or any(v <= 0 for v in self.stability_factors): raise ValueError('invalid stability factors')
        return self

    def to_dict(self): return asdict(self)

def load_config(path=None, overrides=None):
    d = json.loads(Path(path).read_text()) if path else {}
    d.update({k:v for k,v in (overrides or {}).items() if v is not None})
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown: raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    return Config(**d).validate()
