from dataclasses import dataclass, asdict, fields
import json
import math
from pathlib import Path

@dataclass
class Config:
    coarse_spacing: float = 0.012
    fine_spacing: float = 0.003
    band_width: float = 0.02
    shell_radius: float = 0.005
    coarse_error_margin: float = 0.0
    support_radius: float = 0.007
    scout_point_budget: int = 4000000
    extraction: str = 'trace'
    query_batch_size: int = 65536
    sample_timeout_seconds: float = 10800.0
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
    curve_init_samples: int = 256
    curve_fit_samples: int = 512
    curve_refine_samples: int = 2048
    curve_distance_iterations: int = 15
    feature_sdf_tolerance: float = 0.004

    @property
    def feature_sdf_limit(self):
        return self.feature_sdf_tolerance

    def validate(self):
        for key in ('query_batch_size','max_segments','scout_point_budget',
                    'face_samples','spline_controls','max_spline_controls','uv_iterations',
                    'curve_init_samples','curve_fit_samples','curve_refine_samples','curve_distance_iterations'):
            v = getattr(self,key)
            if type(v) is not int or v < 1: raise ValueError(f'{key} must be a positive integer')
        if self.spline_controls < 4 or self.max_spline_controls < self.spline_controls:
            raise ValueError('invalid spline controls')
        if not (8 <= self.curve_init_samples <= self.curve_fit_samples <= self.curve_refine_samples):
            raise ValueError('Require 8 <= curve_init_samples <= curve_fit_samples <= curve_refine_samples')
        for key in ('coarse_spacing','fine_spacing','band_width','shell_radius','support_radius',
                    'sample_timeout_seconds','field_error','surface_tolerance','localization_tolerance',
                    'curve_tolerance','feature_sdf_tolerance','boundary_tolerance','cad_tolerance'):
            v = getattr(self,key)
            if isinstance(v,bool) or not math.isfinite(v) or v <= 0: raise ValueError(f'invalid {key}')
        factor = self.coarse_spacing / self.fine_spacing
        if not (1 <= round(factor) <= 16 and math.isclose(factor, round(factor))):
            raise ValueError('coarse_spacing must be an integer multiple (1–16) of fine_spacing')
        if self.shell_radius >= self.band_width:
            raise ValueError('shell_radius must be smaller than band_width')
        if not math.isfinite(self.coarse_error_margin) or self.coarse_error_margin < 0:
            raise ValueError('coarse_error_margin must be non-negative')
        if self.extraction not in ('trace', 'mc', 'both'):
            raise ValueError('extraction must be trace, mc, or both')
        if self.stop_after not in ('scout','wireframe','split_mesh','solid'):
            raise ValueError('invalid stop_after')
        return self

    def to_dict(self): return asdict(self)

def load_config(path=None, overrides=None):
    """Edge-first defaults < JSON < explicit overrides."""
    d = {}
    d.update(json.loads(Path(path).read_text()) if path else {})
    d.update({k:v for k,v in (overrides or {}).items() if v is not None})
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown: raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    return Config(**d).validate()
