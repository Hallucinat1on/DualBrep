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
    band_connectivity: str = 'edge'
    band_propagation: str = 'distance_only'
    vertex_domains: str = 'threshold'
    vertex_radius_fraction: float = .45
    vertex_cut_factor: float = 1.5
    vertex_min_votes: int = 3
    infer_short_edges: bool = False
    surface_candidates: str = 'basic'
    curve_candidates: str = 'basic'
    periodic_frame_planning: bool = False

    def validate(self):
        if self.band_connectivity not in ('edge', 'vertex'):
            raise ValueError('invalid band_connectivity')
        if self.curve_candidates not in ('basic', 'arcs'):
            raise ValueError('invalid curve_candidates')
        if self.vertex_domains not in ('threshold', 'local'):
            raise ValueError('invalid vertex_domains')
        if self.surface_candidates not in ('basic', 'revolution'):
            raise ValueError('invalid surface_candidates')
        for key in ('infer_short_edges', 'periodic_frame_planning'):
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

# Named presets.  'recommended' is the configuration evaluated on the fresh
# holdout H100b on 2026-10-18 (38/100 STEP vs 19/100 for the same settings with
# the four improvement switches off); see docs/ThreeFieldReconstructionExperiments.md.
PRESETS = {
    'recommended': dict(
        resolution=257, max_resolution=257, query_batch_size=1048576, save_queries=False,
        band_connectivity='vertex', band_propagation='noncore', tau_vertex=.02,
        vertex_domains='local', infer_short_edges=True,
        surface_candidates='revolution', curve_candidates='arcs', periodic_frame_planning=True),
}

def load_config(path=None, overrides=None, preset=None):
    """Precedence: dataclass defaults < preset < JSON file < explicit overrides."""
    if preset is not None and preset not in PRESETS:
        raise ValueError(f'unknown preset: {preset!r}; choose from {sorted(PRESETS)}')
    d = dict(PRESETS[preset]) if preset else {}
    d.update(json.loads(Path(path).read_text()) if path else {})
    d.update({k:v for k,v in (overrides or {}).items() if v is not None})
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown: raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    return Config(**d).validate()
