"""Exact CAD E/V with explicit condition-SDF and fail-closed band semantics."""
import sys
from collections import defaultdict

import numpy as np
from bandbrep.field_provider import FieldProvider
from bandbrep.schemas import require
from bandbrep.vendor.oracle.distance import EdgeOracle, VertexOracle
from alignment_grid import grid_value


class AlignmentField(FieldProvider):
    def __init__(self, source, grid, mode, fingerprint, band_width=.02, progress=None):
        self.source = source
        self.grid = grid
        self.mode = mode
        self.band_width = band_width
        self.fingerprint = fingerprint
        self.edge = EdgeOracle(source.targets.edges, source.config, backend='cad_bvh')
        self.vertex = VertexOracle(source.targets.vertices, source.config)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.progress = progress
        self.last_stage = None
        self.last_points = None
        self.last_edge = None
        self.center_points = None
        self.center_values = None

    def presence(self):
        return ('present' if self.source.targets.edges else 'absent',
                'present' if len(self.source.targets.vertices) else 'absent')

    def stage(self):
        frame = sys._getframe(1)
        stages = {'localize_vertices': 'vertices', 'build_edge_geometry': 'edges',
                  'fit_shared_curves': 'curves', 'validate_structure_coverage': 'coverage',
                  'initialize_face_surfaces': 'surfaces', 'recover_topology': 'topology',
                  'build_bands': 'bands', 'orient_and_validate': 'surface_validation'}
        stage = 'prepare'
        while frame is not None:
            if frame.f_code.co_name in stages:
                stage = stages[frame.f_code.co_name]
                break
            frame = frame.f_back
        if stage != self.last_stage:
            self.last_stage = stage
            if self.progress:
                self.progress(stage)
        return stage

    def sdf(self, xyz):
        p = np.asarray(xyz, float).reshape(-1, 3)
        require(np.isfinite(p).all() and np.max(np.abs(p), initial=0) <= 1.000001,
                'field_domain', 'Query outside [-1,1]')
        self.counts[self.stage()]['sdf_queries'] += len(p)
        return self.source.sdf(p) if self.mode == 'A' else grid_value(self.grid, p).astype(float)

    def check_band(self, p):
        sdf = self.sdf(p)
        # All groups use their supplied SDF to define the same width contract.
        inside = np.abs(sdf) <= self.band_width
        stage = self.stage()
        self.counts[stage]['ev_queries'] += len(p)
        self.counts[stage]['outside_band'] += int((~inside).sum())
        require(inside.all(), 'field_outside_band', 'E/V requested outside the valid band',
                count=int((~inside).sum()), max_abs_sdf=float(np.max(np.abs(sdf), initial=0)),
                band_width=self.band_width, stage=stage)
        return sdf

    def edge_query(self, p):
        if self.last_points is not None and np.array_equal(p, self.last_points):
            return self.last_edge
        result = self.edge.query(p)
        self.last_points, self.last_edge = p.copy(), result
        return result

    def evaluate(self, xyz):
        p = np.asarray(xyz, float).reshape(-1, 3)
        sdf = self.check_band(p)
        if self.center_points is not None and len(p):
            if not hasattr(self, '_center_starts'):
                self._center_starts = {}
                for begin in range(0, len(self.center_points), 65536):
                    self._center_starts.setdefault(tuple(self.center_points[begin]), []).append(begin)
            for begin in self._center_starts.get(tuple(p[0]), []):
                if np.array_equal(p, self.center_points[begin:begin + len(p)]):
                    values = self.center_values[begin:begin + len(p)].copy()
                    values[:, 0] = sdf
                    self.counts[self.stage()]['center_cache_hits'] += len(p)
                    return self.result(values)
        edge = self.edge_query(p)
        vertex = self.vertex.query(p)
        return self.result(np.column_stack((sdf, edge['distance'], vertex['distance'])))

    def gradient(self, xyz, channel, step=1e-5):
        p = np.asarray(xyz, float).reshape(-1, 3)
        if channel == 0:
            if self.mode == 'A':
                return super().gradient(p, channel, step)
            g = grid_value(self.grid, p, gradient=True).astype(float)
            return g, np.isfinite(g).all(1) & (np.linalg.norm(g, axis=1) > 1e-8)
        self.check_band(p)
        result = self.edge_query(p) if channel == 1 else self.vertex.query(p)
        stage = self.stage()
        self.counts[stage]['gradient_queries'] += len(p)
        self.counts[stage]['invalid_gradient'] += int((~result['gradient_valid']).sum())
        return result['gradient'].copy(), result['gradient_valid'].copy()

    def report(self):
        return {'band_width': self.band_width, 'outside_policy': 'fail_closed',
                'edge_backend': 'cad_bvh', 'presence': 'oracle',
                'strict_error_bound': False, 'counts': dict(self.counts),
                'cad_distance_solver_calls': self.edge.calls}
