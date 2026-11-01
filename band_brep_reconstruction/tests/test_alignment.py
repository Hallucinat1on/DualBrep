import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
from alignment_grid import grid_value
from bandbrep.config import Config, load_config
from bandbrep.schemas import ReconstructionError
from bandbrep.vertex_localization import localize_vertices


class AlignmentTests(unittest.TestCase):
    def test_default_tolerance_equivalence(self):
        for value in (.001, .003, .005):
            cfg = Config(surface_tolerance=value)
            self.assertEqual(cfg.feature_sdf_limit, value)
            explicit = replace(cfg, feature_sdf_tolerance=value)
            self.assertEqual(explicit.feature_sdf_limit, cfg.feature_sdf_limit)
        cfg = load_config(preset='recommended', overrides={'feature_sdf_tolerance': .006})
        self.assertEqual(cfg.surface_tolerance, .003)
        self.assertEqual(cfg.boundary_tolerance, .001)
        self.assertEqual(cfg.cad_tolerance, .002)
        self.assertEqual(cfg.feature_sdf_limit, .006)

    def test_invalid_tolerance(self):
        for value in (-1, 0, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                Config(feature_sdf_tolerance=value).validate()

    def test_grid_nodes_and_linear_gradient(self):
        r = 17
        axis = np.linspace(-1, 1, r, dtype=np.float32)
        x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
        grid = x + 2 * y - .5 * z
        p = np.random.default_rng(7).uniform(-1, 1, (1000, 3)).astype(np.float32)
        np.testing.assert_allclose(grid_value(grid, p), p[:, 0] + 2 * p[:, 1] - .5 * p[:, 2], atol=1e-6)
        np.testing.assert_allclose(grid_value(grid, p, gradient=True), np.tile([1, 2, -.5], (len(p), 1)), atol=4e-6)
        nodes = np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]], np.float32)
        np.testing.assert_allclose(grid_value(grid, nodes), [-2.5, 2.5, 0], atol=1e-7)
        with self.assertRaises(ValueError):
            grid_value(grid, [[1.01, 0, 0]])

    def test_production_torch_interpolation(self):
        import importlib.util
        path = '/opt/data/private/yihengxu/Hunyuan3D/hy3dshape/hy3dshape/natex/sdf_condition.py'
        spec = importlib.util.spec_from_file_location('production_condition', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import torch
        rng = np.random.default_rng(2)
        grid = rng.normal(size=(13, 13, 13)).astype(np.float32)
        p = rng.uniform(-.99, .99, (1000, 3)).astype(np.float32)
        condition = mod.SDFCondition(grid)
        expected = condition.value(torch.from_numpy(p[None])).numpy()[0, :, 0]
        np.testing.assert_allclose(grid_value(grid, p), expected, atol=2e-6, rtol=2e-6)
        expected_gradient = condition.gradient(torch.from_numpy(p[None])).numpy()[0]
        np.testing.assert_allclose(grid_value(grid, p, gradient=True), expected_gradient, atol=1e-5, rtol=1e-5)

    def test_fail_closed_band(self):
        from alignment_fields import AlignmentField
        from collections import defaultdict
        field = AlignmentField.__new__(AlignmentField)
        field.grid = np.full((3, 3, 3), .03, np.float32)
        field.mode = 'B'
        field.band_width = .02
        field.counts = defaultdict(lambda: defaultdict(int))
        field.last_stage = None
        field.progress = None
        with self.assertRaises(ReconstructionError) as error:
            field.check_band(np.array([[0, 0, 0.]]))
        self.assertEqual(error.exception.code, 'field_outside_band')
        self.assertEqual(sum(r['outside_band'] for r in field.counts.values()), 1)

    def test_center_cache_batched_exact_match(self):
        from alignment_fields import AlignmentField
        from collections import defaultdict
        field = AlignmentField.__new__(AlignmentField)
        field.grid = np.zeros((3, 3, 3), np.float32)
        field.mode, field.band_width = 'B', .02
        field.counts = defaultdict(lambda: defaultdict(int))
        field.last_stage, field.progress = None, None
        field.center_points = np.random.default_rng(5).uniform(-1, 1, (140000, 3))
        field.center_values = np.tile([0., .1, .2], (140000, 1))
        field.edge_query = Mock(side_effect=AssertionError('Unexpected exact CAD query on cached centers'))
        for begin in range(0, 140000, 65536):
            result = field.evaluate(field.center_points[begin:begin + 65536].copy())
            np.testing.assert_array_equal(result.values, field.center_values[begin:begin + 65536])
        self.assertEqual(sum(r['center_cache_hits'] for r in field.counts.values()), 140000)
        bad = field.center_points[:4].copy()
        bad[-1, 0] += 1e-7
        with self.assertRaises(AssertionError):
            field.evaluate(bad)

    def test_vertex_default_explicit_identical(self):
        provider = Mock()
        topo = Mock(vertices=[])
        bands = Mock()
        self.assertEqual(localize_vertices(provider, bands, topo, Config()),
                         localize_vertices(provider, bands, topo, Config(feature_sdf_tolerance=.003)))


if __name__ == '__main__':
    unittest.main()
