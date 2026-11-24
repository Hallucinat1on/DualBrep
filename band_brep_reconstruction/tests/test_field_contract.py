import tempfile
import unittest
from pathlib import Path
import numpy as np

from bandbrep.config import Config, load_config
from bandbrep.edge_first import reconstruct
from bandbrep.field_provider import (
    CallableUDFProvider,
    FieldEvaluation,
    MeshFieldProvider,
    bounded_update,
    checked_values,
)
from bandbrep.schemas import ReconstructionError, SurfaceMesh
from bandbrep.surface_mesh import validate_mesh
from helpers import box, hollow_sphere


class ContractTests(unittest.TestCase):
    def test_edge_first_defaults_and_removed_config(self):
        cfg = Config().validate()
        self.assertEqual((cfg.coarse_spacing, cfg.fine_spacing, cfg.band_width, cfg.sample_timeout_seconds),
                         (0.024, 0.003, 0.02, 10800.0))
        self.assertEqual(cfg.to_dict(), load_config(Path(__file__).resolve().parents[1] / 'configs/default.json').to_dict())
        for obsolete in ('resolution', 'refinement_max_rounds', 'max_triangles', 'face_second_pass',
                         'curve_recovery', 'region_repair', 'save_queries', 'shell_radius', 'vertex_level',
                         'local_refine_levels'):
            with self.assertRaises(ValueError):
                load_config(overrides={obsolete: 1})
        for coarse in (True, 0.0, -0.01, float('nan'), 0.001):
            with self.assertRaises(ValueError):
                Config(coarse_spacing=coarse).validate()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            p.write_text('{"sample_timeout_seconds": 9}')
            self.assertEqual(load_config(p, {'sample_timeout_seconds': 10}).sample_timeout_seconds, 10)

    def test_both_reconstruction_inputs_reject_legacy_veudf(self):
        from unittest.mock import patch
        from bandbrep.cad_source import load_cad_source

        mesh, udf = box()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            with patch('veudf.schema.SCHEMA', 'edge_vertex_mesh_pools_v1'):
                with self.assertRaises(ReconstructionError) as mesh_error:
                    reconstruct(mesh, udf, Config(), out)
                with self.assertRaises(ReconstructionError) as cad_error:
                    load_cad_source(Path(tmp) / 'source.step', 'probe')
            self.assertEqual(mesh_error.exception.code, 'veudf_schema_mismatch')
            self.assertEqual(cad_error.exception.code, 'veudf_schema_mismatch')
            self.assertFalse(out.exists())

    def test_mesh_sdf_and_8d_offsets(self):
        mesh, udf = box()
        p = MeshFieldProvider(mesh, udf, Config())
        q = np.array([[0.551, 0.53, 0.52]])
        self.assertAlmostEqual(p.sdf(q)[0], 0.001)
        res = p.evaluate(q)
        self.assertEqual(res.values.shape, (1, 3))
        self.assertEqual(res.offsets.shape, (1, 2, 3))
        np.testing.assert_allclose(q[0] + res.offsets[0, 1], [0.55, 0.55, 0.55], atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(res.offsets[0, 1]), res.values[0, 2], atol=1e-10)
        self.assertEqual(checked_values(p, q).shape, (1, 3))

    def test_absent_masks_do_not_block_surface(self):
        mesh, udf = hollow_sphere()
        p = MeshFieldProvider(mesh, udf, Config())
        r = p.evaluate(mesh.vertices[:3])
        self.assertFalse(r.valid[:, 1:].any())
        self.assertTrue(np.isfinite(checked_values(p, mesh.vertices[:3])).all())

    def test_off_band_never_queries_udf_and_update_backtracks(self):
        mesh, udf = box()
        p = MeshFieldProvider(mesh, udf, Config())

        def forbidden(q):
            raise AssertionError('off-band model query')

        udf.fn = forbidden
        with self.assertRaises(ReconstructionError):
            checked_values(p, [[0, 0, 0]])
        start = np.array([[0.55, 0.1, 0.1]])
        q = bounded_update(p, start, start + [0.2, 0, 0])
        self.assertTrue(p.domain_mask(q).all())
        self.assertGreater(p.query_report['backtracked_steps'], 0)

    def test_wrong_shape_negative_nan_and_saturated_rejected(self):
        mesh, _ = box()
        # Reject legacy 2D output
        udf_2d = CallableUDFProvider(lambda q: np.ones((len(q), 2)), fingerprint='legacy2d', presence=('present', 'present'))
        with self.assertRaises(ReconstructionError):
            MeshFieldProvider(mesh, udf_2d, Config()).evaluate(mesh.vertices[:1])
        for value in (-0.1, float('nan')):
            udf = CallableUDFProvider(lambda q, v=value: np.full((len(q), 8), v), fingerprint='bad', presence=('present', 'present'))
            with self.assertRaises(ReconstructionError):
                checked_values(MeshFieldProvider(mesh, udf, Config()), mesh.vertices[:1])
        udf = CallableUDFProvider(
            lambda q: FieldEvaluation(np.ones((len(q), 2)), np.zeros((len(q), 2, 3)),
                                      np.ones((len(q), 2), bool), np.ones((len(q), 2), bool), np.zeros((len(q), 2))),
            fingerprint='clipped', presence=('present', 'present'),
        )
        with self.assertRaises(ReconstructionError):
            checked_values(MeshFieldProvider(mesh, udf, Config()), mesh.vertices[:1])

    def test_mesh_validation_rejects_bad_topology(self):
        mesh, _ = box()
        with self.assertRaises(ReconstructionError):
            validate_mesh(SurfaceMesh(mesh.vertices, mesh.triangles[:-1]), Config())
        bad = mesh.triangles.copy()
        bad[0] = bad[0][::-1]
        with self.assertRaises(ReconstructionError):
            validate_mesh(SurfaceMesh(mesh.vertices, bad), Config())
