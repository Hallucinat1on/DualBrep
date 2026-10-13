"""Geometry and output-contract regression tests; requires pythonocc + numpy/scipy."""
from dataclasses import replace
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                                 BRepPrimAPI_MakeTorus, BRepPrimAPI_MakeSphere)
from OCC.Core.Geom import Geom_Circle, Geom_BezierCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Extend.DataExchange import write_step_file

from evudf.cad import PreprocessError, bounds, collect_targets, compound, read_normalized, triangulate
from evudf.config import Config, load_config
from evudf.distance import EdgeOracle, SDFOracle, VertexOracle
from evudf.io import load_fields
from evudf.manifest import load_manifest, scan_directory
from evudf.pipeline import build_sample, verify_saved
from evudf.sampling import sample_all


SMALL = replace(Config(), query_count=80, surface_count=64, edge_count=64,
                min_face_samples=4, min_edge_samples=4, reference_checks=24)


class GeometryTests(unittest.TestCase):
    def test_cavity_sdf_sign(self):
        outer = BRepPrimAPI_MakeBox(gp_Pnt(-1.5, -1.5, -1.5), 3, 3, 3).Shape()
        inner = BRepPrimAPI_MakeBox(gp_Pnt(-.5, -.5, -.5), 1, 1, 1).Shape()
        solid = BRepAlgoAPI_Cut(outer, inner).Shape()
        targets = collect_targets(solid, SMALL)
        mesh = triangulate(targets, SMALL)
        values = SDFOracle(solid, targets.faces, mesh, SMALL).query([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        np.testing.assert_allclose(values, [.5, -.5, .5], atol=1e-6)

    def test_curved_edge_bvh_and_dense_upper_bound(self):
        controls = TColgp_Array1OfPnt(1, 4)
        for i, p in enumerate([(0, 0, 0), (2, 4, 0), (-2, 4, 0), (0, 0, 0)], 1):
            controls.SetValue(i, gp_Pnt(*p))
        curve = Geom_BezierCurve(controls)
        curved = BRepBuilderAPI_MakeEdge(curve).Edge()
        other = BRepBuilderAPI_MakeEdge(gp_Pnt(-3, 0, 1), gp_Pnt(3, 0, 1)).Edge()
        q = np.random.default_rng(10).uniform([-2, -.5, -.5], [2, 4, .5], (50, 3))
        actual = EdgeOracle([curved, other], SMALL).query(q)["distance"]
        ref = EdgeOracle([curved, other], SMALL, "cad_reference").query(q)["distance"]
        np.testing.assert_allclose(actual, ref, atol=1e-6)
        sample = np.array([[curve.Value(float(t)).X(), curve.Value(float(t)).Y(), curve.Value(float(t)).Z()]
                           for t in np.linspace(0, 1, 5001)])
        oracle = EdgeOracle([curved], SMALL).query(q)["distance"]
        upper = np.linalg.norm(q[:, None, :] - sample[None, :, :], axis=2).min(axis=1)
        self.assertTrue(np.all(oracle <= upper + 1e-6))
        self.assertLess(float(np.max(upper - oracle)), .001)

    def test_cube_counts_and_analytic_distances(self):
        shape = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
        targets = collect_targets(shape, SMALL)
        self.assertEqual((len(targets.edges), len(targets.vertices)), (12, 8))
        q = np.array([[.5, .5, .5], [.25, .25, .25], [2, .5, .5], [0, 0, 0]])
        expected_edge = np.array([np.sqrt(.5), np.sqrt(.125), np.sqrt(1.25), 0])
        expected_vertex = np.array([np.sqrt(.75), np.sqrt(.1875), np.sqrt(1.5), 0])
        np.testing.assert_allclose(EdgeOracle(targets.edges, SMALL).query(q)["distance"], expected_edge, atol=1e-6)
        np.testing.assert_allclose(VertexOracle(targets.vertices, SMALL).query(q)["distance"], expected_vertex, atol=1e-6)
        mesh = triangulate(targets, SMALL)
        np.testing.assert_allclose(SDFOracle(shape, targets.faces, mesh, SMALL).query(q), [-.5, -.25, 1, 0], atol=1e-6)

    def test_cylinder_retains_rings_removes_seam_and_anchors(self):
        targets = collect_targets(BRepPrimAPI_MakeCylinder(1, 2).Shape(), SMALL)
        self.assertEqual((len(targets.edges), len(targets.vertices)), (2, 0))
        self.assertTrue(all(e["vertex_ids"] == [] for e in targets.topology["edges"]))
        self.assertEqual(len(targets.topology["excluded_edges"]), 1)
        q = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [.5, .5, .1]])
        rho = np.linalg.norm(q[:, :2], axis=1)
        expected = np.minimum(np.hypot(rho - 1, q[:, 2]), np.hypot(rho - 1, q[:, 2] - 2))
        np.testing.assert_allclose(EdgeOracle(targets.edges, SMALL).query(q)["distance"], expected, atol=1e-6)
        # Mid-seam is not a target zero. Seam/ring intersection remains a zero.
        self.assertGreater(expected[0], .99)
        self.assertEqual(expected[1], 0)
        empty = VertexOracle(targets.vertices, SMALL).query(q)
        self.assertTrue(np.all(empty["nearest_id"] == -1))
        self.assertFalse(empty["gradient_valid"].any())

    def test_empty_edge_and_vertex_sets_on_torus(self):
        targets = collect_targets(BRepPrimAPI_MakeTorus(1, .25).Shape(), SMALL)
        self.assertEqual((len(targets.edges), len(targets.vertices)), (0, 0))
        result = EdgeOracle(targets.edges, SMALL).query(np.zeros((3, 3)))
        np.testing.assert_equal(result["distance"], SMALL.clip_edge)
        self.assertTrue(np.all(result["nearest_id"] == -1))

    def test_coplanar_interface_retained(self):
        box = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
        faces = collect_targets(box, SMALL).faces
        face = next(f for f in faces if abs(bounds(f)[0][1]) < 1e-6 and abs(bounds(f)[1][1]) < 1e-6)
        split = BRepFeat_SplitShape(box)
        split.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(.5, 0, 0), gp_Pnt(.5, 0, 1)).Edge(), face)
        split.Build()
        targets = collect_targets(split.Shape(), SMALL)
        self.assertEqual((len(targets.faces), len(targets.edges), len(targets.vertices)), (7, 15, 10))
        self.assertLess(EdgeOracle(targets.edges, SMALL).query([[.5, 0, .5]])["distance"][0], 1e-6)

    def test_finite_arc_and_endpoint_distances(self):
        circle = Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1)
        arc = BRepBuilderAPI_MakeEdge(circle, 0, np.pi / 2).Edge()
        line = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0)).Edge()
        self.assertAlmostEqual(EdgeOracle([arc], SMALL).query([[-1, 0, 0]])["distance"][0], np.sqrt(2), places=6)
        self.assertAlmostEqual(EdgeOracle([line], SMALL).query([[2, 0, 0]])["distance"][0], 1, places=6)
        axis_result = EdgeOracle([arc], SMALL).query([[0, 0, 1]])
        self.assertTrue(axis_result["tie"][0])
        self.assertFalse(axis_result["gradient_valid"][0])

    def test_bvh_matches_all_edge_reference(self):
        targets = collect_targets(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), SMALL)
        q = np.random.default_rng(123).uniform(-1, 2, (150, 3))
        accelerated = EdgeOracle(targets.edges, SMALL)
        actual = accelerated.query(q)
        ref = EdgeOracle(targets.edges, SMALL, "cad_reference")
        np.testing.assert_allclose(actual["distance"], ref.query(q)["distance"], atol=1e-7)
        self.assertLess(accelerated.calls, ref.calls)

    def test_unique_projection_gradient_and_ties(self):
        line = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0)).Edge()
        q = np.array([[.5, .2, 0], [.5, 0, 0]])
        result = EdgeOracle([line], SMALL).query(q)
        self.assertTrue(result["gradient_valid"][0])
        np.testing.assert_allclose(q[0] - result["distance"][0] * result["gradient"][0], [.5, 0, 0], atol=1e-7)
        self.assertFalse(result["gradient_valid"][1])
        vr = VertexOracle([[-1, 0, 0], [1, 0, 0]], SMALL).query([[0, 0, 0]])
        self.assertTrue(vr["tie"][0])
        self.assertFalse(vr["gradient_valid"][0])

    def test_degenerate_edges_rejected(self):
        with self.assertRaises(PreprocessError) as failure:
            collect_targets(BRepPrimAPI_MakeSphere(1).Shape(), SMALL)
        self.assertEqual(failure.exception.code, "degenerate_edge")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def step(self, name, shape):
        path = self.root / (name + ".step")
        write_step_file(shape, str(path))
        return path

    def test_normalization_roundtrip_and_short_edge_coverage(self):
        path = self.step("thin", BRepPrimAPI_MakeBox(gp_Pnt(10, 20, 30), 1, 1, .001).Shape())
        solid, transform = read_normalized(path, SMALL)
        targets = collect_targets(solid, SMALL)
        self.assertEqual(len(targets.edges), 12)
        p = np.array([10.2, 20.4, 30.0005, 1])
        np.testing.assert_allclose(np.array(transform["normalized_to_world"]) @ np.array(transform["world_to_normalized"]) @ p, p)
        arrays = sample_all(targets, triangulate(targets, SMALL), SMALL, np.random.default_rng(2))
        self.assertTrue(np.all(np.bincount(arrays["edge_point_id"], minlength=12) >= SMALL.min_edge_samples))

    def test_three_channels_and_resume_integrity(self):
        source = self.step("box", BRepPrimAPI_MakeBox(1, 1, 1).Shape())
        dest = self.root / "output"
        build_sample(source, dest, SMALL)
        arrays, metadata = load_fields(dest)
        self.assertEqual(arrays["query_points"].shape, arrays["query_fields"].shape)
        self.assertEqual(metadata["channel_names"], ["sdf", "dE", "dV"])
        self.assertEqual(build_sample(source, dest, SMALL, resume=True)["status"], "resumed")
        with self.assertRaises(PreprocessError):
            build_sample(source, dest, replace(SMALL, seed=1), resume=True)
        with (dest / "fields.npz").open("ab") as file:
            file.write(b"corruption")
        with self.assertRaises(PreprocessError):
            verify_saved(dest)

    def test_empty_vertex_output_and_gzip(self):
        source = self.step("cylinder", BRepPrimAPI_MakeCylinder(1, 2).Shape())
        compressed = source.with_suffix(".step.gz")
        with gzip.open(compressed, "wb") as out:
            out.write(source.read_bytes())
        dest = self.root / "cylinder-output"
        build_sample(compressed, dest, SMALL)
        arrays, meta = load_fields(dest, normalize=True)
        self.assertEqual(arrays["vertex_points"].shape, (0, 3))
        self.assertFalse(meta["has_vertices"])
        self.assertTrue(np.all(arrays["query_fields"][:, 2] == 1))
        self.assertFalse(arrays["gradient_valid_V"].any())

    def test_empty_both_fields_output(self):
        source = self.step("torus", BRepPrimAPI_MakeTorus(1, .25).Shape())
        build_sample(source, self.root / "torus-output", SMALL)
        arrays, _ = load_fields(self.root / "torus-output", normalize=True)
        self.assertTrue(np.all(arrays["query_fields"][:, 1:] == 1))
        self.assertFalse(bool(arrays["has_edges"]))

    def test_multiple_solids_rejected(self):
        source = self.step("multiple", compound([BRepPrimAPI_MakeBox(1, 1, 1).Shape(),
                                                BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 1, 1, 1).Shape()]))
        with self.assertRaises(PreprocessError) as failure:
            build_sample(source, self.root / "multiple-output", SMALL)
        self.assertEqual(failure.exception.code, "unsupported_multi_solid")
        self.assertFalse((self.root / "multiple-output").exists())

    def test_overwrite_refuses_unrelated_directory(self):
        source = self.step("box", BRepPrimAPI_MakeBox(1, 1, 1).Shape())
        dest = self.root / "unrelated"
        dest.mkdir()
        marker = dest / "keep.txt"
        marker.write_text("user data")
        with self.assertRaises(PreprocessError):
            build_sample(source, dest, SMALL, overwrite=True)
        self.assertEqual(marker.read_text(), "user data")

    def test_portable_cli_batch_and_resume(self):
        self.step("box", BRepPrimAPI_MakeBox(1, 1, 1).Shape())
        self.step("cylinder", BRepPrimAPI_MakeCylinder(1, 2).Shape())
        portable = self.root / "portable"
        shutil.copytree(Path(__file__).resolve().parents[1], portable,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        cmd = [sys.executable, str(portable / "preprocess.py"), "--step-dir", str(self.root),
               "--out-dir", str(self.root / "batch"), "--config", str(portable / "configs/smoke.json"), "--workers", "2"]
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(list((self.root / "batch").glob("*/fields.npz"))), 2)
        result = subprocess.run(cmd + ["--resume"], cwd=self.root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('"status": "resumed"'), 2)

    def test_cli_timeout_records_failure_without_final_sample(self):
        source = self.step("box", BRepPrimAPI_MakeBox(1, 1, 1).Shape())
        script = Path(__file__).resolve().parents[1] / "preprocess.py"
        dest = self.root / "timeout-output"
        result = subprocess.run([sys.executable, str(script), "--step", str(source), "--out", str(dest),
                                 "--timeout", "0.0001"], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('"code": "timeout"', result.stdout)
        self.assertFalse(dest.exists())

    @unittest.skipUnless(importlib.util.find_spec("igl"), "Optional libigl backend is not installed")
    def test_mesh_sdf_pipeline(self):
        source = self.step("cylinder", BRepPrimAPI_MakeCylinder(1, 2).Shape())
        cfg = replace(SMALL, sdf_backend="mesh")
        dest = self.root / "mesh-output"
        build_sample(source, dest, cfg)
        meta = verify_saved(dest)
        self.assertEqual(meta["config"]["sdf_backend"], "mesh")
        quality = json.loads((dest / "quality.json").read_text())
        self.assertLess(quality["reference_max_error"]["sdf"], 2 * cfg.mesh_deflection + cfg.label_tolerance)

    @unittest.skipUnless(importlib.util.find_spec("igl"), "Optional libigl backend is not installed")
    def test_fast_mesh_sdf_pipeline(self):
        source = self.step("cylinder", BRepPrimAPI_MakeCylinder(1, 2).Shape())
        dest = self.root / "fast-mesh-output"
        build_sample(source, dest, replace(SMALL, sdf_backend="mesh_fast"))
        meta = verify_saved(dest)
        self.assertEqual(meta["config"]["sdf_backend"], "mesh_fast")

    @unittest.skipUnless(importlib.util.find_spec("igl"), "Optional libigl backend is not installed")
    def test_fast_mesh_sdf_cavity_and_reversed_mesh(self):
        outer = BRepPrimAPI_MakeBox(gp_Pnt(-1.5, -1.5, -1.5), 3, 3, 3).Shape()
        inner = BRepPrimAPI_MakeBox(gp_Pnt(-.5, -.5, -.5), 1, 1, 1).Shape()
        shape = BRepAlgoAPI_Cut(outer, inner).Shape()
        targets = collect_targets(shape, SMALL)
        mesh = triangulate(targets, SMALL)
        cfg = replace(SMALL, sdf_backend="mesh_fast")
        result = SDFOracle(shape, targets.faces, mesh, cfg).query([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        np.testing.assert_allclose(result, [.5, -.5, .5], atol=1e-6)
        with self.assertRaises(PreprocessError):
            SDFOracle(shape, targets.faces, (mesh[0], mesh[1][:, ::-1], mesh[2]), cfg)


class ManifestTests(unittest.TestCase):
    def test_abc_group_union_for_duplicates_and_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = "a" * 24, "b" * 24
            for name, body in [(f"00000001_{a}_step_000.step", b"shared"),
                               (f"00000002_{a}_step_001.step", b"a-part"),
                               (f"00000003_{b}_step_000.step", b"shared"),
                               (f"00000004_{b}_step_001.step", b"b-part")]:
                (root / name).write_bytes(body)
            records = scan_directory(root)
            self.assertEqual(len({r["group_id"] for r in records}), 1)
            self.assertEqual(len({r["split"] for r in records}), 1)
            self.assertEqual(sum("duplicate_of" in r for r in records), 1)

    def test_manifest_paths_and_unsafe_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "manifest.jsonl"
            file.write_text(json.dumps(dict(sample_id="one", step_path="input.step")) + "\n")
            self.assertEqual(load_manifest(file)[0]["step_path"], str(Path(tmp) / "input.step"))
            file.write_text(json.dumps(dict(sample_id="../escape", step_path="input.step")) + "\n")
            with self.assertRaises(ValueError):
                load_manifest(file)

    def test_invalid_config_fails_early(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"query_weights": [0, 0, 0, 0, 0]})
        with self.assertRaises(ValueError):
            load_config(overrides={"batch_size": 0})
        with self.assertRaises(ValueError):
            EdgeOracle([], SMALL, "unknown")
        with self.assertRaises(ValueError):
            SDFOracle(None, [], None, SMALL, "unknown")


if __name__ == "__main__":
    unittest.main()
