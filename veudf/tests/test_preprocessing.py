"""CAD-side regression tests (pythonocc + libigl + trimesh + networkx):

python -m pytest veudf/tests -q
"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import networkx as nx
import numpy as np
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeTorus,
)
from OCC.Core.Geom import Geom_Circle
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.DataExchange import write_step_file

from veudf.schema import EMPTY_DISTANCE, SCHEMA as SCHEMA_VERSION
from veudf.preprocess.cad import (
    PreprocessError,
    compound,
    collect_targets,
    read_normalized,
    triangulate,
)
from veudf.preprocess.condition import weld
from veudf.preprocess.config import Config, load_config
from veudf.preprocess.dedup import (
    Deduplicator,
    bounded_isomorphic,
    features,
    sample_mesh,
)
from veudf.preprocess.distance import EdgeOracle, VertexOracle
from veudf.preprocess.filtering import audit_shape, check_metrics, face_edge_counts
from veudf.preprocess.manifest import load_manifest, scan_directory
from veudf.preprocess.pipeline import build_asset, verify_asset
from veudf.preprocess.sampling import sample_edges, sample_pools
from veudf.utils import allocate, array_hash
from veudf.geometry.sampling import base_surface
from veudf.geometry.mesh import MeshSDFOracle
from veudf.io.condition import read_geometry

ROOT = Path(__file__).resolve().parents[1]
SMALL = replace(
    Config(),
    geometry_surface_pool_count=1024,
    pool_count=4096,
    feature_min_samples=2,
    evaluation_points_per_pool=64,
    reference_checks=12,
    dedup_surface_points=256,
    dedup_edge_points=64,
    dedup_edge_pool_count=128,
)

HALF = 0.9  # a cube is normalized to [-0.9, 0.9]^3


def write_step(root, name, shape):
    path = Path(root) / f"{name}.step"
    write_step_file(shape, str(path))
    return path


def record(path, uid, split="train", group=None):
    return dict(sample_id=uid, step_path=str(path), group_id=group or uid, split=split)


class DistanceTests(unittest.TestCase):
    def test_cube_counts_and_analytic_distances(self):
        shape = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
        targets = collect_targets(shape, SMALL)
        self.assertEqual((len(targets.edges), len(targets.vertices)), (12, 8))
        q = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [2, 0.5, 0.5], [0, 0, 0]])
        np.testing.assert_allclose(
            EdgeOracle(targets.edges, SMALL).query(q)["distance"],
            [np.sqrt(0.5), np.sqrt(0.125), np.sqrt(1.25), 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            VertexOracle(targets.vertices, SMALL).query(q)["distance"],
            [np.sqrt(0.75), np.sqrt(0.1875), np.sqrt(1.5), 0],
            atol=1e-6,
        )
        mesh = triangulate(targets, SMALL)
        for oracle in [
            MeshSDFOracle(*weld(mesh, SMALL)[0]),
        ]:
            np.testing.assert_allclose(oracle.query(q), [-0.5, -0.25, 1, 0], atol=1e-6)

    def test_bvh_matches_reference_and_prunes(self):
        targets = collect_targets(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), SMALL)
        q = np.random.default_rng(123).uniform(-1, 2, (150, 3))
        fast, ref = EdgeOracle(targets.edges, SMALL), EdgeOracle(
            targets.edges, SMALL, "cad_reference"
        )
        np.testing.assert_allclose(
            fast.query(q)["distance"], ref.query(q)["distance"], atol=1e-7
        )
        self.assertLess(fast.calls, ref.calls)

    def test_cylinder_rings_seam_and_empty_vertices(self):
        targets = collect_targets(BRepPrimAPI_MakeCylinder(1, 2).Shape(), SMALL)
        self.assertEqual((len(targets.edges), len(targets.vertices)), (2, 0))
        q = np.array([[1, 0, 1], [1, 0, 0], [0.5, 0.5, 0.1]])
        rho = np.linalg.norm(q[:, :2], axis=1)
        expected = np.minimum(
            np.hypot(rho - 1, q[:, 2]), np.hypot(rho - 1, q[:, 2] - 2)
        )
        np.testing.assert_allclose(
            EdgeOracle(targets.edges, SMALL).query(q)["distance"], expected, atol=1e-6
        )
        empty = VertexOracle(targets.vertices, SMALL).query(q)
        self.assertTrue(
            np.all(empty["distance"] == EMPTY_DISTANCE)
            and np.all(empty["nearest_id"] == -1)
        )

    def test_arc_axis_is_a_tie(self):
        arc = BRepBuilderAPI_MakeEdge(
            Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1), 0, np.pi / 2
        ).Edge()
        result = EdgeOracle([arc], SMALL).query([[0, 0, 1], [-1, 0, 0]])
        self.assertTrue(result["tie"][0])
        self.assertAlmostEqual(result["distance"][1], np.sqrt(2), places=6)

    def test_cavity_sdf_sign(self):
        solid = BRepAlgoAPI_Cut(
            BRepPrimAPI_MakeBox(gp_Pnt(-1.5, -1.5, -1.5), 3, 3, 3).Shape(),
            BRepPrimAPI_MakeBox(gp_Pnt(-0.5, -0.5, -0.5), 1, 1, 1).Shape(),
        ).Shape()
        targets = collect_targets(solid, SMALL)
        mesh = triangulate(targets, SMALL)
        closed, oracle = weld(mesh, SMALL)
        np.testing.assert_allclose(
            oracle.query([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), [0.5, -0.5, 0.5], atol=1e-6
        )
        with self.assertRaises(ValueError):
            MeshSDFOracle(closed[0], closed[1][:, ::-1])


class FilterTests(unittest.TestCase):
    def test_multiple_solids_choose_largest_without_other_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = BRepPrimAPI_MakeBox(gp_Pnt(10, 0, 0), 1, 1, 1).Shape()
            large = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2, 3, 4).Shape()
            p = write_step(tmp, "multiple", compound([small, large]))
            solid, meta = read_normalized(p, SMALL)
            self.assertEqual(meta["source_solid_count"], 2)
            self.assertEqual(
                meta["selected_solid_policy"], "largest_valid_positive_volume"
            )
            self.assertAlmostEqual(meta["source_volume"], 24.0, places=6)
            np.testing.assert_allclose(
                meta["source_bounds"], [[0, 0, 0], [2, 3, 4]], atol=1e-6
            )
            self.assertEqual(len(collect_targets(solid, SMALL).faces), 6)
            _, again = read_normalized(p, SMALL)
            self.assertEqual(
                meta["selected_solid_index"], again["selected_solid_index"]
            )

    def test_face_count_edge_and_length_endpoints(self):
        cfg = Config()
        self.assertEqual(cfg.min_faces, 2)
        for faces in (2, 100):
            check_metrics([1] * faces, [0.001, 1.0], cfg)
        for faces, code in ((1, "face_count_below_min"), (101, "face_count_above_max")):
            with self.assertRaises(PreprocessError) as caught:
                check_metrics([1] * faces, [1.0], cfg)
            self.assertEqual(caught.exception.code, code)
        with self.assertRaises(PreprocessError) as caught:
            check_metrics([51, 1], [1.0], cfg)
        self.assertEqual(caught.exception.code, "face_edge_count_exceeded")
        with self.assertRaises(PreprocessError) as caught:
            check_metrics([50, 1], [0.000999, 1.0], cfg)
        self.assertEqual(caught.exception.code, "edge_too_short")

    def test_seam_counts_for_complexity_but_not_targets(self):
        solid = BRepPrimAPI_MakeCylinder(1, 2).Shape()
        targets = collect_targets(solid, SMALL)
        self.assertEqual(
            max(face_edge_counts(targets.topology)), 3
        )  # side face: two rings + seam
        report = audit_shape(solid, targets, SMALL)
        self.assertEqual(report["face_count"], 3)
        self.assertAlmostEqual(
            report["min_source_edge_length"], 2.0, places=6
        )  # seam = height

    def test_scale_invariant_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            lengths = []
            for scale in (1.0, 100.0):
                solid, _ = read_normalized(
                    write_step(
                        tmp,
                        str(scale),
                        BRepPrimAPI_MakeBox(scale, scale, scale).Shape(),
                    ),
                    SMALL,
                )
                lengths.append(
                    audit_shape(solid, collect_targets(solid, SMALL), SMALL)[
                        "min_source_edge_length"
                    ]
                )
            self.assertAlmostEqual(lengths[0], lengths[1], places=5)


class ConditionTests(unittest.TestCase):
    def test_mesh_sign_surface_normals_and_gradient_on_cube(self):
        with tempfile.TemporaryDirectory() as tmp:
            solid, _ = read_normalized(
                write_step(tmp, "box", BRepPrimAPI_MakeBox(1, 1, 1).Shape()), SMALL
            )
        mesh, oracle = weld(triangulate(collect_targets(solid, SMALL), SMALL), SMALL)
        q = np.random.default_rng(42).uniform(-1, 1, (512, 3))
        delta = np.abs(q) - np.abs(mesh[0]).max(0)
        exact = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(
            delta.max(-1), 0
        )
        np.testing.assert_allclose(oracle.query(q), exact, atol=1e-10)
        surface, normal = base_surface(mesh, 512)
        self.assertLess(np.abs(oracle.query(surface)).max(), 1e-7)
        axis = np.abs(surface).argmax(1)
        np.testing.assert_allclose(
            np.abs(normal[np.arange(len(surface)), axis]), 1, atol=1e-7
        )
        g, valid = oracle.gradient(
            [
                [0.5, 0.1, 0.2],
                [0, 0, 0],
                [mesh[0][:, 0].max(), mesh[0][:, 1].max(), 0.1],
            ]
        )
        np.testing.assert_allclose(g[0], [1, 0, 0], atol=1e-6)
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertFalse(valid[2])
        with self.assertRaises(ValueError):
            MeshSDFOracle(mesh[0], mesh[1][:, ::-1])
        with self.assertRaises(ValueError):
            MeshSDFOracle(mesh[0], mesh[1][:-1])


class SamplingTests(unittest.TestCase):
    def test_profile_counts_and_strict_old_keys(self):
        cfg = load_config(ROOT / "configs/surface_pools.json")
        np.testing.assert_array_equal(
            allocate(cfg.pool_count, cfg.pool_weights), [393216, 393216, 235930]
        )
        self.assertEqual(cfg.geometry_surface_pool_count, 393216)
        with self.assertRaises(ValueError):
            load_config(overrides={"query_count": 100})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_cube_pools_labels_and_independent_evaluation(self):
        p = write_step(self.root, "box", BRepPrimAPI_MakeBox(1, 1, 1).Shape())
        out = self.root / "cube"
        from unittest.mock import patch

        with patch(
            "veudf.geometry.mesh.MeshSDFOracle.query",
            side_effect=AssertionError("Surface labels must not query SDF"),
        ):
            result = build_asset(record(p, "cube"), out, SMALL)
        verify_asset(out, result["hashes"])
        from veudf.io.dataset import load_surface_pool

        bank = load_surface_pool(out, "surface")
        self.assertIsInstance(bank["fields"], np.memmap)
        with self.assertRaises(ValueError):
            load_surface_pool(out, "near_surface")
        oracle, _, _, _ = read_geometry(out / "condition")
        solid, _ = read_normalized(p, SMALL)
        targets = collect_targets(solid, SMALL)
        from veudf.schema import POOLS, VALID

        self.assertEqual(set(x.name for x in (out / "pools").iterdir()), set(POOLS))
        for parent in ("pools", "evaluation"):
            for name in POOLS:
                bank = out / parent / name
                x = np.load(bank / "points.npy")
                f = np.load(bank / "fields.npy")
                self.assertEqual(f.shape, (len(x), 2))
                ids = np.load(bank / "triangle_ids.npy")
                uv = np.load(bank / "barycentric.npy")
                tri = oracle.vertices[oracle.faces[ids]]
                reconstructed = (
                    tri[:, 0]
                    + uv[:, :1] * (tri[:, 1] - tri[:, 0])
                    + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
                )
                np.testing.assert_allclose(x, reconstructed, atol=1e-7)
                for c, r in enumerate(
                    (
                        EdgeOracle(targets.edges, SMALL, "cad_reference"),
                        VertexOracle(targets.vertices, SMALL),
                    )
                ):
                    np.testing.assert_allclose(
                        f[:, c], r.query(x)["distance"], atol=1e-6
                    )
                self.assertEqual(np.load(bank / "flags.npy").shape, f.shape)
                self.assertTrue(np.all(np.load(bank / "flags.npy") & VALID))
        train = np.load(out / "pools/surface/points.npy")
        evaluation = np.load(out / "evaluation/surface/points.npy")
        seen = {tuple(x) for x in train}
        self.assertFalse(any(tuple(x) in seen for x in evaluation))
        meta = json.loads((out / "meta.json").read_text())
        self.assertEqual(
            sum(v["count"] for v in meta["statistics"].values()), SMALL.pool_count
        )
        self.assertEqual(meta["channel_names"], ["dE", "dV"])
        self.assertEqual(meta["schema"], "edge_vertex_surface_udf_v1")
        self.assertEqual(meta["distance_engine"]["occ_calls"], 0)
        self.assertNotIn("band_max", meta)

    def test_cylinder_empty_vertices_and_geometry_stability(self):
        p = write_step(self.root, "cyl", BRepPrimAPI_MakeCylinder(1, 2).Shape())
        build_asset(record(p, "base"), self.root / "base", SMALL)
        cfg = replace(SMALL, pool_count=8192)
        build_asset(record(p, "grown"), self.root / "grown", cfg)
        for name in ["base", "grown"]:
            bank = self.root / name / "pools/vertex_surface/fields.npy"
            self.assertEqual(len(np.load(bank)), 0)
            f = np.load(self.root / name / "pools/surface/fields.npy")
            self.assertTrue(np.all(f[:, 1] == np.float32(EMPTY_DISTANCE)))
        for array in ["points", "normals"]:
            a = np.load(self.root / "base/pools/surface" / f"{array}.npy")[
                : SMALL.geometry_surface_pool_count
            ]
            b = np.load(self.root / "grown/pools/surface" / f"{array}.npy")[
                : SMALL.geometry_surface_pool_count
            ]
            np.testing.assert_array_equal(a, b)

    def test_single_face_rejected_and_nothing_written(self):
        step = write_step(self.root, "torus", BRepPrimAPI_MakeTorus(1, 0.25).Shape())
        with self.assertRaises(PreprocessError) as caught:
            build_asset(record(step, "torus"), self.root / "torus", SMALL)
        self.assertEqual(caught.exception.code, "face_count_below_min")
        self.assertEqual(list(self.root.glob("*torus*")), [step])

    def test_driver_builds_dedups_publishes_and_resumes(self):
        steps = dict(
            a=write_step(self.root, "a", BRepPrimAPI_MakeBox(1, 1, 1).Shape()),
            b=write_step(
                self.root, "b", BRepPrimAPI_MakeBox(gp_Pnt(5, 5, 5), 2, 2, 2).Shape()
            ),
            c=write_step(self.root, "c", BRepPrimAPI_MakeCylinder(1, 2).Shape()),
            t=write_step(self.root, "t", BRepPrimAPI_MakeTorus(1, 0.25).Shape()),
        )
        manifest = self.root / "manifest.jsonl"
        rows = [
            record(steps["a"], "a", "test"),
            record(steps["b"], "b", "test"),
            record(steps["c"], "c", "train"),
            record(steps["t"], "t", "train"),
        ]
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        config = self.root / "config.json"
        config.write_text(json.dumps(SMALL.to_dict()))
        data = self.root / "dataset"
        cmd = [
            sys.executable,
            "-m",
            "veudf.cli.build_dataset",
            "build",
            "--root",
            str(data),
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--workers",
            "2",
            "--threads",
            "1",
        ]
        first = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        report = json.loads((data / "report.json").read_text())
        index = json.loads((data / "index.json").read_text())
        self.assertEqual(index["schema"], SCHEMA_VERSION)
        self.assertEqual(sorted(a["asset_id"] for a in index["assets"]), ["a", "c"])
        self.assertEqual(report["dedup"], {"near_duplicate": 1})
        self.assertEqual(report["failures"], {"face_count_below_min": 1})
        stamp = (data / "assets/a/meta.json").stat().st_mtime_ns
        second = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual((data / "assets/a/meta.json").stat().st_mtime_ns, stamp)
        config.write_text(json.dumps(replace(SMALL, seed=1).to_dict()))
        changed = self.root / "dataset"
        (changed / "config.json").write_text(
            json.dumps(replace(SMALL, seed=1).to_dict())
        )
        refused = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(refused.returncode, 0)


class DedupTests(unittest.TestCase):
    @staticmethod
    def topology(vertex=True):
        return dict(
            source_faces=[dict(source_face_id=0), dict(source_face_id=1)],
            source_edges=[dict(curve_type=0)],
            vertices=[dict(vertex_id=0)] if vertex else [],
            edges=[
                dict(
                    edge_id=0,
                    source_edge_ids=[0],
                    face_ids=[0, 1],
                    vertex_ids=[0] if vertex else [],
                    closed=False,
                    self_loop=False,
                )
            ],
        )

    def test_duplicate_and_cross_split_conflict(self):
        f = dict(
            dedup_surface=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            dedup_edge=np.array([[0.0, 0, 0], [1, 0, 0]]),
            dedup_vertex=np.array([[0.0, 0, 0]]),
        )
        d = Deduplicator(SMALL)
        self.assertEqual(
            d.consider("test", "test", self.topology(), f)["status"], "passed"
        )
        self.assertEqual(
            d.consider("train", "train", self.topology(), f)["code"], "near_duplicate"
        )
        other = d.consider(
            "topo",
            "train",
            self.topology(False),
            dict(f, dedup_vertex=np.zeros((0, 3))),
        )
        self.assertEqual(other["code"], "near_duplicate_split_conflict")
        moved = {k: v + 0.1 for k, v in f.items()}
        self.assertEqual(
            d.consider("moved", "train", self.topology(), moved)["status"], "passed"
        )

    def test_mesh_sampling_is_order_invariant(self):
        v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        f = np.array([[0, 1, 2], [1, 3, 2]])
        np.testing.assert_array_equal(
            sample_mesh(v, f, 100), sample_mesh(v, f[::-1, ::-1], 100)
        )

    def test_features_do_not_depend_on_supervision_pool_sizes(self):
        targets = collect_targets(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), SMALL)
        mesh = triangulate(targets, SMALL)
        a = features(mesh, targets, SMALL)
        b = features(mesh, targets, replace(SMALL, pool_count=8192))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_symmetric_graph_search_has_a_deadline(self):
        a = nx.disjoint_union_all([nx.cycle_graph(3)] * 12)
        b = nx.disjoint_union_all([nx.cycle_graph(3)] * 10 + [nx.cycle_graph(6)])
        nx.set_node_attributes(a, "face", "kind")
        nx.set_node_attributes(b, "face", "kind")
        start = time.monotonic()
        self.assertEqual(bounded_isomorphic(a, b, 0.02), (None, "inconclusive_timeout"))
        self.assertLess(time.monotonic() - start, 1)


class ManifestConfigTests(unittest.TestCase):
    def test_abc_group_union_for_duplicates_and_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = "a" * 24, "b" * 24
            for name, body in [
                (f"00000001_{a}_step_000.step", b"shared"),
                (f"00000002_{a}_step_001.step", b"a-part"),
                (f"00000003_{b}_step_000.step", b"shared"),
                (f"00000004_{b}_step_001.step", b"b-part"),
            ]:
                (Path(tmp) / name).write_bytes(body)
            rows = scan_directory(tmp)
            self.assertEqual(len({r["group_id"] for r in rows}), 1)
            self.assertEqual(sum("duplicate_of" in r for r in rows), 1)

    def test_manifest_rejects_unsafe_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "manifest.jsonl"
            file.write_text(
                json.dumps(dict(sample_id="../escape", step_path="x.step")) + "\n"
            )
            with self.assertRaises(ValueError):
                load_manifest(file)

    def test_invalid_config_fails_early(self):
        for overrides in (
            {"pool_weights": [0, 0, 0, 0]},
            {"min_faces": 0},
            {"band_max": 0.2},
            {"condition_resolution": 4},
            {"uniform_weight": 1},
        ):
            with self.assertRaises(ValueError):
                load_config(overrides=overrides)


if __name__ == "__main__":
    unittest.main()
