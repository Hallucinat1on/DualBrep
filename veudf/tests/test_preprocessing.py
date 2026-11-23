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
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeTorus,
)
from OCC.Core.Geom import Geom_Circle
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.DataExchange import write_step_file

from veudf.schema import (
    EMPTY_DISTANCE,
    SCHEMA as SCHEMA_VERSION,
    POOLS,
    ALL_POOLS,
    SURFACE,
    SPATIAL,
)
from veudf.preprocess.cad import (
    Targets,
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
from veudf.preprocess.reference import (
    ReferenceEdgeOracle,
    CADSDFOracle,
    ReferenceVertexOracle,
)
from veudf.preprocess.distance import FeatureUDFOracle
from veudf.preprocess.filtering import audit_shape, check_metrics, face_edge_counts
from veudf.preprocess.manifest import load_manifest, scan_directory
from veudf.preprocess.pipeline import build_asset, verify_asset
from veudf.preprocess.sampling import (
    FEATURE_3D_PERTURBATION_MODE,
    sample_edges,
    sample_pools,
    zero_pools,
)
from veudf.utils import allocate, array_hash
from veudf.geometry.sampling import base_surface
from veudf.geometry.mesh import MeshSDFOracle
from veudf.io.condition import read_geometry

ROOT = Path(__file__).resolve().parents[1]
SMALL = replace(
    Config(),
    geometry_surface_pool_count=1024,
    pool_count=4608,
    feature_min_samples=2,
    evaluation_points_per_pool=64,
    reference_checks=12,
    zero_edge_count=256,
    evaluation_zero_edge_count=64,
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
    def test_feature_neighborhoods_match_clipped_independent_truth(self):
        targets = collect_targets(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), SMALL)
        q = np.concatenate(
            (
                np.random.default_rng(91).uniform(-0.1, 1.1, (512, 3)),
                np.array(
                    [
                        [0, 0, 0],
                        [-0.049999, 0, 0],
                        [-0.05, 0, 0],
                        [-0.050001, 0, 0],
                        [0.02, 0.02, 0],
                        [0.5, 0.5, 0.5],
                    ]
                ),
            )
        )
        feature = FeatureUDFOracle(targets, SMALL)
        for channel, reference in zip(
            feature.query(q),
            (
                ReferenceEdgeOracle(targets.edges, SMALL, exhaustive=True),
                ReferenceVertexOracle(targets.vertices, SMALL),
            ),
        ):
            truth = reference.query(q)
            expected = np.minimum(truth["distance"], SMALL.udf_truncation)
            np.testing.assert_allclose(channel["distance"], expected, atol=1e-8, rtol=0)
            active = ~channel["saturated"] & ~truth["tie"]
            np.testing.assert_array_equal(
                channel["nearest_id"][active], truth["nearest_id"][active]
            )
            self.assertTrue(np.all(channel["nearest_id"][channel["saturated"]] == -1))
        full = feature.query(q)
        parts = [feature.query(part) for part in np.array_split(q, 7)]
        for c in (0, 1):
            for key in ("distance", "nearest_id", "tie", "saturated"):
                np.testing.assert_array_equal(
                    full[c][key], np.concatenate([p[c][key] for p in parts])
                )

    def test_feature_arc_endcaps_periodic_axis_and_empty_targets(self):
        arc = BRepBuilderAPI_MakeEdge(
            Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.02),
            1.75 * np.pi,
            2.25 * np.pi,
        ).Edge()
        curve = __import__(
            "OCC.Core.BRepAdaptor", fromlist=["BRepAdaptor_Curve"]
        ).BRepAdaptor_Curve(arc)
        vertex = np.array(
            [
                [
                    curve.Value(curve.FirstParameter()).X(),
                    curve.Value(curve.FirstParameter()).Y(),
                    0,
                ]
            ]
        )
        targets = Targets([arc], vertex, [], {})
        q = np.array([[0, 0, 0.01], [0.02, 0, 0], [-0.02, 0, 0], [0.1, 0, 0]])
        edge, vert = FeatureUDFOracle(targets, SMALL).query(q)
        for actual, reference in (
            (edge, ReferenceEdgeOracle([arc], SMALL, exhaustive=True)),
            (vert, ReferenceVertexOracle(vertex, SMALL)),
        ):
            np.testing.assert_allclose(
                actual["distance"],
                np.minimum(reference.query(q)["distance"], 0.05),
                atol=1e-8,
            )
        self.assertTrue(edge["tie"][0])
        empty = FeatureUDFOracle(Targets([], np.empty((0, 3)), [], {}), SMALL).query(q)
        for channel in empty:
            self.assertTrue(np.all(channel["distance"] == SMALL.udf_truncation))
            self.assertFalse(channel["saturated"].any())
            self.assertTrue(np.all(channel["nearest_id"] == -1))

    def test_cube_counts_and_analytic_distances(self):
        shape = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
        targets = collect_targets(shape, SMALL)
        self.assertEqual((len(targets.edges), len(targets.vertices)), (12, 8))
        q = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [2, 0.5, 0.5], [0, 0, 0]])
        np.testing.assert_allclose(
            ReferenceEdgeOracle(targets.edges, SMALL).query(q)["distance"],
            [np.sqrt(0.5), np.sqrt(0.125), np.sqrt(1.25), 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            ReferenceVertexOracle(targets.vertices, SMALL).query(q)["distance"],
            [np.sqrt(0.75), np.sqrt(0.1875), np.sqrt(1.5), 0],
            atol=1e-6,
        )
        mesh = triangulate(targets, SMALL)
        for oracle in [
            CADSDFOracle(shape, targets.faces, SMALL),
            MeshSDFOracle(*weld(mesh, SMALL)[0]),
        ]:
            np.testing.assert_allclose(oracle.query(q), [-0.5, -0.25, 1, 0], atol=1e-6)

    def test_bvh_matches_reference_and_prunes(self):
        targets = collect_targets(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), SMALL)
        q = np.random.default_rng(123).uniform(-1, 2, (150, 3))
        fast, ref = ReferenceEdgeOracle(targets.edges, SMALL), ReferenceEdgeOracle(
            targets.edges, SMALL, exhaustive=True
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
            ReferenceEdgeOracle(targets.edges, SMALL).query(q)["distance"],
            expected,
            atol=1e-6,
        )
        empty = ReferenceVertexOracle(targets.vertices, SMALL).query(q)
        self.assertTrue(
            np.all(empty["distance"] == EMPTY_DISTANCE)
            and np.all(empty["nearest_id"] == -1)
        )

    def test_arc_axis_is_a_tie(self):
        arc = BRepBuilderAPI_MakeEdge(
            Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1), 0, np.pi / 2
        ).Edge()
        result = ReferenceEdgeOracle([arc], SMALL).query([[0, 0, 1], [-1, 0, 0]])
        self.assertTrue(result["tie"][0])
        self.assertAlmostEqual(result["distance"][1], np.sqrt(2), places=6)

    def test_cavity_sdf_sign(self):
        solid = BRepAlgoAPI_Cut(
            BRepPrimAPI_MakeBox(gp_Pnt(-1.5, -1.5, -1.5), 3, 3, 3).Shape(),
            BRepPrimAPI_MakeBox(gp_Pnt(-0.5, -0.5, -0.5), 1, 1, 1).Shape(),
        ).Shape()
        targets = collect_targets(solid, SMALL)
        mesh = triangulate(targets, SMALL)
        values = CADSDFOracle(solid, targets.faces, SMALL).query(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        )
        np.testing.assert_allclose(values, [0.5, -0.5, 0.5], atol=1e-6)
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
            allocate(cfg.pool_count, cfg.pool_weights), [262144, 393216, 327680, 196608]
        )
        self.assertEqual(cfg.geometry_surface_pool_count, 262144)
        self.assertEqual(cfg.band_max, 0.04)
        self.assertEqual(cfg.udf_truncation, 0.05)
        self.assertEqual(cfg.feature_radii, (0.002, 0.005, 0.01, 0.02, 0.04))
        self.assertEqual(cfg.feature_weights, (0.2, 0.28, 0.27, 0.17, 0.08))
        self.assertEqual(cfg.feature_surface_ratios, (0.0, 0.0))
        self.assertTrue(cfg.unify_same_domain)
        with self.assertRaises(ValueError):
            load_config(overrides={"query_count": 100})
        with self.assertRaises(ValueError):
            load_config(overrides={"feature_surface_ratios": [0.5, 0.5]})

    def test_default_edge_zero_budget_and_all_vertices(self):
        targets = collect_targets(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), SMALL)
        banks = zero_pools(targets, Config())
        edges, vertices = banks["edge_zero"], banks["vertex_zero"]
        self.assertEqual(len(edges["points"]), 65536)
        self.assertEqual(len(np.unique(edges["points"], axis=0)), 65536)
        np.testing.assert_array_equal(
            vertices["points"], targets.vertices.astype(np.float32)
        )
        self.assertTrue(
            np.all(np.bincount(edges["source_instance"], minlength=12) >= 8)
        )
        with self.assertRaises(ValueError):
            load_config(overrides={"zero_checks_per_edge": 3})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_cube_pools_labels_and_independent_evaluation(self):
        p = write_step(self.root, "box", BRepPrimAPI_MakeBox(1, 1, 1).Shape())
        out = self.root / "cube"
        cfg = replace(SMALL, volume_pool_count=128, evaluation_volume_count=64)
        result = build_asset(record(p, "cube"), out, cfg)
        verify_asset(out, result["hashes"])
        oracle, _, _, _ = read_geometry(out / "condition")
        for name in ALL_POOLS:
            bank = out / "pools" / name
            x = np.load(bank / "points.npy")
            f = np.load(bank / "fields.npy")
            np.testing.assert_allclose(f[:, 0], oracle.query(x), atol=1e-7)
            solid, _ = read_normalized(p, SMALL)
            targets = collect_targets(solid, SMALL)
            np.testing.assert_allclose(
                f[:, 1],
                np.minimum(
                    ReferenceEdgeOracle(targets.edges, SMALL).query(x)["distance"],
                    SMALL.udf_truncation,
                ),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                f[:, 2],
                np.minimum(
                    ReferenceVertexOracle(targets.vertices, SMALL).query(x)["distance"],
                    SMALL.udf_truncation,
                ),
                atol=1e-6,
            )
            if name in ("near_surface", "edge_neighborhood", "vertex_neighborhood"):
                self.assertLessEqual(np.abs(f[:, 0]).max(), SMALL.band_max + 1e-7)
        train = np.load(out / "pools/surface/points.npy")
        evaluation = np.load(out / "evaluation/surface/points.npy")
        seen = {tuple(x) for x in train}
        self.assertFalse(any(tuple(x) in seen for x in evaluation))
        meta = json.loads((out / "meta.json").read_text())
        self.assertEqual(
            sum(v["count"] for k, v in meta["statistics"].items() if k in POOLS),
            SMALL.pool_count,
        )
        self.assertEqual(meta["channel_names"], ["s_mesh", "dE", "dV"])
        counts = allocate(SMALL.pool_count, SMALL.pool_weights)
        for name, total in zip(POOLS[1:], counts[1:]):
            bank = out / "pools" / name
            types = np.load(bank / "sample_type.npy")
            self.assertEqual(int((types == SPATIAL).sum()), int(total))
            mode = np.load(bank / "perturbation_mode.npy")
            layer = np.load(bank / "perturbation_layer.npy")
            if name == "near_surface":
                nc, ic = allocate(int(total), [0.8, 0.2])
                for m, quotas in enumerate(
                    (
                        allocate(int(nc), [0.25, 0.35, 0.25, 0.15]),
                        allocate(int(ic), [1, 1, 1, 1]),
                    )
                ):
                    for k, need in enumerate(quotas):
                        self.assertEqual(
                            int(((mode == m) & (layer == k)).sum()), int(need)
                        )
            else:
                self.assertTrue(np.all(mode == FEATURE_3D_PERTURBATION_MODE))
                self.assertTrue(
                    np.all((layer >= 0) & (layer < len(SMALL.feature_radii)))
                )
            owners = np.load(bank / "source_instance.npy")
            self.assertTrue(
                np.all(owners == -1) if name == "near_surface" else np.all(owners >= 0)
            )
        self.assertEqual(
            len(np.load(out / "pools/edge_zero/points.npy")), SMALL.zero_edge_count
        )
        self.assertEqual(len(np.load(out / "pools/vertex_zero/points.npy")), 8)
        self.assertEqual(
            len(np.load(out / "evaluation/edge_zero/points.npy")),
            SMALL.evaluation_zero_edge_count,
        )
        for parent, count in [("pools", 128), ("evaluation", 64)]:
            x = np.load(out / parent / "volume/points.npy")
            sdf = np.load(out / parent / "volume/sdf.npy")
            self.assertEqual(len(x), count)
            np.testing.assert_allclose(sdf[:, 0], oracle.query(x), atol=1e-7)
        from veudf.geometry.sampling import face_normals

        fn = face_normals(oracle.vertices, oracle.faces)
        for name in ("surface", "edge_neighborhood", "vertex_neighborhood"):
            bank = out / "pools" / name
            ids = np.load(bank / "triangle_ids.npy")
            types = np.load(bank / "sample_type.npy")
            normals = np.load(bank / "normals.npy")
            np.testing.assert_array_equal(
                normals[types == SURFACE], fn[ids[types == SURFACE]]
            )
            self.assertTrue(np.all(normals[types == SPATIAL] == 0))

    def test_cylinder_empty_vertices_and_geometry_stability(self):
        p = write_step(self.root, "cyl", BRepPrimAPI_MakeCylinder(1, 2).Shape())
        build_asset(record(p, "base"), self.root / "base", SMALL)
        cfg = replace(SMALL, pool_count=8192)
        build_asset(record(p, "grown"), self.root / "grown", cfg)
        for name in ["base", "grown"]:
            bank = self.root / name / "pools/vertex_neighborhood/fields.npy"
            self.assertEqual(len(np.load(bank)), 0)
            f = np.load(self.root / name / "pools/surface/fields.npy")
            self.assertTrue(np.all(f[:, 2] == np.float32(cfg.udf_truncation)))
            ids = np.load(self.root / name / "pools/surface/nearest_ids.npy")
            sat = np.load(self.root / name / "pools/surface/saturated.npy")
            self.assertTrue(np.all(ids[:, 1] == -1))
            self.assertFalse(sat[:, 1].any())
            evaluation = np.load(self.root / name / "evaluation/surface/fields.npy")
            self.assertTrue(np.all(evaluation[:, 2] == np.float32(EMPTY_DISTANCE)))
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

    def test_full_cone_degenerate_apex_excluded(self):
        step = write_step(self.root, "full_cone", BRepPrimAPI_MakeCone(1.0, 0.0, 2.0).Shape())
        out = self.root / "full_cone_asset"
        result = build_asset(record(step, "full_cone"), out, SMALL)
        verify_asset(out, result["hashes"])
        meta = json.loads((out / "meta.json").read_text())
        topo = json.loads((out / "topology.json").read_text())
        self.assertEqual(meta["target_counts"], {"faces": 2, "edges": 1, "vertices": 0})
        self.assertTrue(
            any(e["reason"] == "degenerate_excluded" for e in topo["excluded_edges"])
        )
        self.assertTrue(
            any(v["reason"] == "no_retained_edge" for v in topo["excluded_vertices"])
        )
        self.assertEqual(len(np.load(out / "pools/vertex_neighborhood/fields.npy")), 0)
        self.assertEqual(len(np.load(out / "pools/vertex_zero/points.npy")), 0)
        self.assertEqual(
            sum(v["count"] for k, v in meta["statistics"].items() if k in POOLS),
            SMALL.pool_count,
        )

    def test_tilted_half_cone_retains_apex_vertex(self):
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 2), gp_Dir(1, -1, 0))
        half_cone = BRepPrimAPI_MakeCone(axis, 1.0, 0.0, 2.0, np.pi).Shape()
        step = write_step(self.root, "tilted_half_cone", half_cone)
        out = self.root / "tilted_half_cone_asset"
        result = build_asset(record(step, "tilted_half_cone"), out, SMALL)
        verify_asset(out, result["hashes"])
        meta = json.loads((out / "meta.json").read_text())
        topo = json.loads((out / "topology.json").read_text())
        self.assertEqual(meta["target_counts"], {"faces": 3, "edges": 4, "vertices": 3})
        self.assertTrue(
            any(e["reason"] == "degenerate_excluded" for e in topo["excluded_edges"])
        )
        self.assertEqual(len(np.load(out / "pools/vertex_zero/points.npy")), 3)
        self.assertGreater(len(np.load(out / "pools/vertex_neighborhood/fields.npy")), 0)

    def test_unify_same_domain_merges_two_half_cones(self):
        # Two fused half-frustums (truncated cones): UnifySameDomain merges 6 -> 3 faces.
        f1 = BRepPrimAPI_MakeCone(
            gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)),
            1.0,
            0.25,
            2.0,
            np.pi,
        ).Shape()
        f2 = BRepPrimAPI_MakeCone(
            gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(-1, 0, 0)),
            1.0,
            0.25,
            2.0,
            np.pi,
        ).Shape()
        step = write_step(self.root, "two_half_frustums", BRepAlgoAPI_Fuse(f1, f2).Shape())
        out_unified = self.root / "unified_frustum"
        result = build_asset(record(step, "unified_frustum"), out_unified, SMALL)
        verify_asset(out_unified, result["hashes"])
        meta = json.loads((out_unified / "meta.json").read_text())
        unify = meta["normalization"]["unify_same_domain"]
        self.assertTrue(unify["applied"])
        self.assertEqual(unify["reason"], "unified")
        self.assertEqual(unify["faces_before"], 6)
        self.assertEqual(unify["faces_after"], 3)
        self.assertEqual(meta["target_counts"], {"faces": 3, "edges": 2, "vertices": 0})
        # Disabled flag preserves the unmerged faces.
        solid_raw, norm_raw = read_normalized(step, replace(SMALL, unify_same_domain=False))
        self.assertFalse(norm_raw["unify_same_domain"]["applied"])
        self.assertEqual(len(collect_targets(solid_raw, SMALL).faces), 6)
        # Two fused sharp half-cones (with degenerate apex edges): when OCC's
        # UnifySameDomain fails on split coaxial cones, reject strictly rather
        # than leaking artificial meridian seams into E/V supervision.
        c1 = BRepPrimAPI_MakeCone(
            gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)),
            1.0,
            0.0,
            2.0,
            np.pi,
        ).Shape()
        c2 = BRepPrimAPI_MakeCone(
            gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(-1, 0, 0)),
            1.0,
            0.0,
            2.0,
            np.pi,
        ).Shape()
        sharp_step = write_step(
            self.root, "two_sharp_half_cones", BRepAlgoAPI_Fuse(c1, c2).Shape()
        )
        with self.assertRaises(PreprocessError) as caught:
            build_asset(
                record(sharp_step, "sharp_half_cones"),
                self.root / "sharp_half_cones_asset",
                SMALL,
            )
        self.assertEqual(caught.exception.code, "unmerged_same_domain_seam")
        # Even if OCC's UnifySameDomain returns the split shape unchanged without
        # throwing, the post-check must still reject residual coaxial cone seams.
        from unittest.mock import patch
        import veudf.preprocess.cad as cad_mod

        class _NoOpUnifier:
            def __init__(self, shape, *args, **kwargs):
                self._shape = shape

            def SetLinearTolerance(self, tol):
                pass

            def SetAngularTolerance(self, tol):
                pass

            def Build(self):
                pass

            def Shape(self):
                return self._shape

        with patch.object(cad_mod, "ShapeUpgrade_UnifySameDomain", _NoOpUnifier):
            with self.assertRaises(PreprocessError) as caught_noop:
                read_normalized(sharp_step, SMALL)
            self.assertEqual(caught_noop.exception.code, "unmerged_same_domain_seam")

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
