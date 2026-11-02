"""Remaining-candidate selection must preserve old exclusions and family splits."""

from pathlib import Path
import unittest
from unittest.mock import patch
import contextlib
import io
import json
import sys
import tempfile
import os
import subprocess

from veudf.cli import prepare_remaining_run as prepare
from veudf.cli import run_guarded_build as guard


def row(uid, digest, group, split="train"):
    return dict(
        sample_id=uid,
        step_path=f"/data/{uid}.step",
        source_sha256=digest,
        group_id=group,
        split=split,
    )


class RemainingTests(unittest.TestCase):
    def test_old_content_exclusion_and_family_pin(self):
        old = [row("old", "hash0", "old_family", "val")]
        alias = row("different_name", "hash0", "merged_global", "train")
        sibling = row("sibling", "hash1", "merged_global", "train")
        duplicate = dict(row("copy", "hash1", "merged_global"), duplicate_of="sibling")
        result, summary = prepare.select_remaining(
            [alias, sibling, duplicate], old, 50000, 0
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(
            (result[0]["sample_id"], result[0]["split"], result[0]["group_id"]),
            ("sibling", "val", "old_family"),
        )
        self.assertEqual(sibling["split"], "train")  # Input remains immutable.
        self.assertEqual(summary["excluded"]["previous_path_or_content"], 1)

    def test_conflicting_old_splits_exclude_new_members(self):
        old = [row("a", "ha", "ga", "train"), row("b", "hb", "gb", "test")]
        full = [
            row("a", "ha", "merged"),
            row("b", "hb", "merged"),
            row("new", "hc", "merged"),
        ]
        result, summary = prepare.select_remaining(full, old, 10, 0)
        self.assertEqual(result, [])
        self.assertEqual(summary["conflicting_global_groups"], ["merged"])

    def test_limit_is_exact_deterministic_and_independent_of_directory_order(self):
        full = [row(str(i), f"hash{i}", str(i)) for i in range(100)]
        a, _ = prepare.select_remaining(full, [], 23, 0)
        b, _ = prepare.select_remaining(full[::-1], [], 23, 0)
        c, _ = prepare.select_remaining(full, [], 23, 1)
        self.assertEqual(len(a), 23)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_order_and_historical_source_group_pin(self):
        old = dict(row("old", "old_hash", "old_family", "val"), source_group_id="document")
        new = dict(row("new", "new_hash", "new_family"), source_group_id="document")
        other = row("other", "other_hash", "other_family")
        result, summary = prepare.select_remaining([other, new], [old], 2, 0, preserve_order=True)
        self.assertEqual([r["sample_id"] for r in result], ["other", "new"])
        self.assertEqual((result[1]["group_id"], result[1]["split"]), ("old_family", "val"))
        self.assertEqual(summary["selection_order"], "manifest")

    def test_limited_scan_stops_before_later_directories(self):
        from veudf.preprocess.manifest import scan_directory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("0000", "0001"):
                (root / name).mkdir()
            for name in ("b.step", "a.step"):
                (root / "0000" / name).write_bytes(name.encode())
            (root / "0001" / "later.step").write_bytes(b"later")
            seen = []
            result = scan_directory(root, limit=2, progress=lambda n, p: seen.append(p))
            self.assertEqual(seen, ["0000/a.step", "0000/b.step"])
            self.assertEqual([r["archive_member"] for r in result], seen)


class ResourceTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "run"
        run.mkdir()
        (run / "selected_manifest.jsonl").write_text("\n")
        (run / "config.json").write_text("{}")
        (run / "code_inventory.json").write_text("{}")
        prep = dict(
            candidate_limit=50000,
            source_identity={"code": "test"},
            selected_manifest_sha256=prepare.digest(run / "selected_manifest.jsonl"),
            config_sha256=prepare.digest(run / "config.json"),
        )
        (run / "preparation.json").write_text(json.dumps(prep))
        return run

    def test_insufficient_memory_never_launches_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root)
            with patch.object(
                sys, "argv", ["guard", "--run-dir", str(root)]
            ), patch.object(guard, "cpu_budget", return_value=64), patch.object(
                guard, "memory_budget", return_value=(180 * guard.GIB, 32 * guard.GIB)
            ), patch.object(
                guard.subprocess, "Popen"
            ) as launch, contextlib.redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(guard.main(), 75)
                launch.assert_not_called()
            self.assertEqual(
                json.loads((run / "resource_status.json").read_text())["state"],
                "insufficient_memory_before_start",
            )

    def test_pressure_stops_running_driver_for_resumable_exit(self):
        class Child:
            pid = 123456
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root)
            child = Child()
            with patch.object(
                sys, "argv", ["guard", "--run-dir", str(root)]
            ), patch.object(guard, "cpu_budget", return_value=64), patch.object(
                guard,
                "memory_budget",
                side_effect=[
                    (180 * guard.GIB, 140 * guard.GIB),
                    (180 * guard.GIB, 32 * guard.GIB),
                ],
            ), patch.object(
                guard, "tree_rss", return_value=guard.GIB
            ), patch.object(
                guard.os, "sched_setaffinity"
            ), patch.object(
                guard.os, "nice"
            ), patch.object(
                guard.signal, "signal"
            ), patch.object(
                guard.subprocess, "check_output", return_value='{"code":"test"}'
            ), patch.object(
                guard.subprocess, "Popen", return_value=child
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(guard.main(), 75)
            self.assertEqual(child.returncode, -15)
            self.assertEqual(
                json.loads((run / "resource_status.json").read_text())["state"],
                "stopped_on_memory_pressure",
            )


if __name__ == "__main__":
    unittest.main()


def test_frozen_build_runs_and_resumes_without_repository(tmp_path):
    from dataclasses import replace
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Extend.DataExchange import write_step_file
    from veudf.preprocess.config import Config
    from veudf.preprocess.manifest import scan_directory

    source = tmp_path / "step"
    source.mkdir()
    step = source / "cube.step"
    write_step_file(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), str(step))
    manifest = tmp_path / "manifest.jsonl"
    records = scan_directory(source, train_fraction=1.0, val_fraction=0.0)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    previous = tmp_path / "old"
    previous.mkdir()
    (previous / "manifest.jsonl").write_text("")
    config = tmp_path / "config.json"
    cfg = replace(
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
    config.write_text(json.dumps(cfg.to_dict()))
    job = tmp_path / "job"
    prepare.main(
        [
            "--manifest",
            str(manifest),
            "--exclude-root",
            str(previous),
            "--run-dir",
            str(job),
            "--config",
            str(config),
            "--limit",
            "1",
        ]
    )
    frozen = job / "run/code"
    assert {p.name for p in frozen.iterdir()} == {"veudf"}
    env = dict(
        os.environ,
        PYTHONPATH=str(frozen),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    location = subprocess.check_output(
        [sys.executable, "-c", "import veudf; print(veudf.__file__)"],
        cwd=tmp_path,
        env=env,
        text=True,
    )
    assert Path(location.strip()).is_relative_to(frozen)
    command = [
        sys.executable,
        "-m",
        "veudf.cli.run_guarded_build",
        "--run-dir",
        str(job),
        "--max-workers",
        "1",
        "--reserve-cpus",
        "0",
        "--reserve-memory-gib",
        "0",
        "--max-rss-gib",
        "4",
        "--timeout",
        "120",
    ]
    first = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
    )
    assert first.returncode == 0, (
        first.stdout + first.stderr + (job / "run/build.log").read_text()
    )
    report = json.loads((job / "data/report.json").read_text())
    assert report["published"] == 1
    # Exercise the adaptive scheduler with a real frozen worker and a fresh job.
    adaptive_job = tmp_path / "adaptive_job"
    prepare.main(["--manifest", str(manifest), "--exclude-root", str(previous),
                  "--run-dir", str(adaptive_job), "--config", str(config), "--limit", "1"])
    adaptive_path = Path(__file__).parents[1] / "adaptive_build.py"
    adaptive_env = dict(env, PYTHONPATH=str(adaptive_job / "run/code"))
    script = ("import importlib.util,sys; from pathlib import Path; "
              "import veudf.cli.run_guarded_build as g; "
              "g.memory_budget=lambda:(180*2**30,100*2**30); "
              "s=importlib.util.spec_from_file_location('adaptive',sys.argv[1]); "
              "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
              "sys.exit(m.worker(Path(sys.argv[2]),'pilot',1,32))")
    adaptive = subprocess.run([sys.executable, "-c", script, str(adaptive_path), str(adaptive_job)],
                              cwd=tmp_path, env=adaptive_env, capture_output=True, text=True, timeout=180)
    assert adaptive.returncode == 0, adaptive.stdout + adaptive.stderr
    assert json.loads((adaptive_job / "data/report.json").read_text())["published"] == 1
    points = next((job / "data/assets").glob("*/pools/surface/points.npy"))
    before = points.stat().st_mtime_ns
    step.unlink()  # A successful resume must not reopen/rebuild the original STEP.
    second = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert points.stat().st_mtime_ns == before
    assert json.loads((job / "data/status.json").read_text())["state"] == "finished"
