"""Dataset driver: manifest -> per-asset subprocess builds -> dedup -> published index.

Layout of ``--root``::

    config.json  manifest.jsonl  identity.json
    assets/<sample_id>/{condition,pools,evaluation,audit,topology.json,meta.json}
    results/<sample_id>.json  logs/<sample_id>.log
    index.json  report.json  status.json
"""

import argparse
from collections import Counter
import concurrent.futures as futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

# Disable synchronous transparent-hugepage compaction before importing NumPy.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import numpy as np

from veudf.schema import SCHEMA as SCHEMA_VERSION
from veudf.preprocess.config import load_config
from veudf.preprocess.manifest import load_manifest, scan_directory
from veudf.preprocess.pipeline import code_fingerprint, sha256_file, verify_asset

SPLIT_ORDER = {"test": 0, "val": 1, "train": 2}
RETRYABLE = {"timeout", "worker_crash"}
CHILDREN, LOCK, STOP = set(), threading.Lock(), threading.Event()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def worker(root, sample_id, record=None):
    """Child process: build one asset and record its outcome."""
    from veudf.preprocess.pipeline import build_asset

    root = Path(root)
    if record is None:
        record = next(
            r for r in load_manifest(root / "manifest.jsonl") if r["sample_id"] == sample_id
        )
    if record["sample_id"] != sample_id:
        raise ValueError("Worker record does not match sample ID")
    started = time.monotonic()
    try:
        result = dict(
            status="passed",
            **build_asset(
                record, root / "assets" / sample_id, load_config(root / "config.json")
            ),
        )
    except Exception as exc:
        result = dict(
            status="failed",
            code=getattr(exc, "code", type(exc).__name__),
            message=str(exc),
            traceback=traceback.format_exc(),
        )
    result.update(
        sample_id=sample_id, split=record["split"], seconds=time.monotonic() - started
    )
    atomic_json(root / "results" / f"{sample_id}.json", result)
    return 0 if result["status"] == "passed" else 1


def invoke(root, record, args):
    uid = record["sample_id"]
    result_path = root / "results" / f"{uid}.json"
    result_path.unlink(missing_ok=True)
    shutil.rmtree(root / "assets" / uid, ignore_errors=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    with (root / "logs" / f"{uid}.log").open("a") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "veudf.cli.build_dataset",
                "_worker",
                str(root),
                uid,
                "--record-stdin",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            stdin=subprocess.PIPE,
        )
        with LOCK:
            CHILDREN.add(child)
        try:
            try:
                child.stdin.write(json.dumps(record).encode())
                child.stdin.close()
            except BrokenPipeError:
                pass
            if STOP.is_set():
                os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            result = dict(status="failed", code="timeout")
        else:
            result = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else dict(
                    status="failed", code="worker_crash", returncode=child.returncode
                )
            )
        finally:
            with LOCK:
                CHILDREN.discard(child)
    result.update(sample_id=uid, split=record["split"])
    if result["status"] != "passed":
        shutil.rmtree(root / "assets" / uid, ignore_errors=True)
    atomic_json(result_path, result)
    return result


def publish(root, records, results, cfg):
    """Dedup in priority order (test, val, train, id) and write the training index."""
    from veudf.preprocess.dedup import Deduplicator

    dedup = Deduplicator(cfg)
    assets, rejected, families, sources = [], {}, {}, {}
    for r in sorted(records, key=lambda r: (SPLIT_ORDER[r["split"]], r["sample_id"])):
        result = results.get(r["sample_id"])
        if not result or result["status"] != "passed":
            continue
        directory = root / "assets" / r["sample_id"]
        verify_asset(directory, result["hashes"])
        feats = {
            k: np.load(directory / "audit" / f"{k}.npy", allow_pickle=False)
            for k in ("dedup_surface", "dedup_edge", "dedup_vertex")
        }
        outcome = dedup.consider(
            r["sample_id"],
            r["split"],
            json.loads((directory / "topology.json").read_text()),
            feats,
        )
        if outcome["status"] != "passed":
            rejected[r["sample_id"]] = outcome
            continue
        for key, seen in (
            (r["group_id"], families),
            (result["source_sha256"], sources),
        ):
            if seen.setdefault(key, r["split"]) != r["split"]:
                raise ValueError(f"Family/source split leakage at {r['sample_id']}")
        assets.append(
            {
                k: result[k]
                for k in ("asset_id", "family_id", "split", "source_sha256", "hashes")
            }
        )
    atomic_json(root / "index.json", dict(schema=SCHEMA_VERSION, assets=assets))
    failures = Counter(
        v.get("code", "unknown") for v in results.values() if v["status"] != "passed"
    )
    faces = Counter()
    for a in assets:
        n = json.loads((root / "assets" / a["asset_id"] / "meta.json").read_text())[
            "target_counts"
        ]["faces"]
        faces[
            "2-9" if n < 10 else "10-19" if n < 20 else "20-39" if n < 40 else "40-100"
        ] += 1
    report = dict(
        candidates=len(records),
        attempted=len(results),
        built=sum(v["status"] == "passed" for v in results.values()),
        published=len(assets),
        splits=dict(Counter(a["split"] for a in assets)),
        face_bins=dict(faces),
        failures=dict(failures.most_common()),
        dedup=dict(Counter(v["code"] for v in rejected.values())),
        dedup_rejected=rejected,
    )
    atomic_json(root / "report.json", report)
    return report


def build_main(args):
    root = Path(args.root).resolve()
    for name in ("assets", "results", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    with (root / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / "config.json").exists():
            cfg = load_config(args.config)
            atomic_json(root / "config.json", cfg.to_dict())
            shutil.copyfile(args.manifest, root / "manifest.jsonl")
        cfg = load_config(root / "config.json")
        identity = dict(
            schema=SCHEMA_VERSION,
            config_sha256=sha256_file(root / "config.json"),
            manifest_sha256=sha256_file(root / "manifest.jsonl"),
            code_sha256=code_fingerprint(),
        )
        if (root / "identity.json").exists() and json.loads(
            (root / "identity.json").read_text()
        ) != identity:
            raise SystemExit(
                "Config/manifest/code changed since this root was created; use a new --root"
            )
        atomic_json(root / "identity.json", identity)
        records = [
            r for r in load_manifest(root / "manifest.jsonl") if "duplicate_of" not in r
        ]
        # Schedule in id order so a small run still mixes splits (splits follow group hashes).
        records.sort(key=lambda r: r["sample_id"])
        if args.limit:
            records = records[: args.limit]
        results = {}
        for r in records:
            p = root / "results" / f"{r['sample_id']}.json"
            if p.exists():
                result = json.loads(p.read_text())
                if result["status"] == "passed" or not (
                    args.retry_failed or result.get("code") in RETRYABLE
                ):
                    results[r["sample_id"]] = result
        queue = [r for r in records if r["sample_id"] not in results]
        started = time.time()

        def passed():
            return sum(v["status"] == "passed" for v in results.values())

        def status(state, report=None):
            info = dict(
                state=state,
                pid=os.getpid(),
                updated_unix=time.time(),
                elapsed_seconds=time.time() - started,
                candidates=len(records),
                done=len(results),
                passed=passed(),
                remaining=len(queue),
                failures=dict(
                    Counter(
                        v.get("code")
                        for v in results.values()
                        if v["status"] != "passed"
                    )
                ),
            )
            if args.target:
                info.update(
                    target=args.target,
                    published=None if report is None else report["published"],
                )
            atomic_json(root / "status.json", info)

        status("building")
        report = None
        with (root / "outcomes.jsonl").open("a") as log:
            while True:
                inflight = {}
                with futures.ThreadPoolExecutor(args.workers) as pool:

                    def fill():
                        # Never schedule after a stop request: children would be killed on
                        # arrival and the queue would be consumed as worker_crash.
                        while (
                            queue
                            and len(inflight) < args.workers
                            and not STOP.is_set()
                            and (not args.target or passed() < args.target)
                        ):
                            r = queue.pop(0)
                            inflight[pool.submit(invoke, root, r, args)] = r

                    fill()
                    while inflight and not STOP.is_set():
                        done, _ = futures.wait(
                            inflight, return_when=futures.FIRST_COMPLETED
                        )
                        for job in done:
                            inflight.pop(job)
                            result = job.result()
                            results[result["sample_id"]] = result
                            brief = {
                                k: v
                                for k, v in result.items()
                                if k not in ("traceback", "hashes")
                            }
                            log.write(json.dumps(brief) + "\n")
                            log.flush()
                            print(json.dumps(brief), flush=True)
                            status("cancelled" if STOP.is_set() else "building")
                        fill()
                    for job in futures.as_completed(list(inflight)):
                        job.result()
                if STOP.is_set():
                    return 130
                status("publishing")
                report = publish(root, records, results, cfg)
                # Dedup rejections can leave the target short: keep building more candidates.
                if not args.target or report["published"] >= args.target or not queue:
                    break
                status("building", report)
        status("finished", report)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "dedup_rejected"}, indent=2
            )
        )
        return 0


def manifest_main(args):
    def progress(count, path):
        if count == 1 or count % 1000 == 0 or count == args.limit:
            print(json.dumps(dict(scanned=count, limit=args.limit, current_path=path)), flush=True)

    rows = scan_directory(
        args.step_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        limit=args.limit,
        progress=progress,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(
        json.dumps(
            dict(
                records=len(rows),
                duplicates=sum("duplicate_of" in r for r in rows),
                manifest=str(out),
            )
        )
    )
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_worker"]:
        record = json.load(sys.stdin) if argv[3:] == ["--record-stdin"] else None
        return worker(*argv[1:3], record=record)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser(
        "manifest", help="Index extracted ABC STEP files with grouped splits"
    )
    m.add_argument("--step-dir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--train-fraction", type=float, default=0.84)
    m.add_argument("--val-fraction", type=float, default=0.08)
    m.add_argument("--limit", type=int, default=0, help="Scan at most the first N STEP paths")
    b = sub.add_parser("build", help="Build, deduplicate and publish (resumable)")
    b.add_argument("--root", required=True)
    b.add_argument("--manifest", help="Required when creating a root")
    b.add_argument(
        "--config", help="Config JSON when creating a root (default: built-in)"
    )
    b.add_argument("--workers", type=int, default=8)
    b.add_argument(
        "--threads",
        type=int,
        default=4,
        help="OpenMP threads per worker (mesh distance queries)",
    )
    b.add_argument("--timeout", type=float, default=3600)
    b.add_argument(
        "--limit", type=int, default=0, help="Only the first N candidates (pilot)"
    )
    b.add_argument(
        "--target",
        type=int,
        default=0,
        help="Stop once N assets are published; keeps building until reached",
    )
    b.add_argument(
        "--retry-failed", action="store_true", help="Rebuild every recorded failure"
    )
    args = parser.parse_args(argv)
    if args.command == "manifest":
        return manifest_main(args)
    if (
        min(args.workers, args.threads) < 1
        or args.timeout <= 0
        or args.limit < 0
        or args.target < 0
    ):
        parser.error("workers/threads/timeout must be positive")
    if not (Path(args.root) / "config.json").exists() and not args.manifest:
        parser.error("--manifest is required for a new root")

    def stop(signum, frame):
        STOP.set()
        with LOCK:
            for child in list(CHILDREN):
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return build_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
