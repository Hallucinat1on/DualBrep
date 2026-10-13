import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback

from .config import Config, load_config
from .manifest import load_manifest, scan_directory


def worker(record, output, config, resume, overwrite, results):
    try:
        from .pipeline import build_sample
        outcome = build_sample(record["step_path"], output, Config(**config),
                               resume=resume, overwrite=overwrite, source_record=record)
    except Exception as exc:
        outcome = dict(status="failed", code=getattr(exc, "code", "missing_dependency" if isinstance(exc, ImportError) else "unhandled_error"),
                       message=str(exc), traceback=traceback.format_exc())
    outcome["sample_id"] = record["sample_id"]
    results.put(outcome)


def run_records(records, output_root, cfg, workers, timeout, resume, overwrite, log_path):
    context = mp.get_context("spawn")
    pending = iter(records)
    active, outcomes = [], []
    exhausted = False
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        try:
            while active or not exhausted:
                while len(active) < workers and not exhausted:
                    try:
                        record = next(pending)
                    except StopIteration:
                        exhausted = True
                        break
                    destination = output_root / record["sample_id"] if output_root is not None else Path(record["output"])
                    result_queue = context.Queue(maxsize=1)
                    process = context.Process(target=worker,
                        args=(record, str(destination), cfg.to_dict(), resume, overwrite, result_queue))
                    process.start()
                    active.append((process, result_queue, record, time.monotonic()))
                for item in active[:]:
                    process, result_queue, record, started = item
                    result = None
                    try:
                        result = result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    if result is None and time.monotonic() - started > timeout:
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join()
                        result = dict(status="failed", code="timeout", sample_id=record["sample_id"],
                                      message=f"Sample exceeded {timeout:g} seconds")
                    elif result is None and not process.is_alive():
                        # Allow the queue feeder to flush after worker exit.
                        try:
                            result = result_queue.get(timeout=0.2)
                        except queue.Empty:
                            result = dict(status="failed", code="worker_crash", sample_id=record["sample_id"],
                                          message=f"Worker exited with code {process.exitcode}")
                    if result is not None:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.terminate()
                            process.join()
                        active.remove(item)
                        result_queue.close()
                        outcomes.append(result)
                        log.write(json.dumps(result, ensure_ascii=False) + "\n")
                        log.flush()
                        print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False), flush=True)
                if active:
                    time.sleep(0.05)
        finally:
            for process, result_queue, _, _ in active:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                result_queue.close()
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portable STEP -> aligned SDF, E-UDF, V-UDF labels")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--step", help="Single .step/.stp (also .gz)")
    inputs.add_argument("--step-dir", help="Recursively scan a directory of extracted STEP files")
    inputs.add_argument("--manifest", help="JSONL: sample_id and step_path, optional SHA256/group/split")
    parser.add_argument("--out", help="Single sample output directory")
    parser.add_argument("--out-dir", help="Batch output root")
    parser.add_argument("--config", help="JSON configuration; unspecified keys use defaults")
    parser.add_argument("--edge-backend", choices=("cad_reference", "cad_bvh"))
    parser.add_argument("--sdf-backend", choices=("cad", "mesh", "mesh_fast"))
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--resume", action="store_true", help="Reuse only hash-verified matching results")
    reuse.add_argument("--overwrite", action="store_true", help="Replace a previous evudf sample after new checks pass")
    parser.add_argument("--split", choices=("train", "val", "test"), help="Restrict batch manifest split")
    parser.add_argument("--limit", type=int, help="Development sample limit")
    parser.add_argument("--log", help="Results/errors JSONL; default alongside output")
    args = parser.parse_args(argv)
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("workers, timeout and limit must be positive")
    try:
        cfg = load_config(args.config, dict(edge_backend=args.edge_backend, sdf_backend=args.sdf_backend,
                                          query_count=args.query_count, seed=args.seed))
        if args.step:
            if not args.out or args.out_dir or args.split:
                parser.error("--step requires --out; --out-dir/--split apply to batches")
            records = [dict(sample_id=Path(args.out).name, step_path=str(Path(args.step).resolve()),
                            output=str(Path(args.out).resolve()))]
            root = None
            log_path = args.log or str(Path(args.out).resolve().parent / (Path(args.out).name + ".run.jsonl"))
        else:
            if not args.out_dir or args.out:
                parser.error("Batch input requires --out-dir")
            records = load_manifest(args.manifest) if args.manifest else scan_directory(args.step_dir, seed=cfg.seed)
            records = [r for r in records if "duplicate_of" not in r and (not args.split or r.get("split") == args.split)]
            if args.limit:
                records = records[:args.limit]
            root = Path(args.out_dir).resolve()
            log_path = args.log or str(root / "run.jsonl")
        if not records:
            parser.error("No matching STEP records")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    results = run_records(records, root, cfg, args.workers, args.timeout, args.resume, args.overwrite, log_path)
    failed = sum(r["status"] == "failed" for r in results)
    print(json.dumps(dict(total=len(results), failed=failed, log=str(log_path)), ensure_ascii=False))
    return 1 if failed else 0


def manifest_main(argv=None):
    parser = argparse.ArgumentParser(description="Index extracted ABC STEP files with deterministic grouped splits")
    parser.add_argument("--step-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    try:
        rows = scan_directory(args.step_dir, seed=args.seed,
                              train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps(dict(records=len(rows), duplicates=sum("duplicate_of" in r for r in rows), manifest=str(out))))
    return 0
