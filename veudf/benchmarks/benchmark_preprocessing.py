"""Random real-STEP full-budget preprocessing run with a frozen code snapshot."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from veudf.preprocess.config import load_config
from veudf.utils import allocate, file_hash


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / "run"
    run.mkdir()
    # The source dataset only selects immutable STEP records, not old labels.
    index = json.loads((source / "index.json").read_text())
    records = {
        r["sample_id"]: r
        for r in map(json.loads, (source / "manifest.jsonl").read_text().splitlines())
    }
    selected = np.random.default_rng(args.seed).choice(
        len(index["assets"]), args.count, replace=False
    )
    chosen = [records[index["assets"][int(i)]["asset_id"]] for i in selected]
    manifest = run / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in chosen))
    config = load_config(args.config)
    config_path = run / "config.json"
    write_json(config_path, config.to_dict())
    package = Path(__file__).resolve().parents[1] / "src" / "veudf"
    shutil.copytree(
        package,
        run / "code" / "veudf",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(__file__, run / Path(__file__).name)
    write_json(
        run / "selection.json",
        dict(
            population=len(index["assets"]),
            seed=args.seed,
            indices=selected.tolist(),
            source_index=str(source / "index.json"),
            source_index_sha256=file_hash(source / "index.json"),
            assets=[r["sample_id"] for r in chosen],
        ),
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "veudf.cli.build_dataset",
        "build",
        "--root",
        str(output / "data"),
        "--manifest",
        str(manifest),
        "--config",
        str(config_path),
        "--workers",
        str(args.workers),
        "--threads",
        "1",
        "--timeout",
        "7200",
    ]
    env = dict(
        os.environ,
        PYTHONPATH=str(run / "code"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    write_json(
        run / "launch.json", dict(command=command, workers=args.workers, threads=1)
    )
    print(f"BUILD {args.count} random assets -> {output / 'data'}", flush=True)
    started = time.monotonic()
    with (output / "build.log").open("w") as log:
        result = subprocess.run(
            command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    batch_seconds = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(
            f"Dataset build failed ({result.returncode}); inspect {output / 'build.log'}"
        )
    data = output / "data"
    published = json.loads((data / "index.json").read_text())["assets"]
    if len(published) != args.count:
        raise RuntimeError(
            "Not all selected assets were published; inspect report.json"
        )
    rows = []
    for record in chosen:
        uid = record["sample_id"]
        asset = data / "assets" / uid
        meta = json.loads((asset / "meta.json").read_text())
        outcome = json.loads((data / "results" / f"{uid}.json").read_text())
        rows.append(
            dict(
                asset_id=uid,
                counts=meta["target_counts"],
                seconds=outcome["seconds"],
                cpu_seconds=outcome["cpu_seconds"],
                timings_seconds=meta["timings_seconds"],
                bytes=sum(p.stat().st_size for p in asset.rglob("*") if p.is_file()),
                pool_counts={
                    name: v["count"] for name, v in meta["statistics"].items()
                },
                quality=meta["quality"],
            )
        )
    seconds = np.array([r["seconds"] for r in rows])
    summary = dict(
        state="complete",
        assets=args.count,
        workers=args.workers,
        threads=1,
        udf_truncation=config.udf_truncation,
        ordinary_points=config.pool_count,
        pool_counts=allocate(config.pool_count, config.pool_weights).tolist(),
        batch_wall_seconds=batch_seconds,
        mean_asset_wall_seconds=float(seconds.mean()),
        median_asset_wall_seconds=float(np.median(seconds)),
        min_asset_wall_seconds=float(seconds.min()),
        max_asset_wall_seconds=float(seconds.max()),
        mean_asset_cpu_seconds=float(np.mean([r["cpu_seconds"] for r in rows])),
        throughput_assets_per_hour=args.count / batch_seconds * 3600,
        effective_seconds_per_asset=batch_seconds / args.count,
        mean_asset_bytes=float(np.mean([r["bytes"] for r in rows])),
        mean_stages={
            k: float(np.mean([r["timings_seconds"][k] for r in rows]))
            for k in rows[0]["timings_seconds"]
        },
        rows=rows,
    )
    write_json(output / "summary.json", summary)
    lines = [
        f"# Feature-first preprocessing: {args.count} random real STEP assets",
        "",
        f"Radius {config.udf_truncation}; {config.pool_count:,} ordinary points per asset; {args.workers} workers x 1 thread.",
        f"Mean asset wall time: {seconds.mean():.3f} seconds; batch wall: {batch_seconds:.3f} seconds.",
        "The source index selects STEP records only; new neighborhood pools are rebuilt and audited.",
        "",
        "| Asset | E/V | Wall seconds | CPU seconds | MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['asset_id'].split('_')[0]} | {r['counts']['edges']}/{r['counts']['vertices']} | {r['seconds']:.3f} | {r['cpu_seconds']:.3f} | {r['bytes']/2**20:.2f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
