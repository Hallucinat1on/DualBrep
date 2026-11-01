"""Run a prepared dataset build with CPU affinity and a soft memory guard.

The guard stops the driver rather than claiming an unavailable cgroup hard cap.
Completed assets can be resumed with exactly the same prepared run directory.
"""

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

GIB = 1 << 30


def cpu_budget():
    budget = float(len(os.sched_getaffinity(0)))
    v2 = Path("/sys/fs/cgroup/cpu.max")
    if v2.is_file():
        quota, period = v2.read_text().split()
        if quota != "max":
            budget = min(budget, int(quota) / int(period))
    else:
        root = Path("/sys/fs/cgroup/cpu")
        quota = int((root / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu.cfs_period_us").read_text())
        if quota > 0:
            budget = min(budget, quota / period)
    return budget


def memory_budget():
    host = psutil.virtual_memory().available
    v2 = Path("/sys/fs/cgroup/memory.max")
    if v2.is_file():
        raw = v2.read_text().strip()
        limit = int(raw) if raw != "max" else psutil.virtual_memory().total
        used = int(v2.with_name("memory.current").read_text())
        stats = dict(
            s.split() for s in v2.with_name("memory.stat").read_text().splitlines()
        )
        reclaimable = int(stats.get("inactive_file", 0))
    else:
        root = Path("/sys/fs/cgroup/memory")
        limit = int((root / "memory.limit_in_bytes").read_text())
        used = int((root / "memory.usage_in_bytes").read_text())
        stats = dict(s.split() for s in (root / "memory.stat").read_text().splitlines())
        reclaimable = int(
            stats.get("total_inactive_file", stats.get("inactive_file", 0))
        )
    return min(limit, psutil.virtual_memory().total), max(
        0, min(host, limit - used + reclaimable)
    )


def tree_rss(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def write_status(run, data):
    data["updated_at"] = datetime.datetime.now().astimezone().isoformat()
    p = run / "resource_status.json"
    temp = p.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--max-workers", type=int, default=48)
    p.add_argument("--reserve-cpus", type=float, default=16)
    p.add_argument("--reserve-memory-gib", type=float, default=64)
    p.add_argument("--max-rss-gib", type=float, default=96)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Optional pilot subset; omit when resuming the full prepared manifest",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if (
        a.max_workers < 1
        or a.candidate_limit < 0
        or min(a.reserve_cpus, a.reserve_memory_gib) < 0
        or min(a.max_rss_gib, a.timeout) <= 0
    ):
        p.error("Invalid resource budget")
    base = Path(a.run_dir).resolve()
    run = base / "run"
    preparation = json.loads((run / "preparation.json").read_text())
    candidate_limit = min(
        a.candidate_limit or preparation["candidate_limit"],
        preparation["candidate_limit"],
    )
    budget = cpu_budget()
    workers = min(a.max_workers, math.floor(budget - a.reserve_cpus))
    if workers < 1:
        p.error("CPU quota is too small to preserve the requested reserve")
    cpus = sorted(os.sched_getaffinity(0))[:workers]
    limit, available = memory_budget()
    code = run / "code"
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMPY_MADVISE_HUGEPAGE="0",
        PYTHONPATH=str(code),
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "veudf.cli.build_dataset",
        "build",
        "--root",
        str(base / "data"),
        "--manifest",
        str(run / "selected_manifest.jsonl"),
        "--config",
        str(run / "config.json"),
        "--limit",
        str(candidate_limit),
        "--workers",
        str(workers),
        "--threads",
        "1",
        "--timeout",
        str(a.timeout),
    ]
    info = dict(
        state="prepared",
        effective_cpus=budget,
        workers=workers,
        affinity=cpus,
        memory_limit_gib=limit / GIB,
        available_gib=available / GIB,
        reserve_memory_gib=a.reserve_memory_gib,
        max_rss_gib=a.max_rss_gib,
        command=command,
    )
    print(json.dumps(info, indent=2), flush=True)
    if a.dry_run:
        return 0
    lock = (run / "resource_guard.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("A resource guard is already running for this directory")
    if available < a.reserve_memory_gib * GIB:
        write_status(run, dict(info, state="insufficient_memory_before_start"))
        return 75
    # Verify the frozen inputs before any resume. The driver also checks its own identity.
    from veudf.cli.prepare_remaining_run import digest

    assert (
        digest(run / "selected_manifest.jsonl")
        == preparation["selected_manifest_sha256"]
    )
    assert digest(run / "config.json") == preparation["config_sha256"]
    inventory = json.loads((run / "code_inventory.json").read_text())
    assert all(digest(run / "code" / name) == h for name, h in inventory.items())
    current_identity = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json; from veudf.preprocess.pipeline import code_fingerprint, runtime_identity; print(json.dumps(dict(code=code_fingerprint(), runtime=runtime_identity())))",
            ],
            env=env,
            text=True,
        )
    )
    if current_identity != preparation["source_identity"]:
        raise RuntimeError(
            "Frozen code or dependencies changed; restore the prepared environment"
        )
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    (run / "guard.pid").write_text(str(os.getpid()))
    with (run / "build.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (run / "driver.pid").write_text(str(child.pid))
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while child.poll() is None:
            _, available = memory_budget()
            rss = tree_rss(child.pid)
            pressure = (
                available < a.reserve_memory_gib * GIB or rss > a.max_rss_gib * GIB
            )
            state = "stopping_on_memory_pressure" if pressure else "building"
            write_status(
                run,
                dict(
                    info,
                    state=state,
                    driver_pid=child.pid,
                    available_gib=available / GIB,
                    job_rss_gib=rss / GIB,
                ),
            )
            if pressure or stopped:
                child.terminate()
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    for process in psutil.Process(child.pid).children(recursive=True):
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    child.kill()
                    child.wait()
                write_status(
                    run,
                    dict(
                        info,
                        state="stopped_on_memory_pressure" if pressure else "cancelled",
                        driver_pid=child.pid,
                        available_gib=available / GIB,
                        job_rss_gib=rss / GIB,
                    ),
                )
                return 75 if pressure else 130
            time.sleep(5)
    write_status(
        run,
        dict(
            info,
            state=(
                "cancelled"
                if stopped
                else ("driver_finished" if child.returncode == 0 else "driver_failed")
            ),
            returncode=child.returncode,
        ),
    )
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
