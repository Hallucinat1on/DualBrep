"""Production Voronoi backend: subprocess wrapper around the C++ executable.

The C++ ``calculate_voronoi`` executable remains the source of the Voronoi
partition; this module adds timeout handling, logging, output validation, and
provenance tracking for cache reuse.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .common import PipelineError, atomic_write_text, sha256_file, write_json


def run_voronoi(executable: Path, step_path: Path, work_dir: Path,
                timeout_seconds: int, resume: bool, overwrite: bool) -> None:
    required = (work_dir / "voronoi.ply", work_dir / "normalized_mesh.ply",
                work_dir / "normalized_params.txt")
    marker = work_dir / "voronoi_backend.json"
    if resume and all(path.is_file() and path.stat().st_size > 0 for path in required) and marker.is_file():
        provenance = json.loads(marker.read_text())
        if provenance.get("backend") == "cpp" and provenance.get("step_sha256") == sha256_file(step_path):
            return
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PipelineError("missing_voronoi_executable", f"not executable: {executable}")
    if work_dir.exists():
        if not overwrite:
            raise PipelineError("partial_work_exists", f"incomplete work directory: {work_dir}")
        archive = work_dir.with_name(f"{work_dir.name}.failed-{int(time.time())}")
        os.replace(work_dir, archive)
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = subprocess.run(
            [str(executable), str(step_path), str(work_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        atomic_write_text(work_dir / "voronoi.log", exc.stdout or "")
        raise PipelineError("voronoi_timeout", f"exceeded {timeout_seconds}s") from exc
    atomic_write_text(work_dir / "voronoi.log", result.stdout)
    if result.returncode != 0:
        raise PipelineError("voronoi_crash", f"exit code {result.returncode}")
    missing = [str(path.name) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise PipelineError("voronoi_empty", f"missing outputs: {', '.join(missing)}")
    write_json(marker, {
        "backend": "cpp",
        "experimental": False,
        "step_sha256": sha256_file(step_path),
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
    })
