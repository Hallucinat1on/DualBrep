"""Select remaining STEP candidates and freeze a resumable preprocessing run.

This prepares files only; it never starts CAD processing.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from veudf.preprocess.config import load_config


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def select_remaining(full, previous, limit, seed, preserve_order=False):
    old_hashes = defaultdict(list)
    old_paths = defaultdict(list)
    old_source_groups = defaultdict(list)
    for r in previous:
        old_hashes[r["source_sha256"]].append(r)
        old_paths[os.path.abspath(r["step_path"])].append(r)
        if r.get("source_group_id"):
            old_source_groups[r["source_group_id"]].append(r)
    pins = defaultdict(set)
    for r in full:
        matches = old_hashes.get(r["source_sha256"], []) + old_paths.get(
            os.path.abspath(r["step_path"]), []
        ) + old_source_groups.get(r.get("source_group_id"), [])
        for old in matches:
            pins[r["group_id"]].add((old["split"], old["group_id"]))
    conflicts = {g for g, p in pins.items() if len({s for s, _ in p}) > 1}
    eligible = []
    excluded = Counter()
    for original in full:
        if (
            original["source_sha256"] in old_hashes
            or os.path.abspath(original["step_path"]) in old_paths
        ):
            excluded["previous_path_or_content"] += 1
            continue
        if "duplicate_of" in original:
            excluded["byte_duplicate"] += 1
            continue
        group = original["group_id"]
        if group in conflicts:
            excluded["conflicting_old_family_splits"] += 1
            continue
        r = dict(original)
        if group in pins:
            r["split"] = next(iter(pins[group]))[0]
            r["group_id"] = min(g for _, g in pins[group])
        eligible.append(r)
    if not preserve_order:
        eligible.sort(
            key=lambda r: (
                hashlib.sha256(f"{seed}:{r['source_sha256']}".encode()).digest(),
                r["sample_id"],
            )
        )
    selected = eligible[:limit]
    return selected, dict(
        full_records=len(full),
        previous_records=len(previous),
        eligible=len(eligible),
        selected=len(selected),
        candidate_limit=limit,
        selection_seed=seed,
        selection_order="manifest" if preserve_order else "content_hash",
        excluded=dict(excluded),
        conflicting_global_groups=sorted(conflicts),
        splits=dict(Counter(r["split"] for r in selected)),
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Full STEP-root manifest")
    p.add_argument("--exclude-root", required=True, action="append", help="Previously built dataset; repeat for multiple roots")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preserve-order", action="store_true", help="Select candidates in manifest order")
    p.add_argument("--config", help="Preprocessing JSON; defaults to Config defaults")
    a = p.parse_args(argv)
    if a.limit < 1:
        p.error("--limit must be positive")
    output = Path(a.run_dir).resolve()
    if output.exists():
        p.error("Run directory already exists; resume with veudf-guarded-build")
    full_path = Path(a.manifest).resolve()
    previous_paths = [Path(root).resolve() / "manifest.jsonl" for root in a.exclude_root]
    previous = [r for path in previous_paths for r in records(path)]
    selected, summary = select_remaining(
        records(full_path), previous, a.limit, a.seed, preserve_order=a.preserve_order
    )
    if not selected:
        p.error("No remaining candidates")
    run = output / "run"
    source = Path(__file__).resolve().parents[1]
    project = source.parents[1]
    source_tree = project.parent if (project / "pyproject.toml").is_file() else source
    if output.is_relative_to(source_tree):
        p.error("--run-dir must be outside the source tree")
    code = run / "code"
    code.mkdir(parents=True)
    shutil.copytree(
        source,
        code / "veudf",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"),
    )
    manifest = run / "selected_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in selected))
    (run / "config.json").write_text(
        json.dumps(load_config(a.config).to_dict(), indent=2)
    )
    env = dict(os.environ, PYTHONPATH=str(code))
    identity = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json; from veudf.preprocess.pipeline import code_fingerprint, runtime_identity; "
                "print(json.dumps(dict(code=code_fingerprint(), runtime=runtime_identity())))",
            ],
            env=env,
            text=True,
        )
    )
    inventory = {
        str(f.relative_to(run / "code")): digest(f)
        for f in (run / "code").rglob("*")
        if f.is_file() and "__pycache__" not in f.parts
    }
    (run / "code_inventory.json").write_text(json.dumps(inventory, indent=2))
    summary.update(
        full_manifest=str(full_path),
        full_manifest_sha256=digest(full_path),
        previous_manifest=str(previous_paths[0]),
        previous_manifest_sha256=digest(previous_paths[0]),
        previous_manifests=[dict(path=str(path), sha256=digest(path)) for path in previous_paths],
        selected_manifest_sha256=digest(manifest),
        config_sha256=digest(run / "config.json"),
        source_identity=identity,
    )
    (run / "preparation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
