#!/usr/bin/env python3
"""Check or import the authoritative production veudf implementation.

Only package source, tests, configuration and packaging files are managed.
README, provenance, datasets, frozen jobs and generated files are never copied.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/opt/data/private/yihengxu/Hunyuan3D/veudf')
TOP_LEVEL = ('pyproject.toml', 'requirements.txt', 'environment.yml')
GROUPS = (('src', '*.py'), ('tests', '*.py'), ('configs', '*.json'))


def managed_files(directory):
    paths = [directory / name for name in TOP_LEVEL]
    for name, pattern in GROUPS:
        paths.extend(p for p in (directory / name).rglob(pattern)
                     if '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
    return sorted(paths)


def inventory(directory):
    result = {}
    for path in managed_files(directory):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Expected a regular source file: {path}')
        result[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(name.startswith('src/veudf/') for name in result):
        raise ValueError(f'No veudf package in {directory}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--apply', action='store_true', help='Replace the local mirror with production files')
    parser.add_argument('--check', action='store_true', help='Read-only comparison (the default)')
    args = parser.parse_args(argv)
    if args.apply and args.check:
        parser.error('Choose --apply or --check')
    source = args.source.resolve()
    destination = ROOT / 'veudf'
    if source == destination.resolve() or destination.resolve() in source.parents:
        parser.error('Source must be outside the local mirror')
    expected = inventory(source)
    if args.apply:
        for name in expected:
            target = destination / name
            if target.is_symlink():
                raise ValueError(f'Refusing to overwrite symlink: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        # Prune only files within the explicitly managed source/config/test sets.
        for path in managed_files(destination):
            if path.relative_to(destination).as_posix() not in expected:
                path.unlink()
        if inventory(source) != expected or inventory(destination) != expected:
            raise RuntimeError('Source changed during sync or local copy differs; repeat the sync')
        record = dict(authoritative_source=str(source), policy='production_is_authoritative',
                      managed_files=expected)
        (destination / 'SOURCE.json').write_text(json.dumps(record, indent=2) + '\n')
    actual = inventory(destination) if (destination / 'pyproject.toml').exists() else {}
    different = sorted(name for name in expected if actual.get(name) != expected[name])
    extra = sorted(set(actual) - set(expected))
    marker = destination / 'SOURCE.json'
    recorded = json.loads(marker.read_text()) if marker.exists() else {}
    metadata_matches = recorded.get('managed_files') == expected and recorded.get('authoritative_source') == str(source)
    passed = not different and not extra and metadata_matches
    print(json.dumps(dict(matches=passed, files=len(expected), changed_or_missing=different,
                          extra=extra, provenance_matches=metadata_matches), indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
