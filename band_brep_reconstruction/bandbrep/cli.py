"""Run the Post1109 reconstruction template from a reference STEP."""
import argparse
from pathlib import Path

from .cache import file_hash
from .cad_source import load_cad_source
from .config import load_config
from .deadline import sample_budget
from .reconstruct import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True, help="source CAD used for the mesh and exact E/V")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="override numerical reconstruction parameters")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=("seeds", "graph", "vertices", "curves", "surfaces", "solid"))
    args = parser.parse_args()
    try:
        cfg = load_config(args.config, {"stop_after": args.stop_after})
        with sample_budget(cfg.sample_timeout_seconds):
            mesh, udf = load_cad_source(args.step, file_hash(args.step))
            reconstruct(mesh, udf, cfg, args.output, resume=args.resume)
    except Exception as exc:
        parser.exit(1, f"Reconstruction failed: {exc}. See {args.output / 'quality.json'} if created.\n")
