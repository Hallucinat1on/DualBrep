"""Compare a completed edge-first run with its normalized source CAD."""
import argparse
import json
from pathlib import Path

import numpy as np

from bandbrep.cache import file_hash
from bandbrep.curve_fitting import CurveRecord
from bandbrep.schemas import EdgeInstance, LogicalGraph, SurfaceMesh, VertexNode
from experiment_scoring import fidelity_score, geometry_score, offline_source, score


def evaluate(source_step, output):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from veudf.preprocess.cad import collect_targets, read_normalized, triangulate
    from veudf.preprocess.condition import weld
    from veudf.preprocess.config import Config

    run = json.loads((output / 'run.json').read_text())
    if run['status'] != 'solid_validated':
        raise ValueError('Evaluate a completed, validated STEP run')
    if file_hash(source_step) != run.get('source_step_sha256'):
        raise ValueError('Source STEP identity does not match the reconstruction')
    step = output / 'reconstruction.step'
    if file_hash(step) != run['step_sha256']:
        raise ValueError('Reconstructed STEP changed after validation')

    cfg = Config().validate()
    shape, normalization = read_normalized(source_step, cfg)
    targets = collect_targets(shape, cfg)
    raw = triangulate(targets, cfg)
    _, oracle = weld(raw, cfg)
    source = offline_source(targets, raw, oracle, normalization)

    region_file = 'face_regions.npz' if run.get('face_method') == 'coarse_projection' else 'split_mesh.npz'
    with np.load(output / region_file, allow_pickle=False) as archive:
        mesh = SurfaceMesh(archive['vertices'], archive['triangles'])
        face_labels = archive['face_labels']
    data = json.loads((output / 'logical_graph.json').read_text())
    vertices = [VertexNode(**{**v, 'position': np.asarray(v['position'])}) for v in data['vertices']]
    edges = [EdgeInstance(**{**e, 'samples': np.asarray(e['samples'])}) for e in data['edges']]
    graph = LogicalGraph(data['face_count'], vertices, edges)
    curves = {}
    for key, item in json.loads((output / 'curves.json').read_text()).items():
        item['data'] = {k: np.asarray(v) if isinstance(v, list) else v for k, v in item['data'].items()}
        curves[int(key)] = CurveRecord(**item)

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError('Unable to read reconstructed STEP')
    return dict(topology=score(mesh, face_labels, graph, curves, source),
                geometry=geometry_score(vertices, curves, source),
                fidelity=fidelity_score(reader.OneShape(), source))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.step, args.run), indent=2))
