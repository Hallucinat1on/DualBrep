"""Rebuild and verify the historical 20 dev input meshes before freezing a run."""
import argparse
from pathlib import Path
import time
import numpy as np
from run_mesh_input_experiment import MODULE, REPO, read, save, digest


def preflight(reference, native_reference):
    from veudf.preprocess.config import Config
    from veudf.preprocess.cad import read_normalized, collect_targets, triangulate
    from veudf.preprocess.condition import weld
    from veudf.preprocess.filtering import check_solid_faces, audit_shape
    from veudf.preprocess.pipeline import validate_geometry
    cfg=Config().validate()
    rows=[r for r in read(reference/'manifest.json') if r['cohort']=='dev']
    if len(rows)!=20 or len({r['id'] for r in rows})!=20:
        raise RuntimeError('Expected the fixed 20 distinct dev IDs')
    cases=[]
    for row in rows:
        started=time.monotonic()
        if digest(row['step'])!=row['source_sha256']:raise RuntimeError('Changed STEP: '+row['id'])
        solid,normalization=read_normalized(row['step'],cfg)
        check_solid_faces(solid,cfg)
        targets=collect_targets(solid,cfg)
        validate_geometry(targets,cfg);audit_shape(solid,targets,cfg)
        mesh,oracle=weld(triangulate(targets,cfg),cfg)
        old=read(native_reference/'runs'/row['id']/'source_audit.json')
        counts=dict(faces=len(targets.faces),edges=len(targets.edges),vertices=len(targets.vertices))
        delta=float(np.max(np.abs(np.asarray(normalization['world_to_normalized'])
                                  -np.asarray(old['normalization']['world_to_normalized']))))
        case=dict(id=row['id'],source_sha256=row['source_sha256'],
                  source_solid_count=normalization['source_solid_count'],
                  selected_solid_policy=normalization['selected_solid_policy'],
                  mesh_fingerprint=oracle.fingerprint,native_mesh_matches=oracle.fingerprint==old['mesh']['sha256'],
                  normalization_max_delta=delta,target_counts=counts,
                  target_counts_match=counts==old['target_counts'],seconds=time.monotonic()-started)
        cases.append(case)
        print(row['id'],case['native_mesh_matches'],case['target_counts_match'],delta,flush=True)
    return dict(checked_unix=time.time(),cases=cases,
                all_twenty_match=all(c['native_mesh_matches'] and c['target_counts_match']
                                    and c['normalization_max_delta']<=1e-10 for c in cases),
                code={str(p.relative_to(REPO)):digest(p) for p in sorted((REPO/'veudf/src/veudf').rglob('*.py'))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=MODULE/'outputs/mesh_density_20261028_v2')
    parser.add_argument('--native-reference',type=Path,default=MODULE/'outputs/native_mesh_L5_20261101_v1')
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Use a new preflight output file')
    result=preflight(args.reference.resolve(),args.native_reference.resolve())
    save(args.output,result)
    if not result['all_twenty_match']:raise RuntimeError('Production input mismatch; do not launch')


if __name__=='__main__':main()
