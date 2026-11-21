"""Mesh + continuous E/V -> parameterized wireframe -> mesh regions -> STEP."""
import shutil
import time
from pathlib import Path

import numpy as np

from .cache import cache_key, file_hash
from .deadline import timed_reconstruction
from .edge_first_embedding import cut_surface
from .edge_first_graph import recover_wireframe
from .edge_first_grid import sample_grid
from .edge_first_mc import extract_shell
from .face_patches import fit_with_patch_fallback
from .fast_faces import coarse_paths, recover_planar_faces
from .field_provider import MeshFieldProvider
from .schemas import ReconstructionError, require, write_json
from .surface_mesh import mesh_edges, validate_mesh


class Run:
    """Stage diagnostics remain readable after a partial run."""

    def __init__(self, out, mesh, udf, cfg):
        identity, inputs = cache_key(mesh, udf, cfg)
        out.mkdir(parents=True, exist_ok=True)
        require(not any(out.iterdir()), 'output_exists', 'Use an empty output directory')
        self.out = out
        self.start = self.tick = time.monotonic()
        self.stage = 'mesh'
        self.report = dict(status='running', extraction='thinning', config=cfg.to_dict(),
                           input_identity=identity, inputs=inputs, stages={})
        if 'source_step_sha256' in mesh.report:
            self.report['source_step_sha256'] = mesh.report['source_step_sha256']
        self.write()

    def write(self):
        write_json(self.out / 'run.json', self.report)

    def record(self, report):
        now = time.monotonic()
        self.report['stages'][self.stage] = dict(seconds=now - self.tick, **report)
        self.tick = now
        self.write()

    def finish(self, status, **objects):
        self.report.update(status=status, elapsed_seconds=time.monotonic() - self.start)
        self.write()
        write_json(self.out / 'quality.json', self.report)
        return dict(report=self.report, **objects)

    def fail(self, exc):
        self.report.update(failed_stage=self.stage,
                           failed_stage_seconds=(0.0 if self.stage in self.report['stages']
                                                 else time.monotonic() - self.tick),
                           error=dict(code=getattr(exc, 'code', type(exc).__name__),
                                      message=str(exc), details=getattr(exc, 'details', {})))
        self.finish('failed')


def export(graph, curves, patches, cfg, out):
    from .occ_assembly import assemble_brep, verify_export_and_roundtrip

    solid, mapping = assemble_brep(graph, curves, patches, cfg, out)
    cad = verify_export_and_roundtrip(solid, graph, curves, patches, mapping, cfg, out)
    return solid, mapping, cad


def try_fast(mesh, graph, curves, cfg, run):
    out = run.out
    run.stage = 'fast_faces'
    try:
        proposed, patches, labels = recover_planar_faces(mesh, graph, curves, cfg)
    except (ReconstructionError, np.linalg.LinAlgError) as exc:
        if getattr(exc, 'code', None) == 'sample_timeout':
            raise
        run.record(dict(accepted=False, reason=getattr(exc, 'code', type(exc).__name__),
                        details=getattr(exc, 'details', {})))
        return None
    run.record(dict(accepted=True, faces=proposed.face_count, **proposed.report['coarse_projection']))
    run.stage = 'cad'
    trial = out / 'planar_trial'
    trial.mkdir()
    try:
        solid, mapping, cad = export(proposed, curves, patches, cfg, trial)
    except ReconstructionError as exc:
        if exc.code == 'sample_timeout':
            raise
        run.stage = 'fast_cad'
        run.record(dict(accepted=False, reason=exc.code))
        return None
    for item in trial.iterdir():
        shutil.move(item, out / item.name)
    trial.rmdir()
    np.savez_compressed(out / 'face_regions.npz', vertices=mesh.vertices, triangles=mesh.triangles, face_labels=labels)
    write_json(out / 'logical_graph.json', proposed)
    write_json(out / 'face_patches.json', patches)
    write_json(out / 'logical_to_cad.json', mapping)
    run.record(cad)
    run.report.update(step_sha256=file_hash(out / 'reconstruction.step'), face_method='coarse_projection')
    return run.finish('solid_validated', solid=solid, graph=proposed, curves=curves, patches=patches)


def run_route(mesh, provider, grid, cfg, run):
    out = run.out
    run.stage = 'wireframe'
    tick = time.monotonic()
    v_shell, v_faces, v_report = extract_shell(grid, cfg, channel=2)
    shell_report = dict(vertex=v_report, seconds=time.monotonic() - tick)
    np.savez_compressed(out / 'vertex_shell.npz', vertices=v_shell, triangles=v_faces)
    graph, curves, support, skeleton = recover_wireframe(grid, provider, cfg, v_shell, v_faces)
    points, links, _ = skeleton
    np.savez_compressed(out / 'edge_skeleton.npz', points=points, links=links)
    write_json(out / 'wireframe.json', graph)
    write_json(out / 'curves.json', curves)
    np.savez_compressed(out / 'edge_support.npz', points=support)
    wireframe = graph.report['wireframe']
    incomplete = provider.presence()[0] == 'present' and bool(
        not curves or wireframe['unresolved'] or wireframe['uncovered_support'] or wireframe['warnings'])
    continue_partial = incomplete and cfg.allow_partial_wireframe and bool(curves)
    run.record({**wireframe, 'shell': shell_report, 'incomplete': incomplete,
                'continued_with_warnings': continue_partial and cfg.stop_after != 'wireframe'})
    if cfg.stop_after == 'wireframe':
        status = 'wireframe_partial' if incomplete else 'wireframe_ready'
        return run.finish(status, graph=graph, curves=curves)
    require(not incomplete or continue_partial, 'wireframe_unresolved',
            'Unresolved branches remain in the wireframe' if curves else 'No edge could be recovered',
            unresolved=len(wireframe['unresolved']), uncovered_support=wireframe['uncovered_support'])

    if cfg.fast_faces and not cfg.hybrid_projection and cfg.stop_after == 'solid':
        fast = try_fast(mesh, graph, curves, cfg, run)
        if fast is not None:
            return fast

    def save_cut(cut, constraints):
        np.savez_compressed(out / 'split_mesh.npz', vertices=cut.vertices,
                            triangles=cut.triangles, constraints=constraints)

    paths = {}
    if cfg.hybrid_projection:
        run.stage = 'coarse_projection'
        radius = min(cfg.fine_spacing, cfg.feature_sdf_tolerance)
        paths, _, projection = coarse_paths(mesh, graph, curves, radius, cfg.fine_spacing,
                                            mesh_edges(mesh.triangles), allow_partial=True,
                                            anchor_tolerance=min(radius, cfg.boundary_tolerance))
        run.record(dict(coarse_edges=sorted(paths), precise_edges=sorted(set(curves) - paths.keys()),
                        **projection))

    run.stage = 'split_mesh'
    for _attempt in range(5 if cfg.allow_feature_collapse else 1):
        try:
            cut, labels, constraints = cut_surface(mesh, provider.oracle, graph, curves, cfg,
                                                   save_partial=save_cut, coarse_paths=paths)
            break
        except ReconstructionError as exc:
            e1 = exc.details.get('edge')
            e2 = exc.details.get('other_edge')
            if (not cfg.allow_feature_collapse or exc.code != 'curve_embedding_overlap'
                    or e1 is None or e2 is None or e1 == e2 or _attempt == 4):
                raise
            # Drop the shorter overlapping edge and mark the surviving edge as collapsed.
            def _elen(eid):
                pts = curves[eid].evaluate(np.linspace(0, 1, 33))
                return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            drop, keep = (e1, e2) if _elen(e1) <= _elen(e2) else (e2, e1)
            new_edges, new_curves, new_paths = [], {}, {}
            old_collapsed = set(graph.report.get('wireframe', {}).get('collapsed_edges', []))
            old_collapsed.add(keep)
            new_collapsed = []
            for e in graph.edges:
                if e.id == drop:
                    continue
                nid = len(new_edges)
                old_id = e.id
                e.id = nid
                c = curves[old_id]
                c.id = nid
                new_edges.append(e)
                new_curves[nid] = c
                if old_id in paths:
                    new_paths[nid] = paths[old_id]
                if old_id in old_collapsed:
                    new_collapsed.append(nid)
            graph.edges = new_edges
            curves = new_curves
            paths = new_paths
            graph.report.setdefault('wireframe', {})['collapsed_edges'] = new_collapsed
    np.savez_compressed(out / 'split_mesh.npz', vertices=cut.vertices, triangles=cut.triangles,
                        face_labels=labels, constraints=constraints)
    write_json(out / 'logical_graph.json', graph)
    run.record(dict(faces=graph.face_count, triangles=len(cut.triangles), constraints=len(constraints)))
    if cfg.stop_after == 'split_mesh':
        return run.finish('split_mesh_ready', graph=graph, curves=curves, mesh=cut, face_labels=labels)

    run.stage = 'surfaces'
    patches, assembly_graph, assembly_curves, face_map, seams = fit_with_patch_fallback(
        cut, labels, constraints, graph, curves, provider, cfg)
    write_json(out / 'face_patches.json', patches)
    run.record(dict(faces=len(patches), patch_seams=seams))

    run.stage = 'cad'
    solid, mapping, cad = export(assembly_graph, assembly_curves, patches, cfg, out)
    mapping['patch_to_logical_face'] = face_map
    mapping['internal_patch_seams'] = seams
    if seams:
        write_json(out / 'assembly_graph.json', assembly_graph)
        write_json(out / 'assembly_curves.json', assembly_curves)
    write_json(out / 'logical_to_cad.json', mapping)
    run.record(cad)
    method = 'hybrid_projection' if cfg.hybrid_projection else 'cut_mesh'
    run.report.update(step_sha256=file_hash(out / 'reconstruction.step'), face_method=method)
    return run.finish('solid_validated', solid=solid, graph=graph, curves=curves, patches=patches)


@timed_reconstruction
def reconstruct(mesh, udf, cfg, output_dir):
    """Run edge-first reconstruction with scalar E/V and no truth topology."""
    cfg.validate()
    run = Run(Path(output_dir), mesh, udf, cfg)
    try:
        mesh = validate_mesh(mesh, cfg)
        provider = MeshFieldProvider(mesh, udf, cfg)
        provider.band_width = min(provider.band_width, cfg.band_width)
        run.stage = 'scout'
        grid = sample_grid(mesh, provider, cfg)
        np.savez_compressed(run.out / 'scout.npz', points=grid.points, keys=grid.keys, values=grid.values,
                            valid=grid.valid, blocks=grid.blocks, block_indices=grid.block_indices)
        run.record(grid.report)
        if cfg.stop_after == 'scout':
            return run.finish('scout_ready')
        return run_route(mesh, provider, grid, cfg, run)
    except Exception as exc:
        run.fail(exc)
        raise
