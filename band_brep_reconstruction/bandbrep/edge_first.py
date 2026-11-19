"""Mesh + continuous E/V -> parameterized wireframe -> mesh cut -> STEP."""
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cache import cache_key, file_hash
from .deadline import checkpoint, timed_reconstruction
from .edge_first_embedding import cut_surface
from .edge_first_graph import recover_wireframe
from .edge_first_grid import sample_grid
from .edge_first_mc import extract_shell
from .field_provider import MeshFieldProvider
from .schemas import require, write_json
from .surface_fitting import initialize_face_surfaces
from .surface_mesh import validate_mesh


class Run:
    """Stage diagnostics remain readable after a partial run."""

    def __init__(self, out, mesh, udf, cfg):
        identity, inputs = cache_key(mesh, udf, cfg)
        out.mkdir(parents=True, exist_ok=True)
        require(not any(out.iterdir()), 'output_exists', 'Use an empty output directory')
        self.out = out
        self.start = self.tick = time.monotonic()
        self.stage = 'mesh'
        self.report = dict(status='running', extraction=cfg.extraction, config=cfg.to_dict(),
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


def run_route(mesh, provider, grid, cfg, run):
    out = run.out
    run.stage = 'wireframe'
    shell = None
    shell_report = {}
    if cfg.extraction == 'mc' and provider.presence()[0] == 'present':
        tick = time.monotonic()
        shell, triangles, shell_report = extract_shell(grid, cfg)
        shell_report['seconds'] = time.monotonic() - tick
        np.savez_compressed(out / 'edge_shell.npz', vertices=shell, triangles=triangles)
    graph, curves, support = recover_wireframe(grid, provider, cfg, shell)
    write_json(out / 'wireframe.json', graph)
    write_json(out / 'curves.json', curves)
    np.savez_compressed(out / 'edge_support.npz', points=support)
    wireframe = graph.report['wireframe']
    run.record({**wireframe, 'shell': shell_report})
    incomplete = (provider.presence()[0] == 'present' and
                  (not curves or wireframe['unresolved'] or wireframe.get('uncovered_support')))
    if cfg.stop_after == 'wireframe':
        status = 'wireframe_partial' if incomplete else 'wireframe_ready'
        return run.finish(status, graph=graph, curves=curves)
    require(not incomplete, 'wireframe_unresolved', 'Unresolved branches remain in the wireframe')

    def save_cut(cut, constraints):
        np.savez_compressed(out / 'split_mesh.npz', vertices=cut.vertices,
                            triangles=cut.triangles, constraints=constraints)

    run.stage = 'split_mesh'
    cut, labels, constraints = cut_surface(mesh, provider.oracle, graph, curves, cfg, save_partial=save_cut)
    np.savez_compressed(out / 'split_mesh.npz', vertices=cut.vertices, triangles=cut.triangles,
                        face_labels=labels, constraints=constraints)
    write_json(out / 'logical_graph.json', graph)
    run.record(dict(faces=graph.face_count, triangles=len(cut.triangles), constraints=len(constraints)))
    if cfg.stop_after == 'split_mesh':
        return run.finish('split_mesh_ready', graph=graph, curves=curves, mesh=cut, face_labels=labels)

    run.stage = 'surfaces'
    patches = initialize_face_surfaces(cut, labels, graph, curves, provider, cfg)
    write_json(out / 'face_patches.json', patches)
    run.record(dict(faces=len(patches)))

    run.stage = 'cad'
    from .occ_assembly import assemble_brep, verify_export_and_roundtrip

    solid, mapping = assemble_brep(graph, curves, patches, cfg, out)
    cad = verify_export_and_roundtrip(solid, graph, curves, patches, mapping, cfg, out)
    write_json(out / 'logical_to_cad.json', mapping)
    run.record(cad)
    run.report['step_sha256'] = file_hash(out / 'reconstruction.step')
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
        if cfg.extraction != 'both':
            return run_route(mesh, provider, grid, cfg, run)

        run.stage = 'comparison'
        results = {}
        for method in ('trace', 'mc'):
            checkpoint(run.stage)
            route_cfg = replace(cfg, extraction=method)
            route = Run(run.out / method, mesh, udf, route_cfg)
            route.report['scout_artifact'] = '../scout.npz'
            route.write()
            try:
                results[method] = run_route(mesh, provider, grid, route_cfg, route)
            except Exception as exc:
                route.fail(exc)
                results[method] = dict(report=route.report)
                if getattr(exc, 'code', None) == 'sample_timeout':
                    raise
            run.report.setdefault('routes', {})[method] = route.report['status']
            run.write()
        failed = [name for name, status in run.report['routes'].items() if status == 'failed']
        require(not failed, 'edge_first_comparison_failed', 'At least one route failed; see route reports',
                routes=failed)
        partial = any(status == 'wireframe_partial' for status in run.report['routes'].values())
        return run.finish('comparison_partial' if partial else 'comparison_complete', routes=results)
    except Exception as exc:
        run.fail(exc)
        raise
