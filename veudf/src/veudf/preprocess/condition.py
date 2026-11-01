"""Final geometry-only CAD mesh. Welding never changes triangle order or face provenance."""

import numpy as np

from veudf.preprocess.cad import PreprocessError


def weld(mesh, cfg):
    """Weld per-face triangulations; the condition requires a closed, consistent mesh."""
    import trimesh

    v, f = mesh[:2]
    welded = trimesh.Trimesh(v, f, process=False)
    welded.merge_vertices(digits_vertex=7)
    welded.remove_unreferenced_vertices()
    if not welded.is_watertight or not welded.is_winding_consistent:
        raise PreprocessError(
            "condition_mesh_not_watertight",
            "CAD mesh does not weld to a closed surface",
        )
    if welded.volume <= 0:
        raise PreprocessError(
            "condition_mesh_not_watertight", "Welded mesh has nonpositive volume"
        )
    from veudf.geometry.mesh import MeshSDFOracle

    v, f = np.asarray(welded.vertices, np.float64), np.asarray(welded.faces, np.int64)
    # Keep the triangle -> original CAD face mapping aligned; no face removal/repair.
    if len(f) != len(mesh[1]):
        raise PreprocessError(
            "condition_mesh_invalid", "Welding changed triangle count"
        )
    try:
        oracle = MeshSDFOracle(v, f)
    except ValueError as exc:
        raise PreprocessError("condition_mesh_invalid", str(exc)) from exc
    return (v, f), oracle
