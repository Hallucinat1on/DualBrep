"""Tests for BrepForge §4.2 narrow-face boundary prior and unified point-cloud reconstruction."""
import numpy as np

from bandbrep.boundary_prior import boundary_prior_samples
from bandbrep.config import Config
from bandbrep.curve_fitting import CurveRecord
from bandbrep.edge_first import reconstruct_point_cloud
from bandbrep.schemas import CoedgeUse
from bandbrep.surface_mesh import normals as mesh_normals
from helpers import box


def test_boundary_prior_samples_narrow_planar_strip():
    """Newell + 2D PCA boundary prior synthesizes clean interior points on a 0.0015 narrow strip."""
    pts = np.array([
        [-0.5, 0.0, -0.00075],
        [0.5, 0.0, -0.00075],
        [0.5, 0.0, 0.00075],
        [-0.5, 0.0, 0.00075],
    ])
    curves = {
        i: CurveRecord(i, 'line', dict(p0=pts[i], p1=pts[(i + 1) % 4]), 0.0)
        for i in range(4)
    }
    cycle = [CoedgeUse(0, i, True) for i in range(4)]
    ref_n = np.array([[0.0, -1.0, 0.0]])
    p_out, n_out = boundary_prior_samples([cycle], curves, Config(), ref_normals=ref_n)
    assert len(p_out) >= 16
    np.testing.assert_allclose(p_out[:, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(n_out, np.broadcast_to(ref_n, n_out.shape), atol=1e-10)


def test_reconstruct_point_cloud_end_to_end(tmp_path):
    """Pure point cloud -> Screened Poisson proxy mesh -> 8D vector_pull -> split_mesh -> raw-point fit -> STEP."""
    mesh, udf = box()
    rng = np.random.default_rng(0)
    tri = mesh.vertices[mesh.triangles]
    tri_n = mesh_normals(mesh)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(tri), size=15000, p=areas / areas.sum())
    r = np.sqrt(rng.random(len(pick)))
    s = rng.random(len(pick))
    bary = np.column_stack((1.0 - r, r * (1.0 - s), r * s))
    raw_pts = np.einsum('ni,nij->nj', bary, tri[pick])
    raw_nrms = tri_n[pick]

    result = reconstruct_point_cloud(
        raw_pts, udf, Config(face_samples=200), tmp_path, normals=raw_nrms, poisson_depth=8
    )
    graph = result['graph']
    assert (graph.face_count, len(graph.edges), len(graph.vertices)) == (6, 12, 8)
    assert result['report']['stages']['cad']['roundtrip_checked']
    assert (tmp_path / 'reconstruction.step').is_file()
