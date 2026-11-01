# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""Model-independent mesh cleanup and conversion helpers."""

import tempfile
from pathlib import Path
import subprocess
import numpy as np
import pymeshlab
import trimesh


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    if max_facenum > mesh.current_mesh().face_number():
        return mesh

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True,
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter(
        "compute_selection_by_small_disconnected_components_per_face", nbfaceratio=0.005
    )
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "mesh.ply"
        mesh.save_current_mesh(str(temp_file))
        mesh = trimesh.load(str(temp_file))
    # 检查加载的对象类型
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        # 如果是Scene，遍历所有的geometry并合并
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "mesh.ply"
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(str(temp_file))
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(str(temp_file))
    return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    scale_factor = 1.2
    vtx_pos = np.asarray(mesh.vertices)
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = (
        torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max()
        * 2.0
    )

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


def simplify_mesh(mesh, executable):
    """Run an explicitly supplied mesh simplifier on geometry only."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "input.obj"
        destination = Path(temp_dir) / "output.obj"
        mesh.export(source)
        subprocess.run([str(executable), str(source), str(destination)], check=True)
        result = trimesh.load(destination, process=False)
        if isinstance(result, trimesh.Scene):
            result = trimesh.util.concatenate(tuple(result.geometry.values()))
        return mesh_normalize(result)


def remove_degenerate(mesh):
    """Reload a PLY through MeshLab, matching the existing cleanup operation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "mesh.ply"
        mesh.save_current_mesh(str(path))
        result = pymeshlab.MeshSet()
        result.load_new_mesh(str(path))
    return result
