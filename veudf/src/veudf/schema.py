"""Dataset channels, pools and geometry sampling versions."""

import numpy as np

SCHEMA = "edge_vertex_mesh_pools_v1"
CHANNELS = ("s_mesh", "dE", "dV")
POOLS = ("surface", "near_surface", "edge_surface", "vertex_surface")
SOURCES = {name: i for i, name in enumerate((*POOLS, "zero"))}
EMPTY_DISTANCE = 2 * np.sqrt(3)
GEOMETRY_VERSION = "cad_mesh_area_face_normal_v1"
