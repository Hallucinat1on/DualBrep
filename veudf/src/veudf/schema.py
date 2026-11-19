"""Dataset channels, pools and geometry sampling versions."""

import numpy as np

SCHEMA = "edge_vertex_neighborhood_pools_v3"
CHANNELS = ("s_mesh", "dE", "dV")
POOLS = ("surface", "near_surface", "edge_neighborhood", "vertex_neighborhood")
ZERO_POOLS = ("edge_zero", "vertex_zero")
ALL_POOLS = (*POOLS, *ZERO_POOLS)
SOURCES = {name: i for i, name in enumerate(ALL_POOLS)}
SURFACE, SPATIAL, ZERO = 0, 1, 2
EMPTY_DISTANCE = 2 * np.sqrt(3)
GEOMETRY_VERSION = "cad_mesh_area_face_normal_v1"
