"""Dataset channels, pools and geometry sampling versions."""

import numpy as np

SCHEMA = "edge_vertex_surface_udf_v1"
CHANNELS = ("dE", "dV")
POOLS = ("surface", "edge_surface", "vertex_surface")
SOURCES = {name: i for i, name in enumerate(POOLS)}
# Per-channel flags; invalid/absent distances are placeholders, not labels.
VALID = 1
TIE = 2
EMPTY_DISTANCE = 2 * np.sqrt(3)
GEOMETRY_VERSION = "cad_mesh_area_face_normal_v1"
