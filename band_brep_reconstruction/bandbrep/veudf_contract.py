"""Dependency check for the production preprocessing contract."""
from .schemas import require


def require_veudf_v3():
    from veudf.schema import SCHEMA

    expected = "edge_vertex_neighborhood_pools_v3"
    require(
        SCHEMA == expected,
        "veudf_schema_mismatch",
        f"Expected veudf schema {expected}, got {SCHEMA}. Install the repository mirror with pip install -e ./veudf",
        expected=expected,
        actual=SCHEMA,
    )
    return SCHEMA
