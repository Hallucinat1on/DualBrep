"""Surface-field data pipeline package (mesh-conditioned Voronoi-UDF from ABC).

Run from the repository root::

    python -m Voronoi.surface_field run \\
        --abc-root /opt/data/private/yihengxu/Datasets/abc \\
        --output-root /opt/data/private/yihengxu/Datasets/surface \\
        --voronoi-exe Voronoi/build/calculate_voronoi/calculate_voronoi \\
        --limit 3 --condition-points 4096
"""

from .cli import main

__all__ = ["main"]
