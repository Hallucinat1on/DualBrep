"""Two-pass face cores define disjoint propagation masks on a fixed mesh."""
import numpy as np
from .schemas import BandState, require
from .surface_mesh import evaluate_batched, face_adjacency
from types import SimpleNamespace
from .vendor.dualbrep_segmentation import hierarchical_segmentation


def build_bands(mesh, provider, cfg):
    presence = provider.presence()
    require(all(p != 'unknown' for p in presence), 'presence_unknown',
            'Supply edge/vertex existence; missed minima do not imply absence')
    points = mesh.vertices[mesh.triangles].mean(axis=1)
    values = evaluate_batched(provider, points, cfg.query_batch_size)
    udf = values[:, 1] if presence[0] == 'present' else np.full(len(points), max(cfg.face_threshold2, cfg.face_avg_udf_threshold)+1.)
    count, labels, reasons, segmentation = hierarchical_segmentation(
        SimpleNamespace(face_adjacency=face_adjacency(mesh.triangles)), udf,
        threshold1=cfg.face_threshold1, threshold2=cfg.face_threshold2,
        filter_size=cfg.face_filter_size, avg_udf_threshold=cfg.face_avg_udf_threshold)
    # A rejected high-distance component is not evidence of an edge.
    eb = np.isin(reasons, [1, 2]) if presence[0] == 'present' else np.zeros(len(points), bool)
    core = labels >= 0
    if cfg.band_propagation == 'noncore' and presence[0] == 'present':
        eb = ~core
    vb = eb & (values[:, 2] < cfg.tau_vertex) if presence[1] == 'present' else np.zeros(len(points), bool)
    vertex_labels = vertex_centers = None
    vertex_report = {'method': 'global_threshold', 'tau_vertex': cfg.tau_vertex}
    if cfg.vertex_domains == 'local' and presence[1] == 'present':
        from .vertex_domains import local_vertex_domains
        support = np.isin(reasons, [1, 2])
        vertex_labels, vertex_centers, vertex_report = local_vertex_domains(points, values, eb, support, provider, cfg)
        vb = vertex_labels >= 0
    segmentation.update(count=count, sizes=np.bincount(labels[core]).tolist(),
        method='dualbrep_two_pass_triangle_components',
        threshold1=cfg.face_threshold1, threshold2=cfg.face_threshold2,
        filter_size=cfg.face_filter_size, avg_udf_threshold=cfg.face_avg_udf_threshold,
        reason_codes={'core':0,'first_threshold':1,'split_threshold':2,'size_rejected':3,
                      'mean_rejected':4,'subcluster_size_rejected':5})
    report = dict(nodes=len(points), edge_nodes=int(eb.sum()), vertex_nodes=int(vb.sum()),
                  regular_nodes=int((eb & ~vb).sum()), core_nodes=int(core.sum()),
                  excluded_nodes=int((~core & ~eb).sum()),
                  exclusion_counts={str(i):int((reasons==i).sum()) for i in range(6)},
                  sampling='triangle_centers', vertex_domain='nested_in_edge_band',
                  vertex_domains=vertex_report,
                  segmentation=segmentation,
                  field_query_points=len(points), strict_topology_guarantee=False)
    return mesh, BandState(points, values, eb, vb, eb & ~vb, core, report, labels, reasons, vertex_labels, vertex_centers)
