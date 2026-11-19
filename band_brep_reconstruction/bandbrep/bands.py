"""Face cores from single-pass segmentation define propagation masks on a fixed mesh."""
import numpy as np
from .schemas import BandState, require
from .surface_mesh import evaluate_batched, face_adjacency
from types import SimpleNamespace
from .vendor.dualbrep_segmentation import segment_faces
from .vertex_domains import local_vertex_domains


def build_bands(mesh, provider, cfg, center_fields=None):
    presence = provider.presence()
    require(all(p != 'unknown' for p in presence), 'presence_unknown',
            'Supply edge/vertex existence; missed minima do not imply absence')
    points = mesh.vertices[mesh.triangles].mean(axis=1)
    values = (evaluate_batched(provider, points, cfg.query_batch_size) if center_fields is None
              else center_fields.validate(points, provider))
    udf = values[:, 1] if presence[0] == 'present' else np.full(len(points), max(cfg.face_threshold1, cfg.face_avg_udf_threshold)+1.)
    count, labels, reasons, segmentation = segment_faces(
        SimpleNamespace(face_adjacency=face_adjacency(mesh.triangles)), udf,
        cfg.face_threshold1, cfg.face_filter_size, cfg.face_avg_udf_threshold)
    core = labels >= 0
    eb = ~core if presence[0] == 'present' else np.zeros(len(points), bool)
    vertex_labels = vertex_centers = None
    vertex_report = {'method': 'absent'}
    vb = np.zeros(len(points), bool)
    if presence[1] == 'present':
        vertex_labels, vertex_centers, vertex_report = local_vertex_domains(
            points, values, eb, reasons == 1, provider, cfg)
        vb = vertex_labels >= 0
    segmentation.update(count=count, sizes=np.bincount(labels[core]).tolist(),
        method='dualbrep_first_pass_triangle_components', threshold1=cfg.face_threshold1,
        filter_size=cfg.face_filter_size, avg_udf_threshold=cfg.face_avg_udf_threshold,
        reason_codes={'core':0,'first_threshold':1,'size_rejected':3,'mean_rejected':4})
    report = dict(nodes=len(points), edge_nodes=int(eb.sum()), vertex_nodes=int(vb.sum()),
                  regular_nodes=int((eb & ~vb).sum()), core_nodes=int(core.sum()),
                  excluded_nodes=int((~core & ~eb).sum()),
                  exclusion_counts={str(i):int((reasons==i).sum()) for i in (0, 1, 3, 4)},
                  sampling='triangle_centers', vertex_domain='nested_in_edge_band',
                  vertex_domains=vertex_report,
                  segmentation=segmentation,
                  field_query_points=len(points) if center_fields is None else 0,
                  reused_center_queries=center_fields is not None, strict_topology_guarantee=False)
    return mesh, BandState(points, values, eb, vb, eb & ~vb, core, report, labels, reasons, vertex_labels, vertex_centers)
