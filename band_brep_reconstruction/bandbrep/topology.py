"""Exact set-valued flood equivalence on fixed domains; vertex domains are sinks."""
import numpy as np
from .face_seeds import core_components
from .surface_mesh import face_adjacency
from .schemas import BandTopology


def collect_labels(pairs, bands, face_labels):
    """R components collect sources. V components absorb face AND edge IDs.

    No V -> R transfer exists, including for labels received after the first.
    The returned component references broadcast full sets to every member node.
    """
    el = core_components(pairs, bands.edge_region)
    vl = core_components(pairs, bands.vertex_band)
    ef = [set() for _ in range(int(el.max()) + 1)]
    vf = [set() for _ in range(int(vl.max()) + 1)]
    ve = [set() for _ in vf]
    contacts = []
    for a, b in pairs:
        if ((face_labels[a] >= 0 and face_labels[b] >= 0) or
                (el[a] >= 0 and el[a] == el[b]) or (vl[a] >= 0 and vl[a] == vl[b])):
            continue
        contacts.append((int(a), int(b)))
        for src, dst in ((a, b), (b, a)):
            if face_labels[src] >= 0:
                if el[dst] >= 0: ef[el[dst]].add(int(face_labels[src]))
                if vl[dst] >= 0: vf[vl[dst]].add(int(face_labels[src]))
            if el[src] >= 0 and vl[dst] >= 0:
                ve[vl[dst]].add(int(el[src]))
    # Full union only after all R sources have been collected; never feed it back.
    for v, incident in enumerate(ve):
        for e in incident: vf[v].update(ef[e])
    edges = [dict(id=e, faces=sorted(faces), vertices=[],
                  nodes=np.flatnonzero(el == e).tolist(),
                  status='supported' if len(faces) == 2 else 'unresolved')
             for e, faces in enumerate(ef)]
    vertices = [dict(id=v, faces=sorted(vf[v]), edges=sorted(ve[v]),
                     nodes=np.flatnonzero(vl == v).tolist()) for v in range(len(vf))]
    for v in vertices:
        for e in v['edges']: edges[e]['vertices'].append(v['id'])
    report = dict(faces=int(face_labels.max()+1), edges=len(edges), vertices=len(vertices),
                  unresolved_edges=[e['id'] for e in edges if e['status'] != 'supported'],
                  vertex_domains_without_edges=[v['id'] for v in vertices if not v['edges']],
                  propagation='component_union_with_absorbing_vertex_domains',
                  vertex_labels_complete=True, orientation_deferred=True)
    return BandTopology(int(face_labels.max()+1),face_labels,el,vl,edges,vertices,
                        np.asarray(contacts,dtype=np.int64).reshape(-1,2),report)


def recover_topology(mesh,bands,face_labels):
    return collect_labels(face_adjacency(mesh.triangles),bands,face_labels)
