"""Set-valued flood on fixed domains; vertex domains never transfer labels back."""
import numpy as np
from scipy.spatial import cKDTree
from .face_seeds import core_components
from .surface_mesh import face_adjacency, vertex_adjacency
from .schemas import BandTopology


def _relations(source, target):
    valid=(source>=0)&(target>=0)
    return np.unique(np.column_stack((target[valid],source[valid])),axis=0)


def _members(labels,count):
    ids=np.flatnonzero(labels>=0)
    order=ids[np.argsort(labels[ids],kind='stable')]
    return np.split(order,np.cumsum(np.bincount(labels[ids],minlength=count))[:-1]) if count else []


def collect_labels(pairs, bands, face_labels):
    el=core_components(pairs,bands.edge_region)
    vl=np.asarray(bands.vertex_labels,np.int64) if bands.vertex_labels is not None else core_components(pairs,bands.vertex_band)
    ef=[set() for _ in range(int(el.max())+1)]
    vf=[set() for _ in range(int(vl.max())+1)]
    ve=[set() for _ in vf]
    a,b=pairs[:,0],pairs[:,1]
    internal=((face_labels[a]>=0)&(face_labels[b]>=0))|((el[a]>=0)&(el[a]==el[b]))|((vl[a]>=0)&(vl[a]==vl[b]))
    contacts=pairs[~internal]
    src=np.r_[contacts[:,0],contacts[:,1]];dst=np.r_[contacts[:,1],contacts[:,0]]
    for target,source in _relations(face_labels[src],el[dst]):ef[target].add(int(source))
    for target,source in _relations(face_labels[src],vl[dst]):vf[target].add(int(source))
    for target,source in _relations(el[src],vl[dst]):ve[target].add(int(source))
    for v,incident in enumerate(ve):
        for e in incident:vf[v].update(ef[e])
    support=np.ones(len(face_labels),bool) if bands.exclusion_reasons is None else np.isin(bands.exclusion_reasons,[1,2])
    em=_members(el,len(ef));vm=_members(vl,len(vf))
    edges=[dict(id=e,faces=sorted(faces),vertices=[],nodes=ids.tolist(),
        support_nodes=ids[support[ids]].tolist(),status='supported' if len(faces)==2 else 'unresolved')
        for e,(faces,ids) in enumerate(zip(ef,em))]
    vertices=[dict(id=v,faces=sorted(vf[v]),edges=sorted(ve[v]),nodes=ids.tolist(),
        support_nodes=ids[support[ids]].tolist()) for v,ids in enumerate(vm)]
    for v in vertices:
        for e in v['edges']:edges[e]['vertices'].append(v['id'])
    report=dict(faces=int(face_labels.max()+1),edges=len(edges),vertices=len(vertices),
        unresolved_edges=[e['id'] for e in edges if e['status']!='supported'],
        vertex_domains_without_edges=[v['id'] for v in vertices if not v['edges']],
        edges_without_geometry_support=[e['id'] for e in edges if not e['support_nodes']],
        vertices_without_geometry_support=[v['id'] for v in vertices if not v['support_nodes']],
        propagation='component_union_with_absorbing_vertex_domains',
        vertex_labels_complete=True,orientation_deferred=True)
    return BandTopology(int(face_labels.max()+1),face_labels,el,vl,edges,vertices,contacts,report)


def infer_short_edges(topology, bands, provider, cfg):
    """Recover edges swallowed when two vertex domains touch with no band between.

    A candidate needs (1) a direct V-V contact, (2) no existing edge joining the
    pair, (3) exactly two shared face labels and (4) e-UDF ~ 0 and SDF ~ 0 along
    the segment between the two vote centers.  Rejected pairs stay in the report.
    """
    report={'contact_pairs':0,'inferred':[],'rejected':[]}
    if bands.vertex_centers is None or provider is None:
        report['skipped']='requires local vertex centers and a field provider';return topology,report
    vl=topology.vertex_labels;c=topology.contacts
    if not len(c):return topology,report
    a,b=vl[c[:,0]],vl[c[:,1]];touch=(a>=0)&(b>=0)&(a!=b)
    pairs=np.unique(np.sort(np.column_stack((a[touch],b[touch])),axis=1),axis=0)
    report['contact_pairs']=len(pairs)
    joined={tuple(sorted(e['vertices'])) for e in topology.edges if len(e['vertices'])==2}
    t=np.linspace(0,1,33)
    for va,vb in pairs.tolist():
        if (va,vb) in joined:continue
        faces=sorted(set(topology.vertices[va]['faces'])&set(topology.vertices[vb]['faces']))
        p0,p1=bands.vertex_centers[va],bands.vertex_centers[vb]
        q=(1-t[:,None])*p0+t[:,None]*p1;v=provider.evaluate(q).values
        entry={'vertices':[va,vb],'shared_faces':faces,'length':float(np.linalg.norm(p1-p0)),
               'max_edge_udf':float(v[:,1].max()),'max_abs_sdf':float(np.abs(v[:,0]).max())}
        if len(faces)>2:
            # Both endpoints may also bound a third face; the edge's own two faces
            # are the ones whose cores approach the middle of the segment.
            middle=q[8:25];distance=[]
            for f in faces:
                core=bands.points[topology.face_labels==f]
                distance.append(float(cKDTree(core).query(middle)[0].min()) if len(core) else np.inf)
            order=np.argsort(distance);entry['core_distances']=dict(zip(map(int,faces),distance))
            if distance[order[2]]>=1.5*distance[order[1]]:faces=sorted(int(faces[i]) for i in order[:2])
        entry['faces']=faces
        ok=len(faces)==2 and entry['max_edge_udf']<=2*cfg.localization_tolerance and entry['max_abs_sdf']<=cfg.surface_tolerance
        if not ok:report['rejected'].append(entry);continue
        eid=len(topology.edges)
        topology.edges.append(dict(id=eid,faces=faces,vertices=[va,vb],nodes=[],support_nodes=[],status='inferred_short'))
        for vid in (va,vb):topology.vertices[vid]['edges']=sorted(set(topology.vertices[vid]['edges'])|{eid})
        report['inferred'].append({**entry,'edge':eid})
    topology.report['edges']=len(topology.edges)
    return topology,report


def recover_topology(mesh,bands,face_labels,cfg=None,provider=None):
    adjacency=vertex_adjacency if cfg is not None and cfg.band_connectivity=='vertex' else face_adjacency
    topology=collect_labels(adjacency(mesh.triangles),bands,face_labels)
    topology.report['connectivity']='vertex' if adjacency is vertex_adjacency else 'edge'
    if cfg is not None and cfg.infer_short_edges:
        topology,topology.report['short_edges']=infer_short_edges(topology,bands,provider,cfg)
    return topology
