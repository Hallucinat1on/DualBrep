"""Offline nearest-support diagnostics; these are not topology certificates."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def dominant(component, truth, count, truth_count):
    active=(component>=0)&(truth>=0)
    if not count:return np.empty(0,int),np.empty(0,int),np.empty(0,bool)
    h=coo_matrix((np.ones(active.sum()),(component[active],truth[active])),shape=(count,max(truth_count,1))).toarray()
    size=h.sum(1).astype(int);sorted_h=np.sort(h,axis=1)
    second=sorted_h[:,-2] if h.shape[1]>1 else np.zeros(count)
    owner=h.argmax(1);owner[size==0]=-1
    return owner,size,second>=np.maximum(3,.05*size)


def score(topology, bands, source):
    targets=source.targets;gt=targets.topology
    nf,ne,nv=len(targets.faces),len(targets.edges),len(targets.vertices)
    points=bands.points;labels=topology.face_labels
    cached=getattr(source,'_scoring_truth',None)
    if cached is None or cached[0] is not points:
        cached=(points,source.truth_face(points),
            source.edge.query(points)[2] if source.edge else np.full(len(points),-1),
            source.vtree.query(points)[1] if source.vtree else np.full(len(points),-1))
        source._scoring_truth=cached
    face_truth=cached[1]
    fd,fs,fm=dominant(labels,face_truth,topology.face_count,nf)
    perface=np.bincount(fd[fd>=0],minlength=nf)
    face_good=bool(topology.face_count==nf and np.all(fs>0) and not fm.any() and np.all(perface==1))
    et,vt=cached[2:]
    ed,es,em=dominant(topology.edge_labels,et,len(topology.edges),ne)
    centers=getattr(bands,'vertex_centers',None)
    if source.edge is not None and centers is not None:
        t=np.linspace(.1,.9,9)
        for c,item in enumerate(topology.edges):
            if item.get('status')!='inferred_short' or es[c]>0:continue
            a,b=centers[item['vertices'][0]],centers[item['vertices'][1]]
            owners=source.edge.query((1-t[:,None])*a+t[:,None]*b)[2]
            counts=np.bincount(owners,minlength=ne)
            ed[c]=int(counts.argmax());es[c]=len(t);em[c]=bool(np.sort(counts)[-2]>=3) if ne>1 else False
    vd,vs,vm=dominant(topology.vertex_labels,vt,len(topology.vertices),nv)
    pec=np.bincount(ed[ed>=0],minlength=ne);pvc=np.bincount(vd[vd>=0],minlength=nv)
    eok=np.zeros(ne,bool);vok=np.zeros(nv,bool)
    for eid in range(ne):
        ids=np.flatnonzero(ed==eid)
        if len(ids)==1:
            c=int(ids[0]);item=topology.edges[c]
            eok[eid]=not em[c] and {int(fd[f]) for f in item['faces']}==set(gt['edges'][eid]['face_ids']) and len(item['faces'])==2
    for vid in range(nv):
        ids=np.flatnonzero(vd==vid)
        if len(ids)==1:
            c=int(ids[0]);item=topology.vertices[c]
            vok[vid]=not vm[c] and {int(ed[e]) for e in item['edges']}==set(gt['vertices'][vid]['edge_ids'])
    edge_vertices_ok=True
    for c,item in enumerate(topology.edges):
        if ed[c]<0 or ed[c]>=ne:
            edge_vertices_ok=False;continue
        expected=set(gt['edges'][ed[c]]['vertex_ids'])
        if {int(vd[v]) for v in item['vertices']}!=expected:edge_vertices_ok=False
    graph_good=bool(face_good and len(ed)==ne and eok.all() and len(vd)==nv and vok.all() and edge_vertices_ok)
    return {'truth':{'faces':nf,'edges':ne,'vertices':nv},
        'predicted':{'faces':topology.face_count,'edges':len(ed),'vertices':len(vd)},
        'face_good':face_good,'missing_faces':int((perface==0).sum()),'split_faces':int((perface>1).sum()),'mixed_cores':int(fm.sum()),
        'edge_ok':int(eok.sum()),'vertex_ok':int(vok.sum()),'missing_edges':int((pec==0).sum()),'fragmented_edges':int((pec>1).sum()),
        'missing_vertices':int((pvc==0).sum()),'duplicated_vertices':int((pvc>1).sum()),
        'mixed_edges':int(em.sum()),'mixed_vertices':int(vm.sum()),'tiny_edges':int((es<3).sum()),
        'unresolved_edges':sum(len(e['faces'])!=2 for e in topology.edges),'graph_good':graph_good,
        'edge_vertex_relation_good':edge_vertices_ok,
        'face_map':fd.tolist(),'edge_map':ed.tolist(),'vertex_map':vd.tolist(),
        'method':'All triangle centers; nearest source triangle / dense edge / vertex. Mixed if second owner >=max(3,5%). Not exact geometry certification.'}


def fidelity_score(solid, source, deflection=5e-4):
    """Offline: volume error and two-sided surface distance to the CAD truth mesh."""
    import igl
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods
    BRepMesh_IncrementalMesh(solid, deflection, False, .15, False).Perform()
    vs, fs, offset = [], [], 0
    ex = TopExp_Explorer(solid, TopAbs_FACE)
    while ex.More():
        face = topods.Face(ex.Current()); loc = TopLoc_Location(); tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            t = loc.Transformation()
            v = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(i).Transformed(t) for i in range(1, tri.NbNodes()+1))])
            f = np.array([[tri.Triangle(i).Value(k)-1 for k in (1, 2, 3)] for i in range(1, tri.NbTriangles()+1)])
            if face.Orientation() == TopAbs_REVERSED: f = f[:, ::-1]
            vs.append(v); fs.append(f+offset); offset += len(v)
        ex.Next()
    rv, rf = np.vstack(vs), np.vstack(fs)
    gv, gf = np.asarray(source.mesh[0], float), np.asarray(source.mesh[1], np.int64)
    def samples(v, f, n=20000, seed=0):
        rng = np.random.default_rng(seed); tri = v[f]
        area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
        pick = rng.choice(len(f), n, p=area/area.sum()); r = np.sqrt(rng.random(n)); s = rng.random(n)
        b = np.column_stack((1-r, r*(1-s), r*s)); return np.einsum('ni,nij->nj', b, tri[pick])
    d_rg = np.sqrt(igl.point_mesh_squared_distance(samples(rv, rf), gv, gf)[0])
    d_gr = np.sqrt(igl.point_mesh_squared_distance(samples(gv, gf, seed=1), rv, rf)[0])
    props = GProp_GProps(); brepgprop.VolumeProperties(solid, props)
    truth = float(source.normalization['normalized_volume'])
    return {'volume': float(props.Mass()), 'truth_volume': truth, 'volume_rel_error': float(abs(props.Mass()-truth)/truth),
            'recon_to_truth_max': float(d_rg.max()), 'truth_to_recon_max': float(d_gr.max()),
            'recon_to_truth_mean': float(d_rg.mean()), 'truth_to_recon_mean': float(d_gr.mean()),
            'method': f'20k area-weighted samples per side; OCC mesh deflection {deflection}; truth = CAD oracle triangulation'}


def geometry_score(vertices, curves, source):
    result={}
    if vertices and len(source.targets.vertices):
        p=np.array([v.position for v in vertices]);d=cKDTree(source.targets.vertices).query(p)[0]
        result['vertex_nearest_max']=float(d.max())
    if curves and source.edge:
        samples=np.vstack([c.evaluate(np.linspace(0,1,2049)) for c in curves.values()])
        de=source.edge.query(samples)[0];sdf=np.abs(source.sdf(samples))
        truth=source.edge.points[::max(1,len(source.edge.points)//50000)]
        reverse=cKDTree(samples).query(truth)[0]
        result.update(curve_to_truth_max=float(de.max()),truth_to_curve_sampled_max=float(reverse.max()),curve_sdf_max=float(sdf.max()))
    return result
