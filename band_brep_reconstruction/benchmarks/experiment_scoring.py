"""Offline nearest-support diagnostics; these are not topology certificates."""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree



def offline_source(targets, raw, oracle, normalization):
    """Build scoring references from the source CAD and its triangulated mesh."""
    from offline_support import PolylineDistance

    return SimpleNamespace(
        targets=targets,
        mesh=raw,
        normalization=normalization,
        edge=PolylineDistance(targets.edges) if targets.edges else None,
        vtree=cKDTree(targets.vertices) if len(targets.vertices) else None,
        truth_face=lambda points: raw[2][oracle.closest(points)[1]],
        sdf=oracle.query,
    )


def dominant(component, truth, count, truth_count, weights):
    active=(component>=0)&(truth>=0)
    if not count:return np.empty(0,int),np.empty(0,int),np.empty(0,bool)
    h=coo_matrix((weights[active],(component[active],truth[active])),shape=(count,max(truth_count,1))).toarray()
    size=h.sum(1);sorted_h=np.sort(h,axis=1)
    second=sorted_h[:,-2] if h.shape[1]>1 else np.zeros(count)
    owner=h.argmax(1);owner[size==0]=-1
    return owner,size,(size>0)&(second>=.05*size)


def score(mesh, face_labels, graph, curves, source, tolerance=.002):
    """Offline instance matching on the actual cut mesh and fitted wireframe."""
    targets = source.targets
    nf, ne, nv = len(targets.faces), len(targets.edges), len(targets.vertices)
    tri = mesh.vertices[mesh.triangles]
    centers = tri.mean(axis=1)
    weights = .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    face_map, sizes, mixed = dominant(face_labels, source.truth_face(centers), graph.face_count, nf, weights)
    face_map[(sizes == 0) | mixed] = -1
    vertex_map = np.full(len(graph.vertices), -1, int)
    if nv and len(vertex_map):
        d, index = source.vtree.query([v.position for v in graph.vertices])
        vertex_map[d <= tolerance] = index[d <= tolerance]
    edge_map = np.full(len(graph.edges), -1, int)
    if ne:
        for edge in graph.edges:
            samples = curves[edge.id].evaluate(np.linspace(.05, .95, 65))
            distances, _, indices = source.edge.query(samples)
            ids = indices[distances <= tolerance]
            counts = np.bincount(ids, minlength=ne)
            if counts.max() >= np.ceil(.95 * len(samples)):
                edge_map[edge.id] = counts.argmax()

    def matched_counts(mapping, total):
        counts = np.bincount(mapping[mapping >= 0], minlength=total)
        return dict(matched=int((counts == 1).sum()), missing=int((counts == 0).sum()),
                    duplicated=int((counts > 1).sum()), unmatched=int((mapping < 0).sum()))

    face_counts = matched_counts(face_map, nf)
    edge_counts = matched_counts(edge_map, ne)
    vertex_counts = matched_counts(vertex_map, nv)
    ev_ok = ef_ok = 0
    for edge in graph.edges:
        target_id = edge_map[edge.id]
        if target_id < 0:
            continue
        truth = targets.topology['edges'][target_id]
        ev_ok += {int(vertex_map[v]) for v in edge.vertices} == set(truth['vertex_ids'])
        ef_ok += {int(face_map[f]) for f in edge.faces} == set(truth['face_ids'])
    result = dict(faces=face_counts, edges=edge_counts, vertices=vertex_counts,
                  edge_vertex_matches=ev_ok, edge_face_matches=ef_ok,
                  edge_map=edge_map.tolist(), vertex_map=vertex_map.tolist(), face_map=face_map.tolist())
    result['graph_good'] = bool(
        face_counts['matched'] == nf == graph.face_count
        and edge_counts['matched'] == ne == len(graph.edges)
        and vertex_counts['matched'] == nv == len(graph.vertices)
        and ev_ok == ef_ok == ne
    )
    return result


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
            'recon_to_truth_quantiles': dict(zip(('median', 'p95', 'p99', 'max'), map(float, np.quantile(d_rg, [.5, .95, .99, 1])))),
            'truth_to_recon_quantiles': dict(zip(('median', 'p95', 'p99', 'max'), map(float, np.quantile(d_gr, [.5, .95, .99, 1])))),
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
