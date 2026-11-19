"""Parameterize E-shell sections and V-shell centers without UDF gradients."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .curve_geometry import closest_parameters, projected_seed
from .curve_fitting import CurveRecord, fit_circle, fit_fixed_arc
from .deadline import checkpoint
from .edge_first_grid import query
from .edge_geometry import circle_order, order_general
from .schemas import EdgeInstance, LogicalGraph, ReconstructionError, require
from .shell_geometry import shell_vertices, sphere_hits, tube_sections
from .surface_mesh import components


@dataclass
class Path:
    nodes: list
    points: np.ndarray
    closed: bool
    ports: list | None = None


def local_tangents(points, tree, radius):
    result = []
    for point, neighbors in zip(points, tree.query_ball_point(points, radius)):
        local = points[neighbors] - point
        _, vec = np.linalg.eigh(local.T @ local)
        result.append(vec[:, -1])
    return np.asarray(result)


def clip_ports(ordered, ids, vertices):
    """Use exact segment intersections instead of dropping the last inside sample."""
    if not ids:
        return ordered, []
    hits = []
    for vid, direction in ((ids[0], 1), (ids[-1], -1)):
        vertex = vertices[vid]
        options = [hit for hit in sphere_hits(ordered, vertex.position, vertex.support_radius)
                   if hit[2] == direction]
        require(options, 'vertex_port_unresolved', 'Center path has no usable sphere crossing', vertex=vid)
        t, point, _ = options[0 if direction == 1 else -1]
        hits.append((t, point))
    (lo, first), (hi, last) = hits
    require(hi > lo, 'vertex_ports_overlap', 'Cut balls cover the entire ordered branch')
    interior = ordered[int(np.ceil(lo)):int(np.floor(hi))+1]
    return np.vstack((first, interior, last)), [first.tolist(), last.tolist()]


def paths_from_sections(cloud, owners, vertices, cfg):
    h = cfg.fine_spacing
    if not len(cloud):
        return [], []
    positions = np.array([v.position for v in vertices]).reshape(-1, 3)
    radii = np.array([v.support_radius for v in vertices])
    # Associate branches on the outside of junctions. Keep the original
    # sections for coverage and for independent short-edge evidence.
    if len(vertices):
        gap, near = cKDTree(positions).query(cloud)
        outside=gap >= radii[near]
        cloud,owners = cloud[outside],owners[outside]
    if not len(cloud):
        return [], []
    tree = cKDTree(cloud)
    pairs = tree.query_pairs(2.25*h, output_type='ndarray')
    pairs = pairs[owners[pairs[:,0]]==owners[pairs[:,1]]]
    tangents = local_tangents(cloud, tree, 2.8*h)
    delta = cloud[pairs[:,1]]-cloud[pairs[:,0]]
    length = np.linalg.norm(delta,axis=1)
    direction = delta/np.maximum(length[:,None],1e-15)
    alignment = abs((tangents[pairs]*direction[:,None,:]).sum(2))
    keep = (alignment > .38).all(1) | (length < .38*h)
    _,labels = components(pairs[keep],len(cloud))
    paths, rejected = [], []
    for group in np.unique(labels):
        points = cloud[labels==group]
        if len(points)<3:
            continue
        ids = np.flatnonzero(cKDTree(points).query(positions)[0]<=radii+2*h).tolist()
        if len(ids)>2:
            rejected.append(dict(reason='ambiguous_branch_endpoints',vertices=ids,points=len(points)))
            continue
        ends = positions[ids]
        try:
            if len(points)>=6:
                circle,error=fit_circle(points)
                ordered=circle_order(points,circle,ends,cfg) if error<=max(cfg.curve_tolerance,.5*h) else None
            else:
                ordered=None
            if ordered is None:
                ordered=order_general(points,ends,cfg)
            if len(ids)==1:
                # A self-loop must approach the same ball from two directions.
                p0,p1=ordered[[1,-2]]-ends[0]
                require(max(np.linalg.norm(p0),np.linalg.norm(p1)) <= radii[ids[0]]+2*h
                        and p0@p1 < .85*np.linalg.norm(p0)*np.linalg.norm(p1),
                        'vertex_port_unresolved','One-port branch is not a supported self-loop')
            path, ports = clip_ports(ordered,ids,vertices)
            paths.append(Path(ids,path,not ids,ports))
        except (ReconstructionError,np.linalg.LinAlgError) as exc:
            rejected.append(dict(reason=getattr(exc,'code','curve_order_failed'),vertices=ids,
                                 details=getattr(exc,'details',{})))
    return paths,rejected


def add_short_paths(paths, vertices, grid, cfg):
    """Keep short segments only if cached off-axis observations support them."""
    if len(vertices)<2:
        return
    h=cfg.fine_spacing
    positions=np.array([v.position for v in vertices])
    connected={tuple(sorted(p.nodes)) for p in paths if len(p.nodes)==2}
    valid=grid.valid[:,1] & (grid.values[:,1] <= cfg.shell_radius)
    q,d=grid.points[valid],grid.values[valid,1]
    tree=cKDTree(q)
    for a,b in sorted(cKDTree(positions).query_pairs(5*h)):
        if (a,b) in connected:
            continue
        p0,p1=positions[[a,b]]
        axis=p1-p0
        length=np.linalg.norm(axis)
        if length<=1e-10:
            continue
        ids=tree.query_ball_point((p0+p1)*.5,.5*length+h)
        points=q[ids]
        t=(points-p0)@axis/(axis@axis)
        radial=np.linalg.norm(points-p0-t[:,None]*axis,axis=1)
        keep=(t>.12)&(t<.88)&(radial<=h)
        if keep.sum()<4 or np.ptp(t[keep]) < .35:
            continue
        residual=radial[keep]-d[ids][keep]
        bias=np.median(residual)
        if abs(bias)>.5*h or np.quantile(abs(residual-bias),.9)>.25*h:
            continue
        occupied=(p0+np.linspace(.2,.8,7)[:,None]*axis)
        if any(cKDTree(p.points).query(occupied)[0].max()<.3*length for p in paths):
            continue
        paths.append(Path([a,b], np.vstack((p0,occupied,p1)), False))


def with_endpoints(points, ends, closed):
    if len(ends)==1:
        ends=np.repeat(ends,2,axis=0)
    if len(ends)==2:
        points=np.vstack((ends[0], points, ends[1]))
    elif closed and np.linalg.norm(points[0]-points[-1])>1e-10:
        points=np.vstack((points, points[0]))
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-10]
    return points[keep],ends


def fit_geometry(edge_id, path, ends, cfg):
    """Line/circle/C1 Bézier with fixed endpoints, using only geometric samples."""
    points, ends = with_endpoints(path.points, ends, path.closed)
    tol=cfg.curve_tolerance
    if len(ends)==2 and np.linalg.norm(ends[1]-ends[0])>1e-10:
        line=CurveRecord(edge_id,'line',dict(p0=ends[0],p1=ends[1]),0.)
        error=closest_parameters(line,points)[1].max()
        if error<=tol:
            line.residual=float(error)
            return line
    if len(points)>=6:
        data,error=fit_circle(points)
        if path.closed and error<=tol and abs(data['sweep'])>5.5:
            data['sweep']=np.sign(data['sweep'])*2*np.pi
            return CurveRecord(edge_id,'circle',data,error)
        if len(ends)==2 and np.linalg.norm(ends[1]-ends[0])>1e-10:
            candidate=fit_fixed_arc(points,*ends)
            if candidate and candidate[1]<=tol:
                return CurveRecord(edge_id,'circle',*candidate)
    periodic=path.closed or (len(ends)==2 and np.linalg.norm(ends[1]-ends[0])<=1e-10)
    for spans in range(4 if periodic else 1, cfg.max_segments+1):
        if 3*spans+1 > 2*len(points)+4:
            break
        checkpoint('geometry_fit')
        breaks=np.linspace(0,1,spans+1)
        record=projected_seed(points,ends,breaks,edge_id,periodic)
        error=closest_parameters(record,points)[1].max()
        if error<=tol:
            record.residual=float(error)
            return record
    require(False,'curve_geometry_failed','Section centers do not support a curve within tolerance',edge=edge_id)


def refine_sections(paths, provider, cfg):
    """Optional sparse scalar samples. Accept only a better supported section center."""
    if cfg.mc_udf_refine == 'off':
        return 0
    offsets=np.array([-.5,0.,.5])*cfg.fine_spacing
    jobs=[]
    for path in paths:
        points=path.points
        # Uniform coarse probes: conservative correction, with a strict budget.
        for i in range(1,len(points)-1,8):
            if len(jobs)>=2048:
                break
            tangent=points[i+1]-points[i-1]
            if np.linalg.norm(tangent)<1e-10:
                continue
            basis=np.linalg.svd(tangent[None,:],full_matrices=True)[2][1:]
            delta=offsets[:,None]*basis[0]+offsets[None,:,None]*basis[1]
            jobs.append((path,i,points[i]+delta.reshape(-1,3)))
    if not jobs:
        return 0
    positions,inverse=np.unique(np.vstack([j[2] for j in jobs]),axis=0,return_inverse=True)
    v,valid=query(provider,positions,cfg.query_batch_size)
    v,valid=v[inverse],valid[inverse]
    count = len(positions)
    for (path,i,probes), values, ok in zip(jobs,v.reshape(-1,9,3)[:,:,1],valid.reshape(-1,9,3)[:,:,1]):
        if not ok.all():
            continue
        # Compare opposite probes rather than chasing a near-zero threshold.
        pairs=np.array([[0,8],[1,7],[2,6],[3,5]])
        baseline=np.mean(abs(values[pairs[:,0]]-values[pairs[:,1]]))
        j=int(values.argmin())
        if (j!=4 and np.linalg.norm(probes[j]-probes[4]) <= .5*cfg.fine_spacing
                and values[4]-values[j] > cfg.field_error and baseline>2*cfg.field_error):
            target=.75*probes[4]+.25*probes[j]
            # Validate tentative movement with a scalar symmetric stencil at
            # the same local orientation; no gradient estimates.
            p=target+(probes-probes[4])
            nv,nok=query(provider,p,cfg.query_batch_size)
            count += len(p)
            distances=nv[:,1]
            score=np.mean(abs(distances[pairs[:,0]]-distances[pairs[:,1]]))
            if nok[:,1].all() and score < baseline:
                path.points[i]=target
    return count


def recover_wireframe(grid, provider, cfg, shell, triangles, v_shell, v_faces):
    started=perf_counter()
    vertices,rejected=shell_vertices(v_shell,v_faces,grid,cfg)
    if provider.presence()[1]=='present' and not vertices:
        rejected.append(dict(reason='vertex_shell_empty'))
    vertices_done=perf_counter()
    cloud,owners=tube_sections(shell,triangles,cfg)
    if len(triangles) and not len(cloud):
        rejected.append(dict(reason='no_trusted_section'))
    sections_done=perf_counter()
    paths,unresolved=paths_from_sections(cloud,owners,vertices,cfg)
    rejected.extend(unresolved)
    add_short_paths(paths,vertices,grid,cfg)
    probes=refine_sections(paths,provider,cfg)
    paths_done=perf_counter()
    edges,curves,ports=[],{},{}
    for path in paths:
        eid=len(edges)
        ends=np.array([vertices[v].position for v in path.nodes]).reshape(-1,3)
        try:
            curve=fit_geometry(eid,path,ends,cfg)
        except (ReconstructionError,np.linalg.LinAlgError) as exc:
            rejected.append(dict(reason=getattr(exc,'code','curve_solve_failed'),vertices=path.nodes))
            continue
        ids=path.nodes if len(path.nodes)!=1 else path.nodes*2
        edges.append(EdgeInstance(eid,(),path.points,path.closed,ids,-1,cfg.support_radius))
        curves[eid]=curve
        if path.ports:
            ports[eid]=path.ports
    used={v for e in edges for v in e.vertices}
    rejected.extend(dict(reason='isolated_vertex',vertex=v.id) for v in vertices if v.id not in used)
    miss=0
    if curves:
        samples=[]
        for curve in curves.values():
            length=np.linalg.norm(np.diff(curve.evaluate(np.linspace(0,1,129)),axis=0),axis=1).sum()
            samples.append(curve.evaluate(np.linspace(0,1,max(129,int(2*length/cfg.fine_spacing)+1))))
        tree=cKDTree(np.vstack(samples))
        miss=int((tree.query(cloud)[0]>2*cfg.fine_spacing).sum())
        # Retain a cache-only check for candidate regions with no shell evidence.
        valid=grid.valid[:,1] & (grid.values[:,1]<=min(cfg.fine_spacing,cfg.shell_radius*.6))
        gap=tree.query(grid.points[valid])[0]
        miss+=int((gap>grid.values[valid,1]+3*cfg.fine_spacing).sum())
    report=dict(vertices=len(vertices),edges=len(edges),sections=len(cloud),unresolved=rejected,
                uncovered_support=miss,optional_e_queries=probes,sphere_ports=ports,
                timings=dict(vertices=vertices_done-started,sections=sections_done-vertices_done,
                             paths=paths_done-sections_done,fit_and_coverage=perf_counter()-paths_done))
    return LogicalGraph(0,vertices,edges,report={'wireframe':report}),curves,cloud
