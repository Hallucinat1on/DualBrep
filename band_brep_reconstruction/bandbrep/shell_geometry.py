"""Geometric tube sections and sphere fits. This module never queries a field."""
from itertools import pairwise

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .deadline import checkpoint
from .schemas import VertexNode
from .surface_mesh import components


def section_loops(segments, tolerance):
    """Assemble a section's segments without closing missing arcs."""
    samples = segments.reshape(-1, 2)
    _, ids, inv = np.unique(np.rint(samples / tolerance).astype(np.int64),
                            axis=0, return_index=True, return_inverse=True)
    points = samples[ids]
    pairs = np.unique(np.sort(inv.reshape(-1, 2), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    _, labels = components(pairs, len(points))
    order = np.argsort(labels, kind='stable')
    groups = np.split(order, np.cumsum(np.bincount(labels))[:-1])
    links = [[] for _ in points]
    for a, b in pairs:
        links[a].append(b)
        links[b].append(a)
    for group in groups:
        if len(group) < 5 or any(len(links[i]) != 2 for i in group):
            continue
        head = group[0]
        ring = [head]
        prev, node = head, links[head][0]
        while node != head:
            ring.append(node)
            nxt = links[node][1] if links[node][0] == prev else links[node][0]
            prev, node = node, nxt
        yield points[ring]


def ring_center(ring, radius, h):
    """Area center handles oblique elliptical sections; screen distorted rings."""
    origin = ring.mean(0)
    ring = ring - origin
    nxt = np.roll(ring, -1, axis=0)
    cross = ring[:, 0]*nxt[:, 1] - ring[:, 1]*nxt[:, 0]
    twice_area = cross.sum()
    if abs(twice_area) < .15 * h*h:
        return None
    center = np.sum((ring+nxt)*cross[:, None], axis=0) / (3*twice_area)
    # A long oblique section is poor evidence even if this component is closed.
    lengths = np.linalg.norm(nxt-ring, axis=1)
    radial = ring-center
    covariance = (radial.T * (lengths+np.roll(lengths,1))) @ radial
    spectrum, basis = np.linalg.eigh(covariance)
    if spectrum[0] < .09*spectrum[-1]:
        return None
    projected = radial @ basis
    spans = np.ptp(projected, axis=0)
    if min(spans) < .55*h or max(spans) > 5*radius:
        return None
    # Most axes give oblique sections. Their area centroids are stable; for
    # near-circular rings improve the center using a weighted circle fit.
    if spans.max() <= 1.18 * spans.min():
        design = np.column_stack((2*ring, np.ones(len(ring))))
        weight = np.sqrt(lengths+np.roll(lengths,1))
        fit = np.linalg.lstsq(weight[:,None]*design, weight*np.sum(ring*ring,axis=1),rcond=None)[0]
        if np.linalg.norm(fit[:2]-center) < .5*h:
            center = fit[:2]
    return origin+center


def component_sections(vertices, triangles, cfg):
    """Slice a connected shell on three axes, including oblique sections."""
    if not len(triangles):
        return np.empty((0, 3))
    tri = vertices[triangles]
    centers = []
    h = cfg.fine_spacing
    for axis in range(3):
        planar = [d for d in range(3) if d != axis]
        coord = tri[:, :, axis]
        # Relatively prime offset avoids the MC voxel planes themselves.
        lo = np.ceil(coord.min(1)/h-.371).astype(int)
        hi = np.floor(coord.max(1)/h-.371).astype(int)
        for k in range(int(lo.min()), int(hi.max())+1):
            checkpoint('shell_sections')
            active = (lo <= k) & (hi >= k)
            local = tri[active]
            if not len(local):
                continue
            level = (k+.371)*h
            delta = coord[active]-level
            src = np.roll(local, -1, axis=1)
            other = np.roll(delta, -1, axis=1)
            crosses = (delta >= 0) != (other >= 0)
            keep = crosses.sum(1) == 2
            idx = crosses[keep]
            a, b = delta[keep][idx], other[keep][idx]
            start, end = local[keep][idx], src[keep][idx]
            segments = (start+(a/(a-b))[:,None]*(end-start))[:,planar].reshape(-1,2,2)
            for ring in section_loops(segments, h*2e-4) if len(segments) else ():
                center = ring_center(ring, cfg.shell_radius, h)
                if center is not None:
                    p = np.full(3, level)
                    p[planar] = center
                    centers.append(p)
    if not centers:
        return np.empty((0, 3))
    points = np.asarray(centers)
    _, ids = np.unique(np.rint(points/(h*.25)).astype(int),axis=0,return_index=True)
    return points[np.sort(ids)]


def tube_sections(vertices, triangles, cfg):
    """Preserve shell component identities through center extraction."""
    if not len(triangles):
        return np.empty((0, 3)), np.empty(0, int)
    pairs=triangles[:,[0,1,1,2,2,0]].reshape(-1,2)
    _, vertex_labels=components(pairs,len(vertices))
    labels=vertex_labels[triangles[:,0]]
    points, owners = [], []
    for label in np.unique(labels):
        local=component_sections(vertices, triangles[labels==label], cfg)
        points.append(local)
        owners.append(np.full(len(local),label,int))
    return np.vstack(points),np.concatenate(owners)


def fit_sphere(points, weights):
    mean = np.average(points, axis=0, weights=weights)
    q = points-mean
    design = np.column_stack((2*q,np.ones(len(q))))
    fit, _, rank, _ = np.linalg.lstsq(weights[:,None]*design, weights*np.sum(q*q,axis=1), rcond=None)
    if rank < 4:
        return None
    initial = np.r_[fit[:3], np.sqrt(max(fit[3]+fit[:3]@fit[:3],1e-16))]
    opt = least_squares(lambda p: weights*(np.linalg.norm(q-p[:3],axis=1)-p[3]), initial,
                       loss='soft_l1', f_scale=.15*initial[3], max_nfev=20)
    center, radius = mean + opt.x[:3], opt.x[3]
    radial = points-center
    lengths = np.linalg.norm(radial,axis=1)
    direction = radial/np.maximum(lengths[:,None],1e-15)
    coverage = np.linalg.norm(np.average(direction,axis=0,weights=weights))
    return center, float(radius), float(np.quantile(abs(lengths-radius),.9)), float(coverage)


def shell_vertices(shell, faces, grid, cfg):
    """Fit V spheres with free radius; cached metric V can refine a shell estimate."""
    if not len(faces):
        return [], []
    pairs = faces[:, [0,1,1,2,2,0]].reshape(-1,2)
    _, labels = components(pairs,len(shell))
    areas = np.linalg.norm(np.cross(shell[faces[:,1]]-shell[faces[:,0]],
                                    shell[faces[:,2]]-shell[faces[:,0]]),axis=1)
    raw_weights = np.bincount(faces.ravel(),weights=np.repeat(areas,3),minlength=len(shell))
    valid = grid.valid[:,2] & (grid.values[:,2] <= 2*cfg.vertex_level)
    q,d = grid.points[valid], grid.values[valid,2]
    tree = cKDTree(q)
    fitted, rejected = [], []
    h = cfg.fine_spacing
    for label in np.unique(labels):
        ids = (labels==label) & (raw_weights>0)
        fit = fit_sphere(shell[ids],np.sqrt(raw_weights[ids]/raw_weights[ids].mean())) if ids.sum()>=6 else None
        if fit is None:
            rejected.append(dict(reason='vertex_shell_insufficient', component=int(label)))
            continue
        center,radius,residual,coverage = fit
        if not (.45*h <= radius <= 2*cfg.vertex_level and residual <= .6*h and coverage <= .63):
            rejected.append(dict(reason='vertex_shell_ambiguous', component=int(label),
                                 radius=radius, residual=residual, coverage=coverage))
            continue
        fitted.append((center,radius,residual))
    if not fitted:
        return [], rejected
    pos = np.asarray([f[0] for f in fitted])
    gaps = cKDTree(pos).query(pos,k=2)[0][:,1] if len(pos)>1 else np.full(len(pos),np.inf)
    refined = []
    for i,(center,radius,residual) in enumerate(fitted):
        ids = tree.query_ball_point(center,min(2.5*radius,.42*gaps[i]))
        if len(ids)>=8:
            samples, distance = q[ids], d[ids]
            # Fit a free constant bias as well as position. Reuse the cache only.
            def residuals(p):
                return np.linalg.norm(samples-center-p[:3],axis=1)+p[3]-distance
            opt = least_squares(residuals,np.zeros(4),loss='soft_l1',f_scale=.25*h,max_nfev=20)
            conditioned = np.linalg.svd(opt.jac,compute_uv=False)
            if (conditioned[-1]>.05*conditioned[0] and np.linalg.norm(opt.x[:3]) < .6*h
                    and np.quantile(abs(opt.fun),.9)<.3*h):
                center=center+opt.x[:3]
        refined.append((center,residual))
    pos=np.asarray([x[0] for x in refined])
    gaps=cKDTree(pos).query(pos,k=2)[0][:,1] if len(pos)>1 else np.full(len(pos),np.inf)
    return [VertexNode(i,c,e,min(2*h,.4*gaps[i])) for i,(c,e) in enumerate(refined)],rejected


def sphere_hits(polyline, center, radius):
    """Sorted intersections with the observed polyline, classified by direction."""
    hits=[]
    for i,(a,b) in enumerate(pairwise(polyline)):
        d=b-a;v=a-center
        aa=d@d;bb=d@v;cc=v@v-radius*radius
        disc=bb*bb-aa*cc
        if aa<=1e-24 or disc<=1e-24:
            continue  # A tangent is not an entrance or exit.
        root=np.sqrt(disc)
        for direction,t in [(-1,(-bb-root)/aa),(1,(-bb+root)/aa)]:
            if 0<=t<1:
                hits.append((i+t,a+t*d,direction))
    return hits
