"""Fit one shared curve to cached scalar distance observations per edge.

The narrow, instance-labelled support supplies approximately instance-specific
distances. No CAD nearest-entity identity or UDF derivative enters optimization.
"""
import time
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .curve_fitting import CurveRecord, arclength_parameters, fit_circle, fit_fixed_arc
from .deadline import checkpoint
from .schemas import EdgeInstance, LogicalGraph, ReconstructionError, require
from .surface_mesh import evaluate_batched


def representative_indices(points, count, seed=0):
    """Deterministic spatial coverage from a bounded reservoir, not input order."""
    if len(points)<=count:return np.arange(len(points))
    rng=np.random.default_rng(seed)
    pool=np.unique(np.r_[rng.choice(len(points),min(len(points),4*count),replace=False),
                         np.argmin(points,axis=0),np.argmax(points,axis=0)])
    p=points[pool];selected=np.empty(count,int)
    selected[0]=0;distance=np.full(len(p),np.inf)
    for i in range(1,count):
        distance=np.minimum(distance,((p-p[selected[i-1]])**2).sum(1))
        distance[selected[:i]]=-1
        selected[i]=int(np.argmax(distance))
    return pool[selected]


def bezier_basis(t, breaks):
    t=np.asarray(t);k=len(breaks)-1
    span=np.clip(np.searchsorted(breaks,t,side='right')-1,0,k-1)
    u=(t-breaks[span])/(breaks[span+1]-breaks[span])
    values=np.column_stack(((1-u)**3,3*(1-u)**2*u,3*(1-u)*u*u,u**3))
    basis=np.zeros((len(t),3*k+1))
    basis[np.arange(len(t))[:,None],3*span[:,None]+np.arange(4)]=values
    return basis


def curve_derivatives(record,t):
    t=np.asarray(t);d=record.data
    if record.kind=='line':
        return np.broadcast_to(d['p1']-d['p0'],(len(t),3)),np.zeros((len(t),3))
    if record.kind=='circle':
        a=d['start']+t*d['sweep'];x,y=d['x'],d['y']
        first=d['radius']*d['sweep']*(-np.sin(a)[:,None]*x+np.cos(a)[:,None]*y)
        second=-d['radius']*d['sweep']**2*(np.cos(a)[:,None]*x+np.sin(a)[:,None]*y)
        return first,second
    breaks=d['breaks'];span=np.clip(np.searchsorted(breaks,t,side='right')-1,0,len(breaks)-2)
    h=breaks[span+1]-breaks[span];u=(t-breaks[span])/h;p=d['controls'][span]
    first=3*((1-u)[:,None]**2*(p[:,1]-p[:,0])+2*(u*(1-u))[:,None]*(p[:,2]-p[:,1])+u[:,None]**2*(p[:,3]-p[:,2]))/h[:,None]
    second=6*((1-u)[:,None]*(p[:,2]-2*p[:,1]+p[:,0])+u[:,None]*(p[:,3]-2*p[:,2]+p[:,1]))/h[:,None]**2
    return first,second


def closest_parameters(record,points,initial=None):
    """Global coarse seeds plus safeguarded local Newton, with both endpoints.

    This is a numerical closest-point search, not a certified global bound.
    Multiple coarse seeds prevent a single warm start from fixing a wrong span.
    """
    p=np.asarray(points,float);d=record.data
    if record.kind=='line':
        direction=d['p1']-d['p0']
        t=np.clip((p-d['p0'])@direction/max(direction@direction,1e-30),0,1)
        return t,np.linalg.norm(p-record.evaluate(t),axis=1)
    if record.kind=='circle':
        rel=p-d['center'];angle=np.arctan2(rel@d['y'],rel@d['x'])
        phase=np.mod(np.sign(d['sweep'])*(angle-d['start']),2*np.pi)
        t=phase/abs(d['sweep']);outside=t>1
        ends=record.evaluate(np.array([0.,1.]))
        t[outside]=(np.linalg.norm(p[outside]-ends[1],axis=1)<np.linalg.norm(p[outside]-ends[0],axis=1)).astype(float)
        return t,np.linalg.norm(p-record.evaluate(t),axis=1)
    breaks=d['breaks']
    grid=np.unique(np.concatenate([np.linspace(a,b,17) for a,b in zip(breaks[:-1],breaks[1:])]))
    ids=cKDTree(record.evaluate(grid)).query(p,k=3)[1]
    candidates=grid[ids]
    if initial is not None:candidates=np.column_stack((candidates,initial))
    q=np.repeat(p,candidates.shape[1],axis=0);t=candidates.ravel()
    for _ in range(8):
        checkpoint('curve_closest_points')
        c=record.evaluate(t);first,second=curve_derivatives(record,t);delta=c-q
        numerator=np.einsum('ij,ij->i',delta,first)
        denominator=np.einsum('ij,ij->i',first,first)+np.einsum('ij,ij->i',delta,second)
        step=np.zeros_like(t);good=denominator>1e-16
        step[good]=np.clip(numerator[good]/denominator[good],-.1,.1)
        proposed=np.clip(t-step,0,1)
        current_error=(delta*delta).sum(1)
        for _ in range(6):
            bad=((record.evaluate(proposed)-q)**2).sum(1)>current_error+1e-24
            if not bad.any():break
            proposed[bad]=(proposed[bad]+t[bad])*.5
        if np.max(np.abs(proposed-t),initial=0)<1e-11:
            t=proposed;break
        t=proposed
    candidates=np.column_stack((t.reshape(len(p),-1),np.zeros(len(p)),np.ones(len(p))))
    distance=np.linalg.norm(record.evaluate(candidates.ravel()).reshape(len(p),-1,3)-p[:,None],axis=2)
    best=np.argmin(distance,axis=1)
    return candidates[np.arange(len(p)),best],distance[np.arange(len(p)),best]


def distance_residuals(record,points,distances,batch=8192):
    parameters=[];residuals=[]
    for start in range(0,len(points),batch):
        checkpoint('curve_distance_validation')
        t,d=closest_parameters(record,points[start:start+batch])
        parameters.append(t);residuals.append(d-distances[start:start+batch])
    return np.concatenate(parameters),np.concatenate(residuals)


def split_bezier(record,span):
    """Exact de Casteljau subdivision: add freedom without moving the curve."""
    nets=record.data['controls'];p=nets[span];a=(p[:-1]+p[1:])*.5;b=(a[:-1]+a[1:])*.5;c=b.mean(0)
    children=np.array([[p[0],a[0],b[0],c],[c,b[1],a[2],p[3]]])
    return CurveRecord(record.id,'bezier_spline',dict(
        breaks=np.insert(record.data['breaks'],span+1,np.mean(record.data['breaks'][span:span+2])),
        controls=np.concatenate((nets[:span],children,nets[span+1:]))),record.residual)


def _poles(record):
    p=record.data['controls']
    return np.vstack((p[0],*[v[1:] for v in p[1:]]))


def _record(edge_id,breaks,poles):
    return CurveRecord(edge_id,'bezier_spline',dict(breaks=breaks,
        controls=np.array([poles[3*j:3*j+4] for j in range(len(breaks)-1)])),0.)


def _constraints(breaks,periodic,ends=None):
    k=len(breaks)-1;m=3*k+1;rows=[]
    for j in range(1,k):
        row=np.zeros(m);h0=breaks[j]-breaks[j-1];h1=breaks[j+1]-breaks[j]
        row[3*j-1]=-1/h0;row[3*j]=1/h0+1/h1;row[3*j+1]=-1/h1
        rows.append(row)
    if periodic:
        row=np.zeros(m);row[-1]=1;row[0]=-1;rows.append(row)
        row=np.zeros(m);h0=breaks[1]-breaks[0];h1=breaks[-1]-breaks[-2]
        row[-1]+=1/h1;row[-2]-=1/h1;row[1]-=1/h0;row[0]+=1/h0;rows.append(row)
    else:
        for index in (0,m-1):
            row=np.zeros(m);row[index]=1;rows.append(row)
    rhs=np.zeros((len(rows),3))
    if not periodic and ends is not None:rhs[-2:]=ends
    return np.asarray(rows),rhs


def continuity_nullspace(breaks,periodic):
    """C0 shared poles, C1 internal joins; fixed endpoints or a periodic seam.

    Interior joins remain optimization variables. Nullspace steps preserve all
    equalities exactly, including derivative scaling for unequal span lengths.
    """
    return null_space(_constraints(breaks,periodic)[0])


def projected_seed(ordered,ends,breaks,edge_id,periodic):
    """Small robust point fit used only as an initializer for distance fitting."""
    from .surface_fitting import constrained_lstsq
    basis=bezier_basis(arclength_parameters(ordered),breaks)
    a,rhs=_constraints(breaks,periodic,ends if len(ends)==2 else None)
    weights=np.ones(len(ordered));poles=None
    for _ in range(4):
        w=np.sqrt(weights)[:,None]
        poles=constrained_lstsq(w*basis,w*ordered,a,rhs,regularization=1e-10)
        residual=np.linalg.norm(basis@poles-ordered,axis=1)
        scale=max(1e-6,1.4826*np.median(abs(residual-np.median(residual))))
        weights=np.minimum(1.,1.5*scale/np.maximum(residual,1e-12))
    return _record(edge_id,breaks,poles)


def refine_bezier_distance(record,points,distances,cfg,periodic=False):
    """Robust distance fitting with analytic control-point Jacobian; no provider."""
    breaks=record.data['breaks'];p0=_poles(record);z=continuity_nullspace(breaks,periodic)
    # Repair initial C1 joins while retaining the initialization's endpoints.
    # Projecting onto the affine constraints leaves a valid base for nullspace steps.
    a,rhs=_constraints(breaks,periodic,p0[[0,-1]])
    p0=p0+np.linalg.lstsq(a,rhs-a@p0,rcond=None)[0]
    regularizer=np.diff(np.eye(len(p0)),n=2,axis=0)
    strength=1e-4/np.sqrt(max(1,len(regularizer)))
    cache={};warm=None
    def compute(theta):
        nonlocal warm
        if 'theta' in cache and np.array_equal(cache['theta'],theta):return
        checkpoint('curve_distance_optimization')
        poles=p0+z@theta.reshape(-1,3);candidate=_record(record.id,breaks,poles)
        warm,rho=closest_parameters(candidate,points,warm)
        c=candidate.evaluate(warm);normal=(c-points)/np.maximum(rho[:,None],1e-10)
        basis=bezier_basis(warm,breaks)@z
        jac=(basis[:,:,None]*normal[:,None,:]).reshape(len(points),-1)
        smooth=strength*regularizer@poles
        smooth_jac=strength*np.kron(regularizer@z,np.eye(3))
        cache.update(theta=theta.copy(),residual=np.r_[rho-distances,smooth.ravel()],
                     jacobian=np.vstack((jac,smooth_jac)),candidate=candidate)
    def residual(theta):compute(theta);return cache['residual']
    def jacobian(theta):compute(theta);return cache['jacobian']
    result=least_squares(residual,np.zeros(3*z.shape[1]),jac=jacobian,loss='huber',
                         f_scale=max(cfg.field_error,1e-6),max_nfev=cfg.curve_distance_iterations,
                         x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8)
    compute(result.x)
    return cache['candidate'],dict(evaluations=result.nfev,cost=float(result.cost),converged=bool(result.success))


def _circle_candidate(points,ends,edge_id):
    if len(points)<6:return None
    if len(ends)==2 and np.linalg.norm(ends[1]-ends[0])>1e-8:
        result=fit_fixed_arc(points,ends[0],ends[1])
        return CurveRecord(edge_id,'circle',result[0],result[1]) if result else None
    data,error=fit_circle(points)
    if data['radius']<1e-8:return None
    if len(ends):
        anchor=ends[0];n=data['normal']
        data['center']-=n*np.dot(data['center']-anchor,n)
        data['radius']=np.linalg.norm(anchor-data['center'])
        if data['radius']<1e-8:return None
        data['x']=(anchor-data['center'])/data['radius'];data['y']=np.cross(n,data['x'])
    data['start']=0.;data['sweep']=2*np.pi
    return CurveRecord(edge_id,'circle',data,error)


def refine_circle_distance(record,points,distances,ends,cfg):
    """Low-dimensional geometry optimization; numerical derivatives of the
    candidate circle only, never finite differences of UDF/model queries.
    """
    d=record.data;n0=d['normal'];x0=d['x'];y0=d['y']
    if len(ends)==2 and np.linalg.norm(ends[1]-ends[0])>1e-8:
        a,b=ends;mid=(a+b)*.5;chord=(b-a)/np.linalg.norm(b-a)
        w=d['center']-mid
        if np.linalg.norm(w)<1e-8:w=np.cross(n0,chord)
        w/=np.linalg.norm(w);v=np.cross(chord,w)
        initial=np.array([np.dot(d['center']-mid,w),0.])
        def candidate(theta):
            direction=np.cos(theta[1])*w+np.sin(theta[1])*v
            center=mid+theta[0]*direction;radius=np.linalg.norm(a-center)
            x=(a-center)/radius;n=np.cross(chord,direction)
            if n@n0<0:n=-n
            y=np.cross(n,x);angle=np.arctan2((b-center)@y,(b-center)@x)
            sweep=np.mod(angle,2*np.pi) if d['sweep']>0 else -np.mod(-angle,2*np.pi)
            return CurveRecord(record.id,'circle',dict(center=center,radius=radius,x=x,y=y,
                               normal=n,start=0.,sweep=sweep),0.)
    else:
        anchored=bool(len(ends));origin=ends[0] if anchored else d['center']
        initial=(np.array([(d['center']-origin)@x0,(d['center']-origin)@y0,0.,0.]) if anchored
                 else np.r_[d['center'],0.,0.,np.log(d['radius'])])
        def candidate(theta):
            tilt=theta[2:4] if anchored else theta[3:5]
            n=n0+tilt[0]*x0+tilt[1]*y0;n/=np.linalg.norm(n)
            x=x0-n*(x0@n);x/=max(np.linalg.norm(x),1e-12);y=np.cross(n,x)
            center=origin+theta[0]*x+theta[1]*y if anchored else theta[:3]
            radius=np.linalg.norm(origin-center) if anchored else np.exp(np.clip(theta[5],-20,10))
            if anchored:x=(origin-center)/max(radius,1e-12);y=np.cross(n,x)
            return CurveRecord(record.id,'circle',dict(center=center,radius=max(radius,1e-12),
                               x=x,y=y,normal=n,start=0.,sweep=2*np.pi),0.)
    def residual(theta):
        checkpoint('curve_circle_optimization')
        _,rho=closest_parameters(candidate(theta),points)
        return rho-distances
    result=least_squares(residual,initial,loss='huber',f_scale=max(cfg.field_error,1e-6),
                         max_nfev=cfg.curve_distance_iterations,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    return candidate(result.x),dict(evaluations=result.nfev,cost=float(result.cost))


def validate_distance_curve(record,points,distances,ends,provider,cfg):
    """Original off-curve samples have nonzero distances: validate residuals,
    not the old point-to-curve zero-distance coverage criterion.
    """
    t,residual=distance_residuals(record,points,distances,cfg.query_batch_size)
    maximum=float(np.max(np.abs(residual)))
    require(maximum<=cfg.curve_tolerance,'curve_distance_failed','Curve disagrees with original distance observations',error=maximum)
    if len(ends):
        expected=ends if len(ends)==2 else np.repeat(ends,2,axis=0)
        require(np.linalg.norm(record.evaluate(np.array([0.,1.]))-expected,axis=1).max()<1e-8,
                'curve_endpoint_moved','Curve moved fixed endpoint')
    if record.kind=='bezier_spline':
        # Prevent an unsupported excursion between sparsely constrained spans.
        grid=np.unique(np.r_[np.linspace(0,1,257),record.data['breaks']])
    else:grid=np.linspace(0,1,257)
    length=np.linalg.norm(np.diff(record.evaluate(grid),axis=0),axis=1).sum()
    count=max(500,int(np.ceil(length/(cfg.curve_tolerance*.5)))+1)
    test=record.evaluate(np.linspace(0,1,count));v=evaluate_batched(provider,test,cfg.query_batch_size)
    require(v[:,1].max()<=2*cfg.curve_tolerance,'curve_field_failed','Fitted curve violates edge UDF',error=float(v[:,1].max()))
    require(np.abs(v[:,0]).max()<=cfg.feature_sdf_limit,'curve_surface_failed','Fitted curve violates SDF')
    # Every part of the candidate must have nearby original support; a small
    # global E value alone could otherwise accept an unrelated zero branch.
    nearest=cKDTree(points).query(test)[1]
    cover=np.linalg.norm(test-points[nearest],axis=1)-distances[nearest]
    require(cover.max()<=cfg.tau_edge,'curve_support_unobserved','Curve leaves the observed instance support',error=float(cover.max()))
    record.residual=maximum
    return dict(max_distance_residual=maximum,field_samples=count,observations=len(points))


def _initial_bezier(ordered,ends,edge_id,periodic):
    t=arclength_parameters(ordered)
    if not periodic:
        return projected_seed(ordered,ends,np.array([0.,1.]),edge_id,False)
    # A nondegenerate C1-periodic loop cannot be represented by one cubic span.
    breaks=np.linspace(0,1,5);h=np.median(np.diff(t)[np.diff(t)>0])
    positions=np.column_stack([np.interp(breaks,t,ordered[:,a]) for a in range(3)])
    positions[-1]=positions[0];nets=[]
    def point(u):return np.array([np.interp(u%1,t,ordered[:,a]) for a in range(3)])
    tangent=np.array([(point(u+h)-point(u-h))/(2*h) for u in breaks]);tangent[-1]=tangent[0]
    for j in range(4):
        width=breaks[j+1]-breaks[j]
        nets.append([positions[j],positions[j]+width*tangent[j]/3,
                     positions[j+1]-width*tangent[j+1]/3,positions[j+1]])
    return CurveRecord(edge_id,'bezier_spline',dict(breaks=breaks,controls=np.asarray(nets)),0.)


def recover_distance_curves(topology,bands,vertices,provider,cfg,progress=None):
    from .edge_geometry import project_support, circle_order, order_general
    edges=[];curves={};reports={};timings=dict(observations=0.,initialization=0.,optimization=0.,validation=0.)
    for item in topology.edges:
        checkpoint('curve_distance_recovery');started=time.monotonic()
        edge_id=item['id'];vids=list(item['vertices'])
        if progress is not None:progress(dict(active_edge=edge_id,completed=len(curves),timings=timings,edges=reports))
        require(len(item['faces'])==2,'edge_neighbors_unresolved','Curve requires two confirmed neighboring faces',edge=edge_id,faces=item['faces'])
        require(len(vids)<=2,'edge_vertices_unresolved','Edge touches more than two vertex candidates',edge=edge_id)
        ids=np.asarray(item.get('support_nodes',item['nodes']),int)
        ends=np.array([vertices[v].position for v in vids]).reshape(-1,3)
        inferred=cfg.infer_short_edges and len(vids)==2 and (item.get('status')=='inferred_short' or len(ids)<3)
        if inferred:
            # A topology-supported short line can lack off-curve observations.
            points=ends.copy();distances=np.zeros(2)
        else:
            require(len(ids)>=3,'edge_support_insufficient','Too few instance distance observations',edge=edge_id)
            points=np.asarray(bands.points[ids],float)
            distances=(np.asarray(bands.values[ids,1],float) if hasattr(bands,'values')
                       else evaluate_batched(provider,points,cfg.query_batch_size)[:,1])
        require(np.isfinite(points).all() and np.isfinite(distances).all() and (distances>=0).all(),
                'invalid_field_query','Invalid cached curve observations')
        timings['observations']+=time.monotonic()-started
        radius=max(cfg.tau_edge*3,cfg.surface_tolerance*4)
        report=dict(observations=len(points),initialization_points=0,
                    projection=dict(iterations=0,gradient_rows=0),attempts=[])
        record=None;validation=None
        def accept(candidate):
            nonlocal validation
            tick=time.monotonic()
            try:validation=validate_distance_curve(candidate,points,distances,ends,provider,cfg)
            except ReconstructionError as exc:
                if exc.code not in ('curve_distance_failed','curve_field_failed','curve_surface_failed','curve_support_unobserved','invalid_field_query'):
                    raise
                report['attempts'].append(dict(kind=candidate.kind,accepted=False,error=exc.code,details=exc.details))
                return False
            finally:timings['validation']+=time.monotonic()-tick
            report['attempts'].append(dict(kind=candidate.kind,accepted=True))
            return True
        if len(ends)==2 and np.linalg.norm(ends[1]-ends[0])>cfg.localization_tolerance:
            candidate=CurveRecord(edge_id,'line',dict(p0=ends[0],p1=ends[1]),0.)
            if accept(candidate):record=candidate
        if record is None:
            require(not inferred,'curve_field_failed','Endpoint-supported short line fails distance/field checks',edge=edge_id)
            tick=time.monotonic();init=representative_indices(points,cfg.curve_init_samples,cfg.seed+edge_id)
            report['initialization_points']=len(init)
            projected=project_support(points[init],provider,radius,cfg,max_iterations=8,stats=report['projection'])
            timings['initialization']+=time.monotonic()-tick
            fit=representative_indices(points,cfg.curve_fit_samples,cfg.seed+edge_id)
            tick=time.monotonic()
            circle=_circle_candidate(projected,ends,edge_id) if cfg.curve_candidates=='arcs' else None
            if circle is not None:
                try:
                    circle,stats=refine_circle_distance(circle,points[fit],distances[fit],ends,cfg)
                    report['circle_optimization']=stats
                except (ValueError,np.linalg.LinAlgError) as exc:
                    circle=None
                    report['attempts'].append(dict(kind='circle',accepted=False,error='circle_optimization_failed',message=str(exc)))
                timings['optimization']+=time.monotonic()-tick
                if circle is not None and accept(circle):record=circle
            else:timings['optimization']+=time.monotonic()-tick
            if record is None:
                tick=time.monotonic();ordered=None
                if len(projected)>=6:
                    data,error=fit_circle(projected)
                    if error<=cfg.curve_tolerance:ordered=circle_order(projected,data,ends,cfg)
                if ordered is None:ordered=order_general(projected,ends,cfg)
                if len(ends)==1:ends=np.repeat(ends,2,axis=0)
                candidate=_initial_bezier(ordered,ends,edge_id,not len(vids))
                require(len(candidate.data['controls'])<=cfg.max_segments,'curve_distance_fit_failed',
                        'Periodic initialization exceeds curve segment budget',edge=edge_id)
                timings['initialization']+=time.monotonic()-tick
                rounds=[]
                for _ in range(2*cfg.max_segments+8):
                    tick=time.monotonic()
                    candidate,stats=refine_bezier_distance(candidate,points[fit],distances[fit],cfg,periodic=not len(vids))
                    timings['optimization']+=time.monotonic()-tick
                    stats.update(segments=len(candidate.data['controls']),fit_samples=len(fit));rounds.append(stats)
                    if accept(candidate):record=candidate;break
                    parameters,residual=distance_residuals(candidate,points,distances,cfg.query_batch_size)
                    if len(fit)<min(len(points),cfg.curve_refine_samples):
                        # Include difficult validation locations before adding freedom.
                        worst=np.argsort(np.abs(residual))[-max(1,cfg.curve_fit_samples//2):]
                        updated=np.unique(np.r_[fit,worst])
                        if len(updated)>cfg.curve_refine_samples:updated=updated[-cfg.curve_refine_samples:]
                        if len(updated)>len(fit):fit=updated;continue
                    if len(candidate.data['controls'])>=cfg.max_segments:break
                    span=np.clip(np.searchsorted(candidate.data['breaks'],parameters,side='right')-1,0,len(candidate.data['controls'])-1)
                    scores=np.array([np.max(np.abs(residual[span==j]),initial=0) for j in range(len(candidate.data['controls']))])
                    candidate=split_bezier(candidate,int(np.argmax(scores)))
                    # The unchanged de Casteljau curve and a robust fit to the
                    # ordered initializer are alternative starts. Select using
                    # ORIGINAL distances, not point-fit residuals, to escape a
                    # poor low-degree local minimum without trusting projections.
                    tick=time.monotonic()
                    seed=projected_seed(ordered,ends,candidate.data['breaks'],edge_id,not len(vids))
                    _,old_res=distance_residuals(candidate,points[fit],distances[fit])
                    _,seed_res=distance_residuals(seed,points[fit],distances[fit])
                    if np.mean(abs(seed_res))<np.mean(abs(old_res)):candidate=seed
                    timings['initialization']+=time.monotonic()-tick
                report['spline_optimization']=rounds
        require(record is not None,'curve_distance_fit_failed','Cannot satisfy original distance and field checks',edge=edge_id,
                attempts=report['attempts'],optimization=report.get('spline_optimization'),timings=timings)
        sample_count=max(64,8*len(record.data.get('controls',[]))+1)
        samples=record.evaluate(np.linspace(0,1,sample_count))
        endpoint_uses=vids if len(vids)!=1 else [vids[0],vids[0]]
        edges.append(EdgeInstance(edge_id,tuple(item['faces']),samples,not len(vids),endpoint_uses,-1,radius))
        curves[edge_id]=record;report.update(validation=validation,kind=record.kind)
        reports[edge_id]=report
        if progress is not None:progress(dict(active_edge=None,completed=len(curves),timings=timings,edges=reports))
    report=dict(method='direct_distance',residual_kind='distance_observation',
                instance_distance_assumption='narrow_instance_support',timings=timings,edges=reports)
    return LogicalGraph(topology.face_count,vertices,edges,report={'distance_fitting':report}),curves,report
