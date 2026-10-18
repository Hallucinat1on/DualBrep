"""Analytic initialization and hard-equality tensor-spline fitting; edges stay frozen."""
from dataclasses import dataclass
import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .schemas import require,ReconstructionError
from .surface_mesh import normals


def constrained_lstsq(A,X,B,C,regularization=1e-8):
    """Rank-revealing null-space solve; rejects conflicting hard constraints."""
    A=np.asarray(A,float);B=np.asarray(B,float);X=np.asarray(X,float);C=np.asarray(C,float)
    n=A.shape[1]
    if not len(B):return np.linalg.lstsq(np.vstack((A,np.sqrt(regularization)*np.eye(n))),np.vstack((X,np.zeros((n,X.shape[1])))),rcond=None)[0]
    P=np.linalg.lstsq(B,C,rcond=1e-11)[0]
    require(np.max(np.abs(B@P-C))<1e-7,'boundary_constraints_conflict','Hard boundary equalities are inconsistent; increase spline freedom')
    N=linalg.null_space(B,rcond=1e-11)
    if N.shape[1]:
        lhs=np.vstack((A@N,np.sqrt(regularization)*N));rhs=np.vstack((X-A@P,-np.sqrt(regularization)*P))
        P+=N@np.linalg.lstsq(lhs,rhs,rcond=1e-11)[0]
    require(np.max(np.abs(B@P-C))<1e-7,'boundary_constraints_failed','Null-space solve failed boundary equalities')
    return P

@dataclass
class FacePatch:
    id: int
    kind: str
    data: dict
    normal: np.ndarray
    report: dict
    def uv(self,points,ordered=False):
        p=np.asarray(points,float);d=self.data
        if self.kind in ('plane','bspline'):
            if self.kind=='bspline' and 'projection_xyz' in d:
                ids=cKDTree(d['projection_xyz']).query(p)[1]
                uv=d['projection_uv'][ids].copy()
            else:uv=np.column_stack(((p-d['origin'])@d['x'],(p-d['origin'])@d['y']))
            if self.kind=='bspline':
                if 'projection_xyz' not in d:uv=(uv-d['uv_lo'])/d['uv_extent']
                uv=np.clip(uv,0,1)
                for _ in range(5):
                    base=self.evaluate(uv);h=1e-4
                    du=(self.evaluate(uv+[h,0])-base)/h;dv=(self.evaluate(uv+[0,h])-base)/h
                    J=np.stack((du,dv),axis=2);lhs=np.einsum('nki,nkj->nij',J,J)+np.eye(2)*1e-10
                    rhs=np.einsum('nki,nk->ni',J,p-base);delta=np.linalg.solve(lhs,rhs[...,None])[...,0]
                    uv=np.clip(uv+np.clip(delta,-.1,.1),0,1)
            return uv
        q=p-d['origin'];z=q@d['axis'];x=q@d['x'];y=q@d['y'];u=np.arctan2(y,x)
        if ordered:u=np.unwrap(u)
        if self.kind=='cylinder':return np.column_stack((u,z))
        if self.kind=='sphere':return np.column_stack((u,np.arctan2(z,np.sqrt(x*x+y*y))))
        if self.kind=='cone':return np.column_stack((u,z/np.cos(d['angle'])))
        v=np.arctan2(z,np.sqrt(x*x+y*y)-d['major'])
        return np.column_stack((u,np.unwrap(v) if ordered else v))
    def evaluate(self,uv):
        uv=np.asarray(uv,float);d=self.data;u,v=uv.T
        if self.kind=='plane':return d['origin']+u[:,None]*d['x']+v[:,None]*d['y']
        if self.kind=='bspline':return tensor_basis(uv,d['knots'],d['degree'])@d['controls'].reshape(-1,3)
        radial=np.cos(u)[:,None]*d['x']+np.sin(u)[:,None]*d['y']
        if self.kind=='cylinder':return d['origin']+d['radius']*radial+v[:,None]*d['axis']
        if self.kind=='sphere':return d['origin']+d['radius']*(np.cos(v)[:,None]*radial+np.sin(v)[:,None]*d['axis'])
        if self.kind=='cone':return d['origin']+(d['radius']+v*np.sin(d['angle']))[:,None]*radial+(v*np.cos(d['angle']))[:,None]*d['axis']
        return d['origin']+(d['major']+d['minor']*np.cos(v))[:,None]*radial+(d['minor']*np.sin(v))[:,None]*d['axis']

def frame(axis):
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
    x=np.eye(3)[np.argmin(np.abs(axis))];x-=np.dot(x,axis)*axis;x/=np.linalg.norm(x)
    return x,np.cross(axis,x)

def tensor_basis(uv,knots,degree=3):
    n=len(knots)-degree-1
    b=BSpline(knots,np.eye(n),degree,extrapolate=True)
    a=b(uv[:,0]);c=b(uv[:,1]);return np.einsum('ni,nj->nij',a,c).reshape(len(uv),n*n)

def sample_face_regions(mesh,labels,cfg):
    """DualBrep [F,N,6] / bbox contract, local NumPy sampling instead of Open3D."""
    rng=np.random.default_rng(cfg.seed);output={};nrm=normals(mesh)
    for face in sorted(np.unique(labels[labels>=0])):
        ids=np.flatnonzero(labels==face);tri=mesh.vertices[mesh.triangles[ids]]
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        chosen=rng.choice(len(ids),cfg.face_samples,p=area/area.sum());root=np.sqrt(rng.random(cfg.face_samples));v=rng.random(cfg.face_samples)
        b=np.column_stack((1-root,root*(1-v),root*v));pts=np.einsum('ni,nij->nj',b,tri[chosen])
        output[int(face)]=(pts,nrm[ids[chosen]])
    return output

def analytic_candidates(points,normal,boundaries):
    allpts=np.vstack((points,boundaries));mean=allpts.mean(axis=0);_,_,vt=np.linalg.svd(allpts-mean,full_matrices=False)
    axis=vt[-1];x,y=frame(axis)
    if len(boundaries)>=3:
        bm=boundaries.mean(axis=0);_,_,bvt=np.linalg.svd(boundaries-bm,full_matrices=False)
        if np.max(np.abs((boundaries-bm)@bvt[-1]))<1e-5:
            mean=bm;axis=bvt[-1];x,y=frame(axis)
    candidates=[FacePatch(-1,'plane',dict(origin=mean,x=x,y=y,axis=axis),normal.mean(axis=0),{})]
    # Sphere LS initialization followed by radial residual optimization.
    A=np.column_stack((2*allpts,np.ones(len(allpts))));fit=np.linalg.lstsq(A,np.sum(allpts**2,axis=1),rcond=None)[0]
    radius=np.sqrt(max(0,fit[3]+fit[:3]@fit[:3]))
    if np.isfinite(radius) and 1e-5<radius<100:
        sphere=least_squares(lambda z:np.linalg.norm(allpts-z[:3],axis=1)-z[3],np.r_[fit[:3],radius],max_nfev=40)
        x,y=frame(np.array([0.,0.,1.]));candidates.append(FacePatch(-1,'sphere',dict(origin=sphere.x[:3],radius=sphere.x[3],axis=np.array([0.,0.,1.]),x=x,y=y),normal.mean(axis=0),{}))
    _,_,vn=np.linalg.svd(normal,full_matrices=False)
    axes=[vn[-1],vt[0],vt[-1]]
    for ax in axes:
        x,y=frame(ax);uv=np.column_stack(((allpts-mean)@x,(allpts-mean)@y))
        fit=np.linalg.lstsq(np.column_stack((2*uv,np.ones(len(uv)))),np.sum(uv*uv,axis=1),rcond=None)[0]
        rad=np.sqrt(max(0,fit[2]+fit[0]**2+fit[1]**2))
        if rad<=1e-6:continue
        c=mean+x*fit[0]+y*fit[1]
        # Axis and center also optimized, not merely PCA for partial cylinders.
        initial=np.r_[c,ax,rad]
        def residual(z):
            a=z[3:6]/max(np.linalg.norm(z[3:6]),1e-10);q=allpts-z[:3];h=q@a
            return np.r_[np.linalg.norm(q-h[:,None]*a,axis=1)-z[6],(np.linalg.norm(z[3:6])-1)*.1]
        res=least_squares(residual,initial,max_nfev=60)
        a=res.x[3:6];a/=np.linalg.norm(a);x,y=frame(a)
        if res.x[6]>0:candidates.append(FacePatch(-1,'cylinder',dict(origin=res.x[:3],axis=a,x=x,y=y,radius=res.x[6]),normal.mean(axis=0),{}))
    return candidates

def _axis_point(points,normals,axis):
    """Point on the axis: every revolution-surface normal line is coplanar with it."""
    m=np.cross(normals,axis);rhs=np.einsum('ij,ij->i',m,points)
    ok=np.linalg.norm(m,axis=1)>1e-3
    if ok.sum()<3:return points.mean(axis=0)
    c=np.linalg.lstsq(m[ok],rhs[ok],rcond=None)[0]
    return c-((c-points.mean(axis=0))@axis)*axis


def _profile(points,origin,axis):
    q=points-origin;z=q@axis;rho=np.linalg.norm(q-z[:,None]*axis,axis=1)
    return rho,z


def revolution_candidates(points,normal,related,boundaries):
    """Cylinder / cone / torus by joint axis + meridian-profile least squares.

    Axis guesses come from boundary circles and from normal statistics; the
    profile model is then refined together with the axis.  Boundary samples are
    weighted higher because shared curves were localized on the e-UDF.
    """
    guesses=[]
    circles=[c for c in related if c.kind=='circle']
    for c in circles:guesses.append((np.asarray(c.data['center'],float),np.asarray(c.data['normal'],float)))
    if len(circles)>=2:
        d=circles[1].data['center']-circles[0].data['center']
        if np.linalg.norm(d)>1e-4:guesses.append((np.asarray(circles[0].data['center'],float),d/np.linalg.norm(d)))
    for centered in (True,False):
        n=normal-normal.mean(axis=0) if centered else normal
        a=np.linalg.svd(n,full_matrices=False)[2][-1]
        guesses.append((_axis_point(points,normal,a),a))
    data=np.vstack((points,boundaries)) if len(boundaries) else points
    weight=np.r_[np.ones(len(points)),np.full(len(boundaries),3.)]
    out=[]
    for kind in ('cylinder','cone','torus'):
        best=None
        for c0,a0 in guesses:
            a0=a0/np.linalg.norm(a0);x0,y0=frame(a0)
            rho,z=_profile(data,c0,a0)
            if kind=='cylinder':p0=[np.median(rho)]
            elif kind=='cone':
                k,r0=np.polyfit(z,rho,1)
                if abs(k)<1e-3:continue
                p0=[r0,k]
            else:
                A=np.column_stack((2*rho,2*z,np.ones(len(z))));sol=np.linalg.lstsq(A,rho*rho+z*z,rcond=None)[0]
                R,zc=sol[:2];r=np.sqrt(max(sol[2]+R*R+zc*zc,0))
                if not (r>1e-4 and R>r*.5):continue
                p0=[R,zc,r]
            def residual(s):
                a=a0+s[0]*x0+s[1]*y0;a/=np.linalg.norm(a);c=c0+s[2]*x0+s[3]*y0
                rho,z=_profile(data,c,a);p=s[4:]
                if kind=='cylinder':r=rho-p[0]
                elif kind=='cone':r=(rho-p[0]-p[1]*z)/np.sqrt(1+p[1]**2)
                else:r=np.hypot(rho-p[0],z-p[1])-p[2]
                return weight*r
            try:fit=least_squares(residual,np.r_[0,0,0,0,p0],max_nfev=200,x_scale='jac')
            except ValueError:continue
            if not np.all(np.isfinite(fit.x)):continue
            score=float(np.abs(fit.fun).max())
            if best is None or score<best[0]:best=(score,fit.x,a0,x0,y0,c0)
        if best is None:continue
        _,s,a0,x0,y0,c0=best
        a=a0+s[0]*x0+s[1]*y0;a/=np.linalg.norm(a);c=c0+s[2]*x0+s[3]*y0;x,y=frame(a);p=s[4:]
        if kind=='cylinder' and p[0]>0:
            out.append(FacePatch(-1,'cylinder',dict(origin=c,axis=a,x=x,y=y,radius=float(p[0])),normal.mean(axis=0),{}))
        elif kind=='cone':
            _,z=_profile(data,c,a);z0=float(np.median(z));radius=float(p[0]+p[1]*z0);angle=float(np.arctan(p[1]))
            if radius>0 and 1e-3<abs(angle)<np.pi/2-1e-3:
                out.append(FacePatch(-1,'cone',dict(origin=c+z0*a,axis=a,x=x,y=y,radius=radius,angle=angle),normal.mean(axis=0),{}))
        elif kind=='torus' and p[0]>0 and p[2]>0:
            out.append(FacePatch(-1,'torus',dict(origin=c+p[1]*a,axis=a,x=x,y=y,major=float(p[0]),minor=float(p[2])),normal.mean(axis=0),{}))
    return out


def fit_general_spline(face,points,boundaries,normal,cfg,initial=None):
    mean=points.mean(axis=0);_,_,vt=np.linalg.svd(points-mean,full_matrices=False);x,y=vt[:2];axis=np.cross(x,y)
    uv=np.column_stack(((points-mean)@x,(points-mean)@y));bu=np.column_stack(((boundaries-mean)@x,(boundaries-mean)@y))
    lo=np.minimum(uv.min(axis=0),bu.min(axis=0));hi=np.maximum(uv.max(axis=0),bu.max(axis=0));extent=np.maximum(hi-lo,1e-8)
    uv=(uv-lo)/extent;bu=(bu-lo)/extent
    projection=None
    if initial is not None:
        grid=np.asarray(initial,float);nu,nv=grid.shape[:2]
        gu,gv=np.meshgrid(np.linspace(0,1,nu),np.linspace(0,1,nv),indexing='ij')
        projection=(grid.reshape(-1,3),np.column_stack((gu.ravel(),gv.ravel())))
        tree=cKDTree(projection[0]);uv=projection[1][tree.query(points)[1]];bu=projection[1][tree.query(boundaries)[1]]
    last=None
    for n in range(cfg.spline_controls,cfg.max_spline_controls+1,2):
        knots=np.r_[np.zeros(4),np.linspace(0,1,n-2)[1:-1],np.ones(4)]
        data=dict(origin=mean,x=x,y=y,axis=axis,uv_lo=lo,uv_extent=extent,knots=knots,degree=3)
        if projection is not None:data.update(projection_xyz=projection[0],projection_uv=projection[1])
        try:
            for _ in range(cfg.uv_iterations):
                A=tensor_basis(uv,knots);B=tensor_basis(bu,knots)
                P=constrained_lstsq(A,points,B,boundaries,regularization=1e-8)
                data['controls']=P.reshape(n,n,3);patch=FacePatch(face,'bspline',data,normal.mean(axis=0),{})
                uv=patch.uv(points);bu=patch.uv(boundaries)
            # Re-solve after UV changes; never leave the equality constraints stale.
            data['controls']=constrained_lstsq(tensor_basis(uv,knots),points,tensor_basis(bu,knots),boundaries).reshape(n,n,3)
            return patch
        except ReconstructionError as exc:last=exc
    raise ReconstructionError('surface_boundary_incompatible','Spline budget cannot satisfy fixed boundary',{'face':face,'cause':str(last)})

def initialize_face_surfaces(mesh,face_labels,graph,curves,provider,cfg,face_model=None):
    regions=sample_face_regions(mesh,face_labels,cfg);patches={}
    model_grids=face_model.predict(regions) if face_model is not None else {}
    for face in range(graph.face_count):
        require(face in regions,'missing_face_region','No sampled region for face',face=face)
        p,n=regions[face]
        # Core triangles provide face-interior samples; shared curves supply boundaries.
        # Initialize from trustworthy interior samples, with frozen boundaries
        # supplying the full trim extent; never fit leaked points across a sharp edge.
        fv=provider.evaluate(p).values
        safe=fv[:,1]>cfg.tau_edge*.5 if provider.presence()[0]=='present' else np.ones(len(p),bool)
        if np.sum(safe)>=20:p,n=p[safe],n[safe]
        related=[curves[e.id] for e in graph.edges if face in e.faces]
        boundary=np.vstack([c.evaluate(np.linspace(0,1,65)) for c in related]) if related else np.empty((0,3))
        candidates=analytic_candidates(p,n,boundary)
        if len(related)==2 and all(c.kind=='circle' for c in related):
            c0,c1=related;axis=c1.data['center']-c0.data['center']
            if np.linalg.norm(axis)>1e-6 and abs(c0.data['radius']-c1.data['radius'])<cfg.boundary_tolerance:
                axis/=np.linalg.norm(axis);x,y=frame(axis)
                candidates.insert(0,FacePatch(face,'cylinder',dict(origin=c0.data['center'],axis=axis,x=x,y=y,
                    radius=(c0.data['radius']+c1.data['radius'])/2),n[0],{}))
        accepted=[]
        def screen(options):
            for patch in options:
                err=np.linalg.norm(patch.evaluate(patch.uv(p))-p,axis=1)
                be=np.linalg.norm(patch.evaluate(patch.uv(boundary))-boundary,axis=1) if len(boundary) else np.zeros(1)
                if err.max()<=cfg.surface_tolerance*2 and be.max()<=cfg.boundary_tolerance:
                    accepted.append((float(err.mean()+be.max()),patch,float(be.max())))
        screen(candidates)
        if cfg.surface_candidates=='revolution':
            # A plane wins only if the frozen boundary really lies in it; otherwise a
            # shallow cylinder (small sagitta) would be flattened onto a plane.
            accepted=[a for a in accepted if a[1].kind!='plane' or a[2]<=cfg.boundary_tolerance*.25] or accepted
        if not accepted and cfg.surface_candidates=='revolution':
            screen(revolution_candidates(p,n,related,boundary))
            # Simplest accepted primitive first; a torus can mimic a cone within tolerance.
            rank={'cylinder':0,'cone':1,'torus':2}
            accepted.sort(key=lambda item:(rank.get(item[1].kind,3),item[0]))
            if accepted:accepted=accepted[:1]
        if accepted:patch=next((v for _,v,_ in accepted if v.kind=='plane'),min(accepted,key=lambda z:z[0])[1]);patch.id=face
        else:
            require(len(boundary)>0,'surface_initialization_failed','Unbounded nonanalytic face needs explicit UV/model initialization')
            patch=fit_general_spline(face,p,boundary,n,cfg,model_grids.get(face))
        patch.normal=n[0].copy();patch.data['orientation_point']=p[0].copy()
        maximum=0.
        for curve in related:
            # Adaptive evaluation on all progressively finer master-parameter grids.
            for size in (129,257,513):
                q=curve.evaluate(np.linspace(0,1,size));uv=patch.uv(q,ordered=True)
                e=np.linalg.norm(patch.evaluate(uv)-q,axis=1);maximum=max(maximum,float(e.max()))
            require(maximum<=cfg.boundary_tolerance,'surface_boundary_incompatible','Fixed curve does not lie on support surface',face=face,error=maximum)
        uv=patch.uv(p);q=patch.evaluate(uv);errors=np.linalg.norm(q-p,axis=1)
        sdf=np.abs(provider.sdf(q));require(errors.max()<=cfg.surface_tolerance*2 and sdf.max()<=cfg.surface_tolerance*2,
            'surface_fit_failed','Interior fit exceeds error budget',face=face)
        # Finite difference Jacobian checks only inside sampled valid trim region.
        h=1e-5;du=(patch.evaluate(uv+[h,0])-q)/h;dv=(patch.evaluate(uv+[0,h])-q)/h
        jac=np.linalg.norm(np.cross(du,dv),axis=1)
        require(np.all(jac>1e-10),'surface_fold','Degenerate local parameterization',face=face)
        patch.report={'boundary_max_error':maximum,'interior_max_error':float(errors.max()),'sdf_max_error':float(sdf.max()),
            'jacobian_min':float(jac.min()),'continuous_boundary_guarantee':False,'global_self_intersection_certified':False,
            'boundary_check_samples_per_edge':513,'periodic_u':patch.kind in ('cylinder','sphere','cone','torus')}
        patches[face]=patch
    return patches

# Exact compatible four-sided tensor Bezier boundary locking, exposed for model adapters.
def fit_four_sided_bezier(uv,points,bottom,right,top,left):
    from scipy.special import comb
    bottom,right,top,left=map(lambda x:np.asarray(x,float),(bottom,right,top,left))
    nu,nv=len(bottom),len(left)
    require(len(top)==nu and len(right)==nv,'boundary_basis_incompatible','Opposite side degrees must agree')
    net=np.zeros((nu,nv,3));net[:,0]=bottom;net[:,-1]=top;net[0]=left;net[-1]=right
    require(np.allclose(bottom[0],left[0]) and np.allclose(bottom[-1],right[0]) and np.allclose(top[0],left[-1]) and np.allclose(top[-1],right[-1]),
            'boundary_corner_conflict','Boundary corners disagree')
    def basis(t,n):return np.column_stack([comb(n-1,i)*t**i*(1-t)**(n-1-i) for i in range(n)])
    A=np.einsum('ni,nj->nij',basis(uv[:,0],nu),basis(uv[:,1],nv)).reshape(len(uv),nu*nv)
    fixed=np.zeros((nu,nv),bool);fixed[[0,-1],:]=True;fixed[:,[0,-1]]=True
    rhs=points-A[:,fixed.ravel()]@net[fixed]
    net[~fixed]=np.linalg.lstsq(A[:,~fixed.ravel()],rhs,rcond=None)[0]
    return net
