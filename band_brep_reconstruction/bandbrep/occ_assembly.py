"""Create shared topology once, attach pcurves, repair only missing periodic seams."""
import numpy as np
import copy
from OCC.Core.gp import gp_Pnt,gp_Dir,gp_Ax2,gp_Ax3,gp_Pnt2d,gp_Ax2d,gp_Dir2d
from OCC.Core.Geom import (Geom_Line,Geom_Circle,Geom_BSplineCurve,Geom_Plane,
                          Geom_CylindricalSurface,Geom_SphericalSurface,Geom_ConicalSurface,Geom_ToroidalSurface,Geom_BSplineSurface)
from OCC.Core.Geom2d import Geom2d_BSplineCurve,Geom2d_Circle
from OCC.Core.TColgp import TColgp_Array1OfPnt,TColgp_Array1OfPnt2d,TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger
from OCC.Core.BRep import BRep_Builder,BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex,BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Face,TopoDS_Wire,TopoDS_Shell,TopoDS_Solid,topods
from OCC.Core.BRepLib import breplib
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepTools import breptools
from OCC.Core.ShapeFix import ShapeFix_Face,ShapeFix_Solid
from OCC.Core.TopAbs import TopAbs_EDGE,TopAbs_FACE,TopAbs_VERTEX,TopAbs_SHELL
from OCC.Core.TopExp import TopExp_Explorer,topexp
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Writer,STEPControl_Reader,STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from .schemas import require


def array1(cls,values):
    a=cls(1,len(values))
    for i,v in enumerate(values):a.SetValue(i+1,v)
    return a

def occ_curve(record):
    d=record.data
    if record.kind=='line':
        # A degree-1 spline gives the shared master interval [0,1].
        poles=array1(TColgp_Array1OfPnt,[gp_Pnt(*d['p0']),gp_Pnt(*d['p1'])])
        return Geom_BSplineCurve(poles,array1(TColStd_Array1OfReal,[0.,1.]),array1(TColStd_Array1OfInteger,[2,2]),1),0.,1.
    if record.kind=='circle':
        sign=np.sign(d['sweep']);x=np.cos(d['start'])*d['x']+np.sin(d['start'])*d['y']
        axis=d['normal']*sign
        return Geom_Circle(gp_Ax2(gp_Pnt(*d['center']),gp_Dir(*axis),gp_Dir(*x)),d['radius']),0.,abs(d['sweep'])
    controls=d['controls'];poles=np.vstack((controls[0],*[p[1:] for p in controls[1:]]))
    return Geom_BSplineCurve(array1(TColgp_Array1OfPnt,[gp_Pnt(*p) for p in poles]),
        array1(TColStd_Array1OfReal,d['breaks'].tolist()),array1(TColStd_Array1OfInteger,[4]+[3]*(len(controls)-1)+[4]),3),0.,1.

def occ_surface(patch):
    d=patch.data
    if patch.kind=='plane':
        return Geom_Plane(gp_Ax3(gp_Pnt(*d['origin']),gp_Dir(*np.cross(d['x'],d['y'])),gp_Dir(*d['x'])))
    if patch.kind=='bspline':
        net=d['controls'];poles=TColgp_Array2OfPnt(1,net.shape[0],1,net.shape[1])
        for i in range(net.shape[0]):
            for j in range(net.shape[1]):poles.SetValue(i+1,j+1,gp_Pnt(*net[i,j]))
        knots,mults=np.unique(d['knots'],return_counts=True)
        return Geom_BSplineSurface(poles,array1(TColStd_Array1OfReal,knots.tolist()),array1(TColStd_Array1OfReal,knots.tolist()),
            array1(TColStd_Array1OfInteger,mults.tolist()),array1(TColStd_Array1OfInteger,mults.tolist()),d['degree'],d['degree'])
    axis=gp_Ax3(gp_Pnt(*d['origin']),gp_Dir(*d['axis']),gp_Dir(*d['x']))
    if patch.kind=='cylinder':return Geom_CylindricalSurface(axis,d['radius'])
    if patch.kind=='sphere':return Geom_SphericalSurface(axis,d['radius'])
    if patch.kind=='cone':return Geom_ConicalSurface(axis,d['angle'],d['radius'])
    return Geom_ToroidalSurface(axis,d['major'],d['minor'])

PERIODIC_KINDS=('cylinder','cone','sphere','torus')


def periodic_dims(kind):
    return (0,1) if kind=='torus' else (0,) if kind in PERIODIC_KINDS else ()


def shift_periods(uv,starts,kind):
    """Move an ordered pcurve by whole periods into [start, start+2*pi) per periodic dim.

    `starts` places each parameter cut inside a boundary gap, so a continuous
    pcurve that never crosses the cut lands entirely inside one period.
    """
    for d in periodic_dims(kind):
        uv[:,d]-=2*np.pi*np.floor((uv[:,d].mean()-starts[d])/(2*np.pi))
    return uv


def make_pcurve(patch,record,first,last,tolerance,adjust=None):
    coplanar=False
    if patch.kind=='plane' and record.kind=='circle':
        d=record.data;axis=np.cross(patch.data['x'],patch.data['y'])
        full=abs(abs(d['sweep'])-2*np.pi)<1e-9
        coplanar=full or (abs(abs(np.dot(d['normal'],axis))-1)<1e-9 and abs(np.dot(d['center']-patch.data['origin'],axis))<tolerance*.01)
    if coplanar:
        d=record.data;c=patch.uv(d['center'][None])[0]
        start=np.cos(d['start'])*d['x']+np.sin(d['start'])*d['y']
        tangent=np.sign(d['sweep'])*(-np.sin(d['start'])*d['x']+np.cos(d['start'])*d['y'])
        x=np.array([start@patch.data['x'],start@patch.data['y']]);y=np.array([tangent@patch.data['x'],tangent@patch.data['y']])
        sense=float(x[0]*y[1]-x[1]*y[0])>0
        pc=Geom2d_Circle(gp_Ax2d(gp_Pnt2d(*c),gp_Dir2d(*x)),d['radius'],sense)
        uv=patch.uv(record.evaluate(np.linspace(0,1,65)),ordered=True)
        return pc,uv,0.
    for count in (65,129,257,513,1025):
        t=np.linspace(0,1,count);points=record.evaluate(t);uv=patch.uv(points,ordered=True)
        if adjust is not None:uv=adjust(uv)
        # Use piecewise linear UV interpolation, then independently check midpoints.
        middle=(uv[:-1]+uv[1:])*.5;target=record.evaluate((t[:-1]+t[1:])*.5)
        error=np.linalg.norm(patch.evaluate(middle)-target,axis=1).max()
        if error<=tolerance*.5:break
    require(error<=tolerance,'pcurve_fit_failed','pcurve interpolation misses shared curve',error=float(error),face=patch.id)
    curve=Geom2d_BSplineCurve(array1(TColgp_Array1OfPnt2d,[gp_Pnt2d(*p) for p in uv]),
        array1(TColStd_Array1OfReal,np.linspace(first,last,count).tolist()),
        array1(TColStd_Array1OfInteger,[2]+[1]*(count-2)+[2]),1)
    return curve,uv,float(error)

def plan_periodic_band(patch,cycles,curves,graph,offsets,anchor):
    """UV layout for a cylinder/cone face bounded by two loops that each wrap once.

    Both loops are shifted so their u-range is [u_a, u_a + 2*pi], where u_a is the
    meridian through the anchor vertex.  Non-wrapping loops must lie strictly
    inside that strip (they become holes).  Returns None if the layout fails.
    """
    offsets={k:np.array(v,float) for k,v in offsets.items()}
    u_a=float(patch.uv(anchor[None])[0,0]);wraps=[];holes=[]
    for cycle in cycles:
        traced=[]
        for use in cycle:
            uv=patch.uv(curves[use.edge].evaluate(np.linspace(0,1,129)),ordered=True)+offsets[use.edge]
            traced.append(uv if use.forward else uv[::-1])
        uv=np.vstack(traced);turn=uv[-1,0]-uv[0,0]
        (wraps if abs(abs(turn)-2*np.pi)<.1 else holes).append((cycle,uv,turn))
    if len(wraps)!=2 or any(abs(t)>.1 for _,_,t in holes):return None
    layout=[]
    for cycle,uv,turn in wraps:
        shift=2*np.pi*np.round((u_a-uv[:,0].min())/(2*np.pi))
        if abs(uv[:,0].min()+shift-u_a)>1e-3:return None
        for use in cycle:offsets[use.edge][0]+=shift
        start=uv[0]+[shift,0];layout.append(dict(cycle=cycle,turn=turn,v=float(start[1]),u_start=float(start[0])))
    for cycle,uv,_ in holes:
        shift=2*np.pi*np.floor((uv[:,0].min()-u_a)/(2*np.pi))
        if uv[:,0].max()-shift>=u_a+2*np.pi:return None
        for use in cycle:offsets[use.edge][0]-=shift
    layout.sort(key=lambda item:item['v'])
    return dict(offsets=offsets,u_a=u_a,layout=layout,holes=[c for c,_,_ in holes])


def make_periodic_band(patch,surface,plan,graph,edges,occ_vertices,cfg,builder,face):
    """Seam = straight generator through the two loops' anchor vertices."""
    def anchor_vertex(cycle):
        use=cycle[0];e=graph.edges[use.edge]
        if e.closed:return topexp.FirstVertex(edges[use.edge])
        return occ_vertices[e.vertices[0 if use.forward else 1]]
    lower,upper=plan['layout'];u_a=plan['u_a']
    a=anchor_vertex(lower['cycle']);b=anchor_vertex(upper['cycle'])
    pa,pb=BRep_Tool.Pnt(a),BRep_Tool.Pnt(b)
    v0,v1=lower['v'],upper['v']
    require(v1>v0+1e-9,'periodic_band_degenerate','Band loops share one parameter level',face=patch.id)
    geom=Geom_BSplineCurve(array1(TColgp_Array1OfPnt,[pa,pb]),array1(TColStd_Array1OfReal,[v0,v1]),array1(TColStd_Array1OfInteger,[2,2]),1)
    make=BRepBuilderAPI_MakeEdge(geom,a,b,v0,v1)
    require(make.IsDone(),'seam_creation_failed','Cannot create band seam',face=patch.id,kind=patch.kind,error=int(make.Error()))
    seam=make.Edge()
    def line(u):
        return Geom2d_BSplineCurve(array1(TColgp_Array1OfPnt2d,[gp_Pnt2d(u,v0),gp_Pnt2d(u,v1)]),
            array1(TColStd_Array1OfReal,[v0,v1]),array1(TColStd_Array1OfInteger,[2,2]),1)
    builder.UpdateEdge(seam,line(u_a+2*np.pi),line(u_a),face,cfg.boundary_tolerance);builder.Range(seam,face,v0,v1)
    builder.UpdateVertex(a,cfg.boundary_tolerance);builder.UpdateVertex(b,cfg.boundary_tolerance)
    def uses(item,increasing):
        cycle=item['cycle'];forward=item['turn']>0
        if forward==increasing:seq=[(u.edge,u.forward) for u in cycle]
        else:seq=[(u.edge,not u.forward) for u in reversed(cycle)]
        return [edges[e] if f else topods.Edge(edges[e].Reversed()) for e,f in seq]
    wire=TopoDS_Wire();builder.MakeWire(wire)
    # Counter-clockwise in UV: lower loop +u, seam up at u_a+2pi, upper loop -u, seam down at u_a.
    for edge in uses(lower,True)+[seam]+uses(upper,False)+[topods.Edge(seam.Reversed())]:builder.Add(wire,edge)
    wire.Closed(True);builder.Add(face,wire)
    for cycle in plan['holes']:
        hole=TopoDS_Wire();builder.MakeWire(hole)
        for use in cycle:builder.Add(hole,edges[use.edge] if use.forward else topods.Edge(edges[use.edge].Reversed()))
        hole.Closed(True);builder.Add(face,hole)
    uv=patch.uv(np.asarray(patch.data['orientation_point'])[None]);h=1e-5;q=patch.evaluate(uv)
    n=np.cross((patch.evaluate(uv+[h,0])-q)[0],(patch.evaluate(uv+[0,h])-q)[0])
    if n@patch.normal<0:face=topods.Face(face.Reversed())
    return face


def make_periodic_annulus(patch,surface,incident,curves,edges,ranges,cfg,builder):
    face=TopoDS_Face();builder.MakeFace(face,surface,cfg.cad_tolerance*.1)
    rings=[]
    for e in incident:
        record=curves[e.id];t=np.linspace(0,1,129);uv=patch.uv(record.evaluate(t),ordered=True)
        # Align a whole turn to the common [0,2*pi] domain.
        uv[:,0]-=round(float(uv[:,0].min())/(2*np.pi))*2*np.pi
        uv[:,0]-=uv[:,0].min()
        v=float(np.mean(uv[:,1]));require(np.ptp(uv[:,1])<cfg.boundary_tolerance,'periodic_domain_incompatible','Boundary is not a latitude / cylinder ring')
        rings.append([v,e.id,bool(uv[-1,0]>uv[0,0]),float(uv[0,0]),float(uv[-1,0])])
    rings.sort()
    if patch.kind=='torus':
        # Two meridian levels split the tube into two arcs; keep the one with the interior.
        inner=float(patch.uv(np.asarray(patch.data['orientation_point'])[None])[0,1])
        if rings[0][0]+np.mod(inner-rings[0][0],2*np.pi)>rings[1][0]:
            rings[0][0]+=2*np.pi;rings=[rings[1],rings[0]]
    for v,eid,_,ua,ub in rings:
        first,last=ranges[eid]
        pc=Geom2d_BSplineCurve(array1(TColgp_Array1OfPnt2d,[gp_Pnt2d(ua,v),gp_Pnt2d(ub,v)]),
             array1(TColStd_Array1OfReal,[first,last]),array1(TColStd_Array1OfInteger,[2,2]),1)
        builder.UpdateEdge(edges[eid],pc,face,cfg.boundary_tolerance);builder.Range(edges[eid],face,first,last)
    (v0,e0,positive0,_,_),(v1,e1,positive1,_,_)=rings
    a=topexp.FirstVertex(edges[e0]);b=topexp.FirstVertex(edges[e1])
    builder.UpdateVertex(a,cfg.boundary_tolerance);builder.UpdateVertex(b,cfg.boundary_tolerance)
    d=patch.data
    if patch.kind in ('cylinder','cone'):
        # Generator S(0,v) is linear in v for both cylinders and cones.
        poles=array1(TColgp_Array1OfPnt,[surface.Value(0,v0),surface.Value(0,v1)])
        geom=Geom_BSplineCurve(poles,array1(TColStd_Array1OfReal,[v0,v1]),array1(TColStd_Array1OfInteger,[2,2]),1)
    elif patch.kind=='torus':
        # Meridian S(0,v)=O+major*x+minor*(cos(v)*x+sin(v)*axis).
        center=np.asarray(d['origin'])+d['major']*np.asarray(d['x'])
        geom=Geom_Circle(gp_Ax2(gp_Pnt(*center),gp_Dir(*np.cross(d['x'],d['axis'])),gp_Dir(*d['x'])),d['minor'])
    else:
        # Meridian S(0,v)=center+R*(cos(v)*x+sin(v)*axis).
        geom=Geom_Circle(gp_Ax2(gp_Pnt(*d['origin']),gp_Dir(*np.cross(d['x'],d['axis'])),gp_Dir(*d['x'])),d['radius'])
    seam=BRepBuilderAPI_MakeEdge(geom,a,b,v0,v1)
    require(seam.IsDone(),'seam_creation_failed','Cannot create shared-anchor seam',face=patch.id,kind=patch.kind,
            levels=[float(v0),float(v1)],error=int(seam.Error()))
    seam=seam.Edge()
    def line(u):
        return Geom2d_BSplineCurve(array1(TColgp_Array1OfPnt2d,[gp_Pnt2d(u,v0),gp_Pnt2d(u,v1)]),
            array1(TColStd_Array1OfReal,[v0,v1]),array1(TColStd_Array1OfInteger,[2,2]),1)
    builder.UpdateEdge(seam,line(2*np.pi),line(0.),face,cfg.boundary_tolerance)
    builder.Range(seam,face,v0,v1)
    wire=TopoDS_Wire();builder.MakeWire(wire)
    bottom=edges[e0] if positive0 else edges[e0].Reversed()
    top=edges[e1].Reversed() if positive1 else edges[e1]
    for use in (bottom,seam,top,seam.Reversed()):builder.Add(wire,use)
    wire.Closed(True);builder.Add(face,wire)
    uv=patch.uv(np.asarray(patch.data.get('orientation_point',patch.evaluate(np.array([[.2,(v0+v1)/2]]))[0]))[None]);h=1e-5;q=patch.evaluate(uv)
    n=np.cross((patch.evaluate(uv+[h,0])-q)[0],(patch.evaluate(uv+[0,h])-q)[0])
    if n@patch.normal<0:face=topods.Face(face.Reversed())
    return face

def make_spherical_cap(patch,surface,instance,curves,edges,ranges,cfg,builder):
    from OCC.Core.TopoDS import TopoDS_Edge
    from OCC.Core.TopAbs import TopAbs_FORWARD,TopAbs_REVERSED
    face=TopoDS_Face();builder.MakeFace(face,surface,cfg.cad_tolerance*.1)
    record=curves[instance.id];uv=patch.uv(record.evaluate(np.linspace(0,1,129)),ordered=True)
    v=float(np.mean(uv[:,1]));north=patch.uv(patch.data['orientation_point'][None])[0,1]>v
    low,high=(v,np.pi/2) if north else (-np.pi/2,v)
    ring=edges[instance.id];first,last=ranges[instance.id];positive=uv[-1,0]>uv[0,0]
    pc=Geom2d_BSplineCurve(array1(TColgp_Array1OfPnt2d,[gp_Pnt2d(0 if positive else 2*np.pi,v),gp_Pnt2d(2*np.pi if positive else 0,v)]),
        array1(TColStd_Array1OfReal,[first,last]),array1(TColStd_Array1OfInteger,[2,2]),1)
    builder.UpdateEdge(ring,pc,face,cfg.boundary_tolerance);builder.Range(ring,face,first,last)
    anchor=topexp.FirstVertex(ring);builder.UpdateVertex(anchor,cfg.boundary_tolerance)
    pole=BRepBuilderAPI_MakeVertex(surface.Value(0,high if north else low)).Vertex()
    d=patch.data;geom=Geom_Circle(gp_Ax2(gp_Pnt(*d['origin']),gp_Dir(*np.cross(d['x'],d['axis'])),gp_Dir(*d['x'])),d['radius'])
    make=BRepBuilderAPI_MakeEdge(geom,anchor if north else pole,pole if north else anchor,low,high)
    require(make.IsDone(),'seam_creation_failed','Cannot construct spherical-cap seam');seam=make.Edge()
    def line(u):
        return Geom2d_BSplineCurve(array1(TColgp_Array1OfPnt2d,[gp_Pnt2d(u,low),gp_Pnt2d(u,high)]),
            array1(TColStd_Array1OfReal,[low,high]),array1(TColStd_Array1OfInteger,[2,2]),1)
    builder.UpdateEdge(seam,line(2*np.pi),line(0),face,cfg.boundary_tolerance);builder.Range(seam,face,low,high)
    degen=TopoDS_Edge();builder.MakeEdge(degen);builder.Degenerated(degen,True)
    builder.Add(degen,pole.Oriented(TopAbs_FORWARD));builder.Add(degen,pole.Oriented(TopAbs_REVERSED))
    pv=high if north else low
    pc=Geom2d_BSplineCurve(array1(TColgp_Array1OfPnt2d,[gp_Pnt2d(0,pv),gp_Pnt2d(2*np.pi,pv)]),
            array1(TColStd_Array1OfReal,[0.,2*np.pi]),array1(TColStd_Array1OfInteger,[2,2]),1)
    builder.UpdateEdge(degen,pc,face,cfg.boundary_tolerance);builder.Range(degen,face,0.,2*np.pi)
    plus=ring if positive else ring.Reversed();minus=plus.Reversed()
    wire=TopoDS_Wire();builder.MakeWire(wire)
    uses=(plus,seam,degen.Reversed(),seam.Reversed()) if north else (degen,seam,minus,seam.Reversed())
    for use in uses:builder.Add(wire,use)
    wire.Closed(True);builder.Add(face,wire)
    q=patch.data['orientation_point'][None];uv=patch.uv(q);base=patch.evaluate(uv);h=1e-5
    n=np.cross((patch.evaluate(uv+[h,0])-base)[0],(patch.evaluate(uv+[0,h])-base)[0])
    if n@patch.normal<0:face=topods.Face(face.Reversed())
    return face

def unique(shape,kind):
    index=TopTools_IndexedMapOfShape();topexp.MapShapes(shape,kind,index)
    return [index.FindKey(i) for i in range(1,index.Size()+1)]

def check_statuses(shape):
    import OCC.Core.BRepCheck as module
    names={int(v):k for k,v in vars(module).items() if k.startswith('BRepCheck_') and isinstance(v,int)}
    analyzer=BRepCheck_Analyzer(shape);report=[]
    for kind in (TopAbs_FACE,TopAbs_EDGE,TopAbs_VERTEX):
        for i,sub in enumerate(unique(shape,kind)):
            result=analyzer.Result(sub);iterator=module.BRepCheck_ListIteratorOfListOfStatus(result.Status());statuses=[]
            while iterator.More():statuses.append(names.get(int(iterator.Value()),str(iterator.Value())));iterator.Next()
            result.InitContextIterator()
            while result.MoreShapeInContext():
                iterator=module.BRepCheck_ListIteratorOfListOfStatus(result.StatusOnShape())
                while iterator.More():statuses.append(names.get(int(iterator.Value()),str(iterator.Value())));iterator.Next()
                result.NextShapeInContext()
            if any(s!='BRepCheck_NoError' for s in statuses):report.append({'kind':int(kind),'index':i,'statuses':statuses})
    return report

def tolerance_max(shape):
    v=[BRep_Tool.Tolerance(topods.Vertex(x)) for x in unique(shape,TopAbs_VERTEX)]
    v += [BRep_Tool.Tolerance(topods.Edge(x)) for x in unique(shape,TopAbs_EDGE)]
    v += [BRep_Tool.Tolerance(topods.Face(x)) for x in unique(shape,TopAbs_FACE)]
    return max(v,default=0.)

def assemble_brep(graph,curves,patches,cfg,out):
    builder=BRep_Builder();occ_vertices={v.id:BRepBuilderAPI_MakeVertex(gp_Pnt(*v.position)).Vertex() for v in graph.vertices}
    edges={};ranges={};mapping={'logical_edges':{},'postprocess_edges':[],'auxiliary_vertices':[],'faces':{},'repairs':[]}
    # Plan one common meridian across periodic faces sharing coaxial rings.
    # This changes parameter frames / loop origins, not support or curve geometry.
    curves=copy.deepcopy(curves);patches=copy.deepcopy(patches);planned=set()
    ring_kinds=PERIODIC_KINDS if cfg.periodic_frame_planning else ('cylinder','sphere')
    def ring_face(fid,patch):
        incident=[e for e in graph.edges if fid in e.faces]
        ok=patch.kind in ring_kinds and (len(incident)==2 or (patch.kind=='sphere' and len(incident)==1))
        return ok and all(e.closed and curves[e.id].kind=='circle' for e in incident)
    def wrap_anchor(fid):
        # A loop of open edges is cut in UV at its first use's start vertex.
        for cycle in graph.cycles.get(fid,[]):
            for use in cycle:
                e=graph.edges[use.edge]
                if not e.closed:return np.asarray(graph.vertices[e.vertices[0 if use.forward else 1]].position,float)
        return None
    ring_anchor={}
    if cfg.periodic_frame_planning:
        for fid,patch in patches.items():
            if patch.kind not in PERIODIC_KINDS or ring_face(fid,patch):continue
            incident=[e for e in graph.edges if fid in e.faces]
            rings=[e for e in incident if e.closed and curves[e.id].kind=='circle']
            if not rings or len(rings)==len(incident):continue
            anchor=wrap_anchor(fid)
            if anchor is None:continue
            for e in rings:ring_anchor.setdefault(e.id,anchor)
    if cfg.periodic_frame_planning:
        # One reference direction per family of coaxial ring faces, so every ring
        # start (and every seam) lies on one common meridian half-plane.  A family
        # touching a wrapping face adopts that face's loop vertex as its meridian.
        members=[f for f,p in patches.items() if ring_face(f,p)];family={f:f for f in members}
        def root(f):
            while family[f]!=f:family[f]=family[family[f]];f=family[f]
            return f
        for e in graph.edges:
            a,b=e.faces
            if a in family and b in family:family[root(a)]=root(b)
        reference={}
        for f in members:
            for e in graph.edges:
                if f in e.faces and e.id in ring_anchor and root(f) not in reference:
                    reference[root(f)]=ring_anchor[e.id]-np.asarray(curves[e.id].data['center'],float)
        for f in members:reference.setdefault(root(f),np.asarray(patches[f].data['x'],float))
        for f in members:
            patch=patches[f];axis=np.asarray(patch.data['axis'],float);ref=reference[root(f)]
            x=ref-(ref@axis)*axis
            if np.linalg.norm(x)<1e-6:continue
            x/=np.linalg.norm(x);patch.data['x']=x;patch.data['y']=np.cross(axis,x)
            for e in graph.edges:
                if f not in e.faces or e.id in planned:continue
                d=curves[e.id].data;xr=ref-(ref@d['normal'])*d['normal'];xr/=np.linalg.norm(xr)
                d['start']=float(np.arctan2(xr@d['y'],xr@d['x']));planned.add(e.id)
    for fid,patch in sorted(patches.items(),key=lambda item:0 if item[1].kind=='sphere' else 1):
        if cfg.periodic_frame_planning or not ring_face(fid,patch):continue
        incident=[e for e in graph.edges if fid in e.faces]
        first=next((e for e in incident if e.id in planned),None)
        if first is not None:
            d=curves[first.id].data;anchor=np.cos(d['start'])*d['x']+np.sin(d['start'])*d['y']
            axis=patch.data['axis'];x=anchor-(anchor@axis)*axis;x/=np.linalg.norm(x)
            patch.data['x']=x;patch.data['y']=np.cross(axis,x)
        for e in incident:
            if e.id in planned:continue
            d=curves[e.id].data;x=patch.data['x'];x=x-(x@d['normal'])*d['normal'];x/=np.linalg.norm(x)
            d['start']=float(np.arctan2(x@d['y'],x@d['x']));planned.add(e.id)
    # A periodic face whose boundary leaves an angular gap needs no seam at all:
    # rotate its parameter frame so the u-cut falls in that gap.  Shared 3D curves
    # and the support surface are unchanged; only the parameter origin moves.
    seamless={};chained={};band={}
    if cfg.periodic_frame_planning:
        for fid,patch in patches.items():
            if patch.kind not in PERIODIC_KINDS:continue
            incident=[e for e in graph.edges if fid in e.faces]
            if not incident or any(e.closed for e in incident):continue
            pts=np.vstack([curves[e.id].evaluate(np.linspace(0,1,257)) for e in incident])
            gaps=[]
            for dim in periodic_dims(patch.kind):
                angle=np.sort(np.mod(patch.uv(pts)[:,dim],2*np.pi))
                gap=np.diff(np.r_[angle,angle[0]+2*np.pi]);i=int(gap.argmax());gaps.append((float(gap[i]),float(angle[i]+gap[i]/2)))
            if min(g for g,_ in gaps)<.05:continue
            rotation=gaps[0][1]-np.pi
            x=patch.data['x'].copy();y=patch.data['y'].copy()
            patch.data['x']=np.cos(rotation)*x+np.sin(rotation)*y
            patch.data['y']=-np.sin(rotation)*x+np.cos(rotation)*y
            # After rotation the u-cut sits at +-pi; a torus v-cut sits mid-gap.
            seamless[fid]=np.array([-np.pi]+([gaps[1][1]] if len(gaps)>1 else []))
            mapping.setdefault('frame_planning',[]).append({'face':fid,'kind':patch.kind,'rotation':float(rotation),'gaps':[g for g,_ in gaps]})
        # Fully wrapping faces still need a seam.  Chain each loop's pcurves by
        # whole periods so a loop is open in UV only at its first vertex, and start
        # unplanned closed rings on that vertex's meridian; FixMissingSeam can then
        # join vertices instead of splitting a shared edge.
        for fid,patch in patches.items():
            if patch.kind not in PERIODIC_KINDS or fid in seamless or ring_face(fid,patch):continue
            incident=[e for e in graph.edges if fid in e.faces]
            if not incident or all(e.closed for e in incident):continue
            offsets={};anchor=wrap_anchor(fid)
            for cycle in graph.cycles.get(fid,[]):
                previous=None
                for use in cycle:
                    uv=patch.uv(curves[use.edge].evaluate(np.linspace(0,1,65)),ordered=True)
                    start,end=(uv[0],uv[-1]) if use.forward else (uv[-1],uv[0])
                    offset=np.zeros(2)
                    if previous is not None:
                        for d in periodic_dims(patch.kind):offset[d]=2*np.pi*np.round((previous[d]-start[d])/(2*np.pi))
                    offsets[use.edge]=offset;previous=end+offset
            aligned=[]
            if anchor is not None:
                for e in incident:
                    if not e.closed or e.id in planned or curves[e.id].kind!='circle':continue
                    d=curves[e.id].data;r=anchor-d['center'];r=r-(r@d['normal'])*d['normal']
                    if np.linalg.norm(r)<1e-9:continue
                    d['start']=float(np.arctan2(r@d['y'],r@d['x']));planned.add(e.id);aligned.append(e.id)
            # FixMissingSeam joins wires into one rectangle only if every wire sits in
            # the same period; move each open-edge loop onto the closed rings' period.
            rings=[e for e in incident if e.closed]
            if rings:
                target=np.mean([patch.uv(curves[e.id].evaluate(np.linspace(0,1,65)),ordered=True)[:,0].mean() for e in rings])
                for cycle in graph.cycles.get(fid,[]):
                    if len(cycle)==1 and graph.edges[cycle[0].edge].closed:continue
                    mid=np.mean([(patch.uv(curves[u.edge].evaluate(np.linspace(0,1,65)),ordered=True)[:,0]+offsets[u.edge][0]).mean() for u in cycle])
                    shift=2*np.pi*np.round((target-mid)/(2*np.pi))
                    for u in cycle:offsets[u.edge]=offsets[u.edge]+np.array([shift,0.])
            plan=plan_periodic_band(patch,graph.cycles.get(fid,[]),curves,graph,offsets,anchor) if patch.kind in ('cylinder','cone') and anchor is not None else None
            if plan is not None:band[fid]=plan;offsets=plan['offsets']
            chained[fid]=offsets
            mapping.setdefault('seam_vertex_planning',[]).append({'face':fid,'kind':patch.kind,'aligned_rings':aligned,'chained_edges':len(offsets)})
    for eid,record in curves.items():
        curve,first,last=occ_curve(record);instance=graph.edges[eid]
        if instance.vertices:
            make=BRepBuilderAPI_MakeEdge(curve,occ_vertices[instance.vertices[0]],occ_vertices[instance.vertices[1]],first,last)
        elif instance.closed:
            anchor=BRepBuilderAPI_MakeVertex(curve.Value(first)).Vertex()
            make=BRepBuilderAPI_MakeEdge(curve,anchor,anchor,first,last);mapping['auxiliary_vertices'].append({'logical_edge':eid,'kind':'closed_loop_anchor'})
        else:make=BRepBuilderAPI_MakeEdge(curve,first,last)
        require(make.IsDone(),'cad_edge_failed','OCC rejected shared edge',edge=eid)
        edges[eid]=make.Edge();ranges[eid]=(first,last)
    faces={};surfaces={};uv_reports={}
    for fid,patch in patches.items():
        surface=occ_surface(patch);face=TopoDS_Face();builder.MakeFace(face,surface,cfg.cad_tolerance*.1)
        faces[fid]=face;surfaces[fid]=surface;uv_reports[fid]=[]
        for instance in graph.edges:
            if fid not in instance.faces:continue
            if fid in seamless:adjust=lambda uv,s=seamless[fid],k=patch.kind:shift_periods(uv,s,k)
            elif fid in chained and instance.id in chained[fid]:adjust=lambda uv,o=chained[fid][instance.id]:uv+o
            else:adjust=None
            pcurve,uv,error=make_pcurve(patch,curves[instance.id],*ranges[instance.id],cfg.boundary_tolerance,adjust)
            builder.UpdateEdge(edges[instance.id],pcurve,face,cfg.boundary_tolerance)
            builder.Range(edges[instance.id],face,*ranges[instance.id]);uv_reports[fid].append({'edge':instance.id,'error':error})
        incident=[e for e in graph.edges if fid in e.faces]
        if not incident:
            require(patch.kind in ('sphere','torus'),'unbounded_face','Boundary-free face requires a closed support surface',face=fid)
            face=BRepBuilderAPI_MakeFace(surface,cfg.boundary_tolerance).Face();faces[fid]=face
            mapping['postprocess_edges'].extend({'face':fid,'kind':'natural_periodic_boundary'} for _ in unique(face,TopAbs_EDGE))
            q=np.asarray(patch.data['orientation_point'])[None];uv=patch.uv(q);h=1e-5;base=patch.evaluate(uv)
            n=np.cross((patch.evaluate(uv+[h,0])-base)[0],(patch.evaluate(uv+[0,h])-base)[0])
            if n@patch.normal<0:face=topods.Face(face.Reversed());faces[fid]=face
        explicit_periodic=patch.kind in ring_kinds and (len(incident)==2 or (patch.kind=='sphere' and len(incident)==1)) and all(e.closed and curves[e.id].kind=='circle' for e in incident)
        if explicit_periodic:
            face=make_spherical_cap(patch,surface,incident[0],curves,edges,ranges,cfg,builder) if len(incident)==1 else make_periodic_annulus(patch,surface,incident,curves,edges,ranges,cfg,builder)
            faces[fid]=face
            mapping['postprocess_edges'].append({'face':fid,'kind':'explicit_periodic_seam'})
        elif fid in band:
            face=make_periodic_band(patch,surface,band[fid],graph,edges,occ_vertices,cfg,builder,face)
            faces[fid]=face;explicit_periodic=True
            mapping['postprocess_edges'].append({'face':fid,'kind':'explicit_band_seam'})
        for cycle in ([] if explicit_periodic else graph.cycles[fid]):
            wire=TopoDS_Wire();builder.MakeWire(wire)
            for use in cycle:
                edge=edges[use.edge] if use.forward else topods.Edge(edges[use.edge].Reversed())
                builder.Add(wire,edge)
            wire.Closed(True);builder.Add(face,wire)
        # Numeric patch may have parameter normal opposite to oriented mesh normal.
        # The cycles already use orientation of the physical face; invert face plus
        # its wires if necessary so that both UV and 3D conventions remain valid.
        if graph.edges and not explicit_periodic:
            e=next((e for e in graph.edges if fid in e.faces),None)
            if e:
                q=np.asarray(patch.data.get('orientation_point',curves[e.id].evaluate(np.array([.35]))[0]))[None];uv=patch.uv(q);h=1e-5
                n=np.cross((patch.evaluate(uv+[h,0])-patch.evaluate(uv))[0],(patch.evaluate(uv+[0,h])-patch.evaluate(uv))[0])
                if np.dot(n,patch.normal)<0:
                    # Reverse all contained wires relative to support orientation,
                    # then reverse face to recover physical coedge orientation.
                    replacement=TopoDS_Face();builder.MakeFace(replacement,surface,cfg.cad_tolerance*.1)
                    from OCC.Core.TopAbs import TopAbs_WIRE
                    for wire in unique(face,TopAbs_WIRE):builder.Add(replacement,wire.Reversed())
                    face=topods.Face(replacement.Reversed());faces[fid]=face
        if incident and (surface.IsUPeriodic() or surface.IsVPeriodic()) and not explicit_periodic and fid not in seamless:
            fixer=ShapeFix_Face(face);fixer.SetPrecision(cfg.boundary_tolerance);fixer.SetMaxTolerance(cfg.cad_tolerance)
            fixer.FixMissingSeam()
            candidate=fixer.Face()
            # Original shared edges must survive by identity; replacement is not accepted.
            present=unique(candidate,TopAbs_EDGE)
            for e in graph.edges:
                if fid in e.faces:require(any(x.IsSame(edges[e.id]) for x in present),'repair_changed_shared_edge','Seam repair replaced a logical edge',face=fid,edge=e.id)
            additions=[x for x in present if not any(x.IsSame(y) for y in edges.values())]
            mapping['postprocess_edges'].extend({'face':fid,'kind':'periodic_seam'} for x in additions)
            faces[fid]=candidate;mapping['repairs'].append({'face':fid,'operation':'FixMissingSeam','added_edges':len(additions)})
        for x in unique(faces[fid],TopAbs_EDGE):breplib.SameParameter(topods.Edge(x),cfg.boundary_tolerance,True)
        analyzer=BRepCheck_Analyzer(faces[fid])
        if not analyzer.IsValid():
            breptools.Write(faces[fid],str(out/f'rejected_face_{fid}.brep'))
            require(False,'cad_face_invalid','OCC face invalid; inspect pcurves/wires/seams',face=fid,kind=patch.kind,
                    checks=check_statuses(faces[fid]))
        mapping['faces'][fid]={'kind':patch.kind,'pcurves':uv_reports[fid]}
    from .surface_mesh import components
    pairs=np.array([e.faces for e in graph.edges],int).reshape(-1,2)
    _,shell_labels=components(pairs,graph.face_count)
    solid=TopoDS_Solid();builder.MakeSolid(solid);shell_volumes=[]
    for sid in np.unique(shell_labels):
        shell=TopoDS_Shell();builder.MakeShell(shell)
        for fid in np.flatnonzero(shell_labels==sid):builder.Add(shell,faces[int(fid)])
        shell.Closed(True);builder.Add(solid,shell)
        props=GProp_GProps();brepgprop.VolumeProperties(shell,props);shell_volumes.append(float(props.Mass()))
    require(sum(v>0 for v in shell_volumes)==1,'unsupported_multi_solid','Expected one exterior shell and optional inward cavity shells',shell_volumes=shell_volumes)
    mapping['shell_volumes']=shell_volumes
    # Orient shell membership without an unrestricted ShapeFix pass that copies
    # edges / increases tolerances. Direct assembly keeps shared identities.
    require(breplib.OrientClosedSolid(solid),'solid_orientation_failed','Cannot orient closed solid')
    for eid,edge in edges.items():
        breplib.SameParameter(edge,cfg.boundary_tolerance,True)
        uses=[]
        for fid,face in faces.items():
            explorer=TopExp_Explorer(face,TopAbs_EDGE)
            while explorer.More():
                x=explorer.Current()
                if x.IsSame(edge):uses.append((fid,int(x.Orientation())))
                explorer.Next()
        require(len(uses)==2 and uses[0][0]!=uses[1][0] and uses[0][1]!=uses[1][1],
                'cad_incidence_invalid','Logical edge must have two opposite face uses',edge=eid,uses=uses)
        mapping['logical_edges'][eid]={'cad_edge_index':next(i for i,x in enumerate(unique(solid,TopAbs_EDGE)) if x.IsSame(edge)),
                                       'face_uses':uses,'parameter_range':ranges[eid]}
    require(tolerance_max(solid)<=cfg.cad_tolerance*1.01,'cad_tolerance_exceeded','Repair exceeded tolerance budget')
    require(BRepCheck_Analyzer(solid).IsValid(),'cad_solid_invalid','OCC solid invalid')
    breptools.Write(solid,str(out/'curves_and_faces.brep'))
    return solid,mapping

def verify_export_and_roundtrip(solid,graph,curves,patches,mapping,cfg,out):
    props=GProp_GProps();brepgprop.VolumeProperties(solid,props);volume=props.Mass()
    require(volume>0,'cad_volume_invalid','Solid has nonpositive volume')
    mesher=BRepMesh_IncrementalMesh(solid,cfg.surface_tolerance,False,.15,False);mesher.Perform()
    require(mesher.IsDone(),'cad_mesh_failed','Final solid cannot be triangulated')
    path=out/'reconstruction.unverified.step'
    writer=STEPControl_Writer();require(writer.Transfer(solid,STEPControl_AsIs)==IFSelect_RetDone,'step_transfer_failed','STEP transfer failed')
    require(writer.Write(str(path))==IFSelect_RetDone,'step_write_failed','STEP export failed')
    reader=STEPControl_Reader();require(reader.ReadFile(str(path))==IFSelect_RetDone,'step_read_failed','STEP readback failed');reader.TransferRoots();readback=reader.OneShape()
    require(BRepCheck_Analyzer(readback).IsValid(),'step_readback_invalid','Exported STEP fails CAD validity')
    rprops=GProp_GProps();brepgprop.VolumeProperties(readback,rprops)
    surface_props=GProp_GProps();brepgprop.SurfaceProperties(solid,surface_props)
    volume_budget=max(1e-7,abs(volume)*1e-5,surface_props.Mass()*cfg.boundary_tolerance)
    require(abs(rprops.Mass()-volume)<=volume_budget,'step_volume_changed','STEP roundtrip changed volume beyond geometry budget',
            original=volume,readback=rprops.Mass(),budget=volume_budget)
    require(len(unique(readback,TopAbs_FACE))==graph.face_count,'step_topology_changed','STEP roundtrip changed face count')
    # Geometry checks survive STEP re-numbering; never rely on original OCC IDs.
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    read_edges=unique(readback,TopAbs_EDGE);maximum=0.
    for record in curves.values():
        for point in record.evaluate(np.linspace(0,1,33)):
            vertex=BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
            errors=[BRepExtrema_DistShapeShape(vertex,x).Value() for x in read_edges]
            maximum=max(maximum,min(errors))
    require(maximum<=cfg.cad_tolerance,'step_geometry_changed','Readback lost shared curve geometry',error=maximum)
    path.rename(out/'reconstruction.step')
    return {'valid':True,'volume':volume,'roundtrip_volume':rprops.Mass(),'roundtrip_volume_error':abs(rprops.Mass()-volume),'roundtrip_volume_budget':volume_budget,'max_tolerance':tolerance_max(solid),
        'readback_curve_error':maximum,'faces':len(unique(solid,TopAbs_FACE)),'cad_edges':len(unique(solid,TopAbs_EDGE)),
        'logical_edges':len(graph.edges),'logical_vertices':len(graph.vertices),'roundtrip_checked':True}
