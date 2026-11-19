"""Conservative scalar-field-only face repair with immutable evidence.

Face IDs, absorbed masks and downstream retries are separate decisions. No CAD
identity, truth labels, topology scores or target geometry enter this module.
"""
from dataclasses import dataclass
import heapq
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from .deadline import checkpoint
from .face_seeds import core_components
from .schemas import ReconstructionError, require
from .surface_mesh import face_adjacency, normals, evaluate_batched


@dataclass
class RegionState:
    raw_labels: np.ndarray
    raw_reasons: np.ndarray
    labels: np.ndarray
    trusted: np.ndarray
    absorbed: np.ndarray
    protected_e: np.ndarray
    protected_v: np.ndarray
    report: dict


def compact(labels):
    result=np.full(len(labels),-1,np.int32);ids=np.flatnonzero(labels>=0)
    if len(ids):result[ids]=np.searchsorted(np.unique(labels[ids]),labels[ids])
    return result


def sample_ids(ids,count):
    ids=np.asarray(ids,int)
    return ids if len(ids)<=count else ids[np.linspace(0,len(ids)-1,count).astype(int)]


def _neighbor_graph(pairs,points):
    n=len(points);a,b=pairs.T
    weight=np.maximum(np.linalg.norm(points[a]-points[b],axis=1),1e-12)
    return csr_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(n,n))


def local_owners(ids,labels,graph,radius,max_nodes=4096):
    """Walk the surface graph, retaining witness paths instead of jumping sheets."""
    starts=set(map(int,ids));distance={i:0. for i in starts};previous={};queue=[(0.,i) for i in starts]
    heapq.heapify(queue);owners={};visited=0
    while queue and visited<max_nodes:
        d,i=heapq.heappop(queue)
        if d!=distance[i]:continue
        visited+=1
        if labels[i]>=0 and i not in starts:
            f=int(labels[i])
            if f not in owners:
                path=[i]
                while path[-1] in previous:path.append(previous[path[-1]])
                owners[f]=(d,np.asarray(path,int))
            # Do not walk through another face to find a spatially nearby face.
            continue
        for pos in range(graph.indptr[i],graph.indptr[i+1]):
            j=int(graph.indices[pos]);nd=d+graph.data[pos]
            if nd<=radius and nd<distance.get(j,np.inf):
                distance[j]=nd;previous[j]=i;heapq.heappush(queue,(nd,j))
    return owners


def _patch_fit(points,nrm,reference,extra,cfg,areas=None):
    """Fit on deterministic area-distributed references; validate held-out nodes."""
    from .surface_fitting import analytic_candidates
    def area_sample(count,seed):
        if len(reference)<=count:return np.asarray(reference,int)
        rng=np.random.default_rng(cfg.seed+seed)
        weight=areas[reference] if areas is not None else np.ones(len(reference))
        return np.asarray(reference)[rng.choice(len(reference),count,replace=False,p=weight/weight.sum())]
    ref=area_sample(cfg.region_repair_samples,0)
    if len(ref)<3:return None
    options=analytic_candidates(points[ref],nrm[ref],np.empty((0,3)))
    hold=area_sample(4*cfg.region_repair_samples,1)
    test=np.unique(np.r_[hold,extra]);p=points[test];best=None
    for patch in options:
        uv=patch.uv(p);q=patch.evaluate(uv);error=np.linalg.norm(p-q,axis=1)
        if not np.isfinite(error).all() or error.max()>cfg.boundary_tolerance:continue
        h=1e-5;u=(patch.evaluate(uv+[h,0])-q)/h;v=(patch.evaluate(uv+[0,h])-q)/h
        pn=np.cross(u,v);length=np.linalg.norm(pn,axis=1)
        if np.any(length<1e-10):continue
        alignment=np.abs(np.einsum('ij,ij->i',pn/length[:,None],nrm[test]))
        if alignment.min()<.98:continue
        score=float(error.max())
        if best is None or score<best['error']:best=dict(kind=patch.kind,error=score,normal_min=float(alignment.min()))
    return best


def _probe_ids(ids,values,count):
    ids=np.asarray(ids,int)
    if not len(ids):return ids
    # Include minima and difficult locations; a uniform reservoir alone misses
    # short E/V branches. The original centers are always retained unchanged.
    k=min(len(ids),max(4,count//4))
    return np.unique(np.r_[sample_ids(ids,count),ids[np.argsort(values[ids,1])[:k]],
                            ids[np.argsort(values[ids,2])[:k]]])


def _field_guard(points,values,ids,provider,cfg,tree):
    """Keep near-zero probes independently of changed labels or masks.

    A valley projection landing back in the proposed patch vetoes absorption.
    Projections outside it must be covered by retained curves in a later check.
    """
    chosen=_probe_ids(ids,values,cfg.region_repair_samples)
    if not len(chosen):return dict(accepted=False,reason='empty_patch'),np.empty((0,3)),np.empty((0,3))
    selected=set(map(int,ids));threshold=max(cfg.field_error*.5,cfg.curve_tolerance*.05)
    original=points[chosen].copy();q=original.copy();v=values[chosen].copy()
    if np.min(v[:,1])<=threshold:
        return dict(accepted=False,reason='observed_E_valley'),q[v[:,1]<=threshold],np.empty((0,3))
    if provider.presence()[1]=='present' and np.min(v[:,2])<=cfg.localization_tolerance:
        return dict(accepted=False,reason='necessary_V_near_patch'),np.empty((0,3)),q[v[:,2]<=cfg.localization_tolerance]
    for _ in range(8):
        checkpoint('region_repair_probes')
        active=np.flatnonzero(v[:,1]>threshold)
        if not len(active):break
        if hasattr(provider,'gradient_from_values'):g,valid=provider.gradient_from_values(q[active],1,v[active])
        else:g,valid=provider.gradient(q[active],1)
        norm=np.linalg.norm(g,axis=1);ok=valid&(norm>1e-10)
        # Invalid / medial-axis derivatives are uncertainty, never permission to merge.
        if not ok.all():return dict(accepted=False,reason='uncertain_E_gradient'),np.empty((0,3)),np.empty((0,3))
        proposal=q[active]-.8*v[active,1,None]*g/norm[:,None]
        if np.max(np.linalg.norm(proposal-original[active],axis=1))>cfg.refinement_rho_edge*2:
            return dict(accepted=False,reason='projection_left_local_patch'),np.empty((0,3)),np.empty((0,3))
        q[active]=proposal;v[active]=evaluate_batched(provider,proposal,cfg.query_batch_size)
    localized=v[:,1]<=cfg.curve_tolerance
    if not localized.all():return dict(accepted=False,reason='E_valley_unlocalized'),q[localized],np.empty((0,3))
    owner=tree.query(q)[1]
    if any(int(i) in selected for i in owner):
        return dict(accepted=False,reason='independent_E_branch_in_patch'),q,np.empty((0,3))
    return dict(accepted=True,reason='valleys_outside_patch',samples=len(chosen)),q,np.empty((0,3))


def _side_contacts(ids,labels,pairs):
    """Require actual cross-edge contact on both sides, not vertex-only contact."""
    active=np.zeros(len(labels),bool);active[ids]=True
    a,b=pairs.T;cut=active[a]^active[b]
    inside=np.where(active[a[cut]],a[cut],b[cut]);outside=np.where(active[a[cut]],b[cut],a[cut])
    known=labels[outside]>=0
    return inside[known],outside[known],np.unique(labels[outside[known]])


def repair_regions(mesh,points,values,labels,reasons,provider,cfg,progress=None):
    started=time.monotonic();pairs=face_adjacency(mesh.triangles)
    active=(values[:,1]>cfg.face_threshold1 if cfg.band_threshold_inclusive else values[:,1]>=cfg.face_threshold1)
    if provider.presence()[0]=='absent':active=np.ones(len(points),bool)
    raw=core_components(pairs,active)
    current=labels.copy();trusted=current>=0;absorbed=np.zeros(len(points),bool)
    report=dict(mode=cfg.region_repair,actions=[],rounds=0,accepted=0,
                raw_components=int(raw.max()+1),original_faces=int(current.max()+1),
                local_threshold_recovery='only witnessed original small components; no blind new face creation')
    ep=[];vp=[];nrm=normals(mesh);tree=cKDTree(points);surface_graph=None
    tri=mesh.vertices[mesh.triangles]
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    del tri
    attempted=set()
    def record(entry):
        report['actions'].append(entry)
        if progress:progress({**report,'elapsed_seconds':time.monotonic()-started})
    def test_patch(ids,owner,reference,kind,extra=None):
        nonlocal current,trusted,absorbed
        checkpoint('region_repair_candidate')
        patch=np.asarray(ids,int);entry=dict(kind=kind,target=int(owner),nodes=patch.tolist(),accepted=False)
        # Cheap independent evidence rejects ordinary real interfaces before
        # constructing expensive analytic surface candidates.
        if np.min(values[patch,1])<=max(cfg.field_error*.5,cfg.curve_tolerance*.05):
            entry['reason']='observed_E_valley';record(entry);return False
        if provider.presence()[1]=='present' and np.min(values[patch,2])<=cfg.localization_tolerance:
            entry['reason']='necessary_V_near_patch';record(entry);return False
        fit=_patch_fit(points,nrm,reference,patch if extra is None else np.unique(np.r_[patch,extra]),cfg,area)
        entry['geometry']=fit
        if fit is None:entry['reason']='incompatible_support_surface';record(entry);return False
        try:guard,e,v=_field_guard(points,values,patch,provider,cfg,tree)
        except ReconstructionError as exc:
            if exc.code=='sample_timeout':raise
            guard=dict(accepted=False,reason=exc.code);e=v=np.empty((0,3))
        entry['field']=guard
        if not guard['accepted']:entry['reason']=guard['reason'];record(entry);return False
        ep.extend(e);vp.extend(v)
        current[patch]=owner;absorbed[patch]=True
        # Small original high-distance cores remain trustworthy surface evidence;
        # absorbed low-E corridors do not become interior samples.
        trusted[patch]=active[patch]
        entry.update(accepted=True,reason='verified_absorption');report['accepted']+=1;record(entry);return True
    if cfg.region_repair in ('small','joint'):
        surface_graph=_neighbor_graph(pairs,points)
        small=core_components(pairs,(raw>=0)&(reasons==3))
        for sid in range(int(small.max()+1)):
            ids=np.flatnonzero(small==sid);owners=local_owners(ids,current,surface_graph,cfg.refinement_rho_edge)
            options=[]
            for f,(distance,path) in sorted(owners.items()):
                ref=np.flatnonzero((current==f)&trusted)
                fit=_patch_fit(points,nrm,ref,ids,cfg,area)
                if fit is not None:options.append((f,distance,path,ref))
            if len(options)==1:
                f,_,path,ref=options[0]
                # The witness corridor must be explained too, otherwise relabeling
                # the island leaves a one-face pseudo-edge between equal IDs.
                corridor=np.unique(np.r_[ids,path[current[path]<0]])
                test_patch(corridor,f,ref,'small_component')
            elif not options and len(owners)>=2:
                # Restore a filtered real patch only when independent interior
                # samples stay above threshold and its normals are coherent.
                tri=mesh.vertices[mesh.triangles[ids]]
                samples=np.vstack((tri.mean(1),.6*tri[:,0]+.2*tri[:,1]+.2*tri[:,2],
                    .2*tri[:,0]+.6*tri[:,1]+.2*tri[:,2],.2*tri[:,0]+.2*tri[:,1]+.6*tri[:,2]))
                vv=evaluate_batched(provider,samples,cfg.query_batch_size)
                coherent=np.min(nrm[ids]@nrm[ids].mean(0))>.98
                if coherent and np.min(vv[:,1])>cfg.face_threshold1+cfg.field_error:
                    f=int(current.max()+1);current[ids]=f;trusted[ids]=True
                    report['accepted']+=1;record(dict(kind='restore_small_face',target=f,nodes=ids.tolist(),accepted=True,reason='independent_interior'))
                else:record(dict(kind='small_component',nodes=ids.tolist(),accepted=False,reason='independent_face_unconfirmed'))
            else:record(dict(kind='small_component',nodes=ids.tolist(),accepted=False,reason='ambiguous_owner' if options else 'no_local_owner'))
    # Build temporary instances with the same local V domains as the production
    # graph. Short edges are deliberately inferred only after repair stabilizes.
    if cfg.region_repair in ('single_face','joint'):
        from .bands import bands_from_labels
        from .topology import collect_labels
        from .surface_mesh import vertex_adjacency
        adjacency=None
        for round_id in range(cfg.region_repair_max_rounds):
            checkpoint('region_repair_round');report['rounds']=round_id+1;changed=False
            current=compact(current)
            effective=reasons.copy();effective[current>=0]=0
            temporary=bands_from_labels(mesh,points,values,current,effective,provider,cfg,dict(count=int(current.max()+1)))
            if adjacency is None:adjacency=vertex_adjacency(mesh.triangles) if cfg.band_connectivity=='vertex' else pairs
            topo=collect_labels(adjacency,temporary,current)
            for edge in topo.edges:
                faces=edge['faces'];ids=np.asarray(edge['nodes'],int)
                if len(faces)!=1 and not (cfg.region_repair=='joint' and len(faces)==2):continue
                key=(tuple(faces),tuple(ids))
                if key in attempted:continue
                attempted.add(key)
                inside,outside,contact=_side_contacts(ids,current,pairs)
                if not len(outside) or set(contact.tolist())!=set(faces):continue
                # A coherent patch must have genuine shared-edge contacts spanning
                # its boundary. Unknown V/junction sectors veto wholesale absorption.
                active_ids=np.zeros(len(current),bool);active_ids[ids]=True
                a,b=pairs.T;cut=active_ids[a]^active_ids[b]
                boundary=np.where(active_ids[a[cut]],b[cut],a[cut])
                if np.any((current[boundary]<0)&~active_ids[boundary]):
                    record(dict(kind='single_face_band' if len(faces)==1 else 'merge_face_pair',nodes=ids.tolist(),accepted=False,reason='open_junction_or_unknown_side'));continue
                f=int(faces[0]);ref=np.flatnonzero(np.isin(current,faces)&trusted)
                if test_patch(ids,f,ref,'single_face_band' if len(faces)==1 else 'merge_face_pair',extra=outside):
                    if len(faces)==2:current[current==faces[1]]=f
                    changed=True
                    # Rebuild all temporary instances after each transaction;
                    # never use stale edge IDs or endpoint domains for another edit.
                    break
            if not changed:break
    current=compact(current)
    report.update(final_faces=int(current.max()+1),absorbed_triangles=int(absorbed.sum()),
                  trusted_triangles=int(trusted.sum()),elapsed_seconds=time.monotonic()-started,
                  probe_guarantee='finite local samples; downstream E/V coverage and full geometry acceptance required')
    return RegionState(raw,reasons.copy(),current,trusted,absorbed,np.asarray(ep).reshape(-1,3),np.asarray(vp).reshape(-1,3),report)


def validate_repair_coverage(bands,curves,vertices,provider,cfg):
    """Candidate-local checks survive label edits and supplement global probes."""
    state=getattr(bands,'repair_state',None)
    if state is None or not state.report['accepted']:return dict(tested_e=0,tested_v=0)
    from .curve_distance_fitting import closest_parameters
    e=state.protected_e
    if len(e):
        require(bool(curves),'repair_edge_coverage_missing','No retained curves explain absorbed-region probes')
        distance=np.full(len(e),np.inf)
        for curve in curves.values():distance=np.minimum(distance,closest_parameters(curve,e)[1])
        require(distance.max()<=cfg.curve_tolerance*3,'repair_edge_coverage_failed','Absorption deleted an independent E branch',error=float(distance.max()))
    # Check original low-V evidence at all changed triangles, not only random
    # global probes. Low V is protected through fitted vertex positions.
    ids=np.flatnonzero(state.absorbed & (bands.values[:,2]<=cfg.tau_vertex)) if provider.presence()[1]=='present' else np.empty(0,int)
    if len(ids):
        ids=_probe_ids(ids,bands.values,cfg.region_repair_samples)
        p=bands.points[ids];g,valid=provider.gradient(p,2);norm=np.linalg.norm(g,axis=1)
        require((valid&(norm>1e-10)).all(),'repair_vertex_probe_invalid','Cannot verify changed V neighborhood')
        votes=p-bands.values[ids,2,None]*g/norm[:,None]
        require(bool(vertices),'repair_vertex_coverage_missing','No retained vertices explain changed V neighborhood')
        error=cKDTree(np.array([v.position for v in vertices])).query(votes)[0]
        require(error.max()<=cfg.localization_tolerance*2,'repair_vertex_coverage_failed','Absorption removed a necessary V feature',error=float(error.max()))
    return dict(tested_e=len(e),tested_v=len(ids),continuous_coverage_certified=False)
