from collections import defaultdict
import numpy as np
from .schemas import EdgeInstance,LogicalGraph,require


def instantiate_edges(interface,vertices,bands,cfg,face_count):
    bypair=defaultdict(list)
    for s in interface.segments:bypair[tuple(s['faces'])].append(s)
    edges=[]
    for pair,segments in sorted(bypair.items()):
        neighbors=defaultdict(list);records={}
        for s in segments:
            a,b=s['nodes'];key=tuple(sorted((a,b)))
            if key in records:continue
            records[key]=s;neighbors[a].append(b);neighbors[b].append(a)
        require(all(len(v)<=2 for v in neighbors.values()),'interface_branch_ambiguous','Multiway interface outside vertex band; revisit bands',faces=pair)
        remaining=set(records)
        while remaining:
            start_edge=min(remaining);a,b=start_edge
            # Find chain endpoint before walking; a degree-2-only branch is a cycle.
            component=set();stack=[a]
            while stack:
                x=stack.pop()
                if x in component:continue
                component.add(x);stack.extend(neighbors[x])
            endpoints=[x for x in component if len(neighbors[x])==1]
            require(len(endpoints) in (0,2),'interface_broken','Interface component has invalid endpoint count')
            start=min(endpoints) if endpoints else min(component);path=[start];prev=None;cur=start
            while True:
                options=[x for x in neighbors[cur] if x!=prev and tuple(sorted((cur,x))) in remaining]
                if not options:break
                nxt=min(options);remaining.remove(tuple(sorted((cur,nxt))));path.append(nxt);prev,cur=cur,nxt
                if cur==start:break
            closed=path[-1]==start
            pts=interface.points[path]
            s=records[tuple(sorted(path[:2]))]
            left=s['left'] if tuple(path[:2])==tuple(s['nodes']) else (pair[1] if s['left']==pair[0] else pair[0])
            if left!=pair[0]:pts=pts[::-1];left=pair[0]
            ends=[]
            if not closed:
                require(len(vertices)>0,'unexplained_edge_endpoint','Open interface without vertex candidates')
                for p in (pts[0],pts[-1]):
                    distance=np.array([np.linalg.norm(p-v.position) for v in vertices]);order=np.argsort(distance)
                    require(distance[order[0]]<cfg.tau_vertex+cfg.max_band_edge*4,'port_unmatched','Interface endpoint outside all vertex supports')
                    if len(order)>1:require(distance[order[1]]-distance[order[0]]>cfg.localization_tolerance,'port_ambiguous','Port compatible with several vertices; shrink vertex band')
                    ends.append(int(order[0]))
                pts=np.vstack((vertices[ends[0]].position,pts,vertices[ends[1]].position))
            # Required degree-two vertices may lie inside an otherwise closed/continuous chain.
            pieces=[(pts,closed,ends)]
            for v in vertices:
                updated=[]
                for p,c,e in pieces:
                    if v.id in e:updated.append((p,c,e));continue
                    distance=np.linalg.norm(p-v.position,axis=1);j=int(distance.argmin())
                    if distance[j]>cfg.localization_tolerance*3:updated.append((p,c,e));continue
                    if c:
                        ring=p[:-1];j=min(j,len(ring)-1);q=np.vstack((v.position,ring[j+1:],ring[:j+1],v.position))
                        updated.append((q,False,[v.id,v.id]))
                    elif 0<j<len(p)-1:
                        updated.extend([(np.vstack((p[:j],v.position)),False,[e[0],v.id]),
                                        (np.vstack((v.position,p[j+1:])),False,[v.id,e[1]])])
                    else:updated.append((p,c,e))
                pieces=updated
            for p,c,e in pieces:
                require(len(p)>=3,'insufficient_edge_support','Edge has insufficient interface samples')
                edge=EdgeInstance(len(edges),pair,p,c,e,left,cfg.tau_edge+cfg.max_band_edge*2)
                edges.append(edge)
                for end,vid in enumerate(e):vertices[vid].ports.append({'edge':edge.id,'end':end,'faces':pair,'point':p[0 if end==0 else -1]})
    graph=LogicalGraph(face_count,vertices,edges)
    require(all(v.ports for v in vertices),'vertex_without_ports','Necessary vertex has no recovered incident edges; short edge may be swallowed')
    return graph
