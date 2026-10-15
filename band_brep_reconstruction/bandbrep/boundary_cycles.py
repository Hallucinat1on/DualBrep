from .schemas import CoedgeUse,require

def complete_face_regions_and_cycles(graph):
    cycles={}
    for f in range(graph.face_count):
        uses=[CoedgeUse(f,e.id,f==e.left_face) for e in graph.edges if f in e.faces]
        closed=[];open_uses=[]
        for u in uses:
            e=graph.edges[u.edge]
            if e.closed:closed.append([u])
            else:open_uses.append(u)
        remaining={u.edge:u for u in open_uses}
        while remaining:
            start=next(iter(remaining.values()));cycle=[];u=start
            e=graph.edges[u.edge];first=e.vertices[0 if u.forward else 1]
            for _ in range(len(open_uses)+1):
                require(u.edge in remaining,'wire_repeated_use','Boundary revisits a coedge before closure')
                del remaining[u.edge];cycle.append(u);e=graph.edges[u.edge]
                end=e.vertices[1 if u.forward else 0]
                if end==first:break
                possible=[v for v in remaining.values() if graph.edges[v.edge].vertices[0 if v.forward else 1]==end]
                # Direction is determined after fitting, from each face interior and its normal.
                require(len(possible)==1,'face_sector_ambiguous','No unique next use in oriented face sector',face=f,vertex=end,candidates=[v.edge for v in possible])
                u=possible[0]
            else:require(False,'wire_not_closed','Face boundary does not close')
            closed.append(cycle)
        require(closed or not uses,'face_without_boundary','Recovered face has unclosed boundary')
        cycles[f]=closed
    graph.cycles=cycles
    return graph
