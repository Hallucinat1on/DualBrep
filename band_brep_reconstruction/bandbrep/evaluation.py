"""Optional offline scoring. This is the ONLY reconstruction module reading topology.json."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def score_vertices(graph,target_points,tolerance):
    source=np.array([v.position for v in graph.vertices]).reshape(-1,3);target=np.asarray(target_points).reshape(-1,3)
    if not len(source) or not len(target):
        return {'predicted':len(source),'target':len(target),'matched':0,'empty_agreement':len(source)==len(target)}
    dist=np.linalg.norm(source[:,None]-target[None],axis=2);rows,cols=linear_sum_assignment(dist)
    good=dist[rows,cols]<=tolerance
    return {'predicted':len(source),'target':len(target),'matched':int(good.sum()),'matched_distances':dist[rows[good],cols[good]].tolist(),
            'mapping':{int(r):int(c) for r,c in zip(rows[good],cols[good])}}

def score_counts(graph,sample_dir):
    # Counts are diagnostics, never sufficient to certify topology correctness.
    topology=json.loads((Path(sample_dir)/'topology.json').read_text())
    counts={'faces':len(topology['source_faces']),'edges':len(topology['edges']),'vertices':len(topology['vertices'])}
    prediction={'faces':graph.face_count,'edges':len(graph.edges),'vertices':len(graph.vertices)}
    return {'prediction':prediction,'target':counts,'count_agreement':prediction==counts,'topology_correctness_certified':False}

def match_face_regions(predicted_labels,target_labels):
    """Paired evaluation positions only; accepts permutation and reports splits/merges."""
    pred=np.asarray(predicted_labels);truth=np.asarray(target_labels);p=np.unique(pred[pred>=0]);t=np.unique(truth[truth>=0])
    table=np.zeros((len(p),len(t)),int)
    for i,a in enumerate(p):
        for j,b in enumerate(t):table[i,j]=np.sum((pred==a)&(truth==b))
    r,c=linear_sum_assignment(-table)
    return {'mapping':{int(p[i]):int(t[j]) for i,j in zip(r,c)},'confusion':table.tolist(),
            'missing_targets':[int(t[j]) for j in range(len(t)) if table[:,j].sum()==0],
            'merged_predictions':[int(p[i]) for i in range(len(p)) if np.sum(table[i]>0)>1],
            'split_targets':[int(t[j]) for j in range(len(t)) if np.sum(table[:,j]>0)>1]}
