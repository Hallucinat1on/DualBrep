"""Graph components and face labels retained by the two-pass segmentation."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .schemas import require
from .surface_mesh import face_adjacency

def core_components(pairs,active):
    ids=np.flatnonzero(active);labels=np.full(len(active),-1,np.int32)
    if not len(ids):return labels
    remap=np.full(len(active),-1,np.int64);remap[ids]=np.arange(len(ids))
    pairs=pairs[active[pairs].all(axis=1)]
    graph=coo_matrix((np.ones(len(pairs)),(remap[pairs[:,0]],remap[pairs[:,1]])),shape=(len(ids),len(ids))).tocsr()
    _,local=connected_components(graph,directed=False);labels[ids]=local
    return labels

def extract_face_seeds(mesh,bands,provider,cfg):
    labels=bands.face_labels
    require(labels is not None,'missing_face_segmentation','Build two-pass bands before extracting seeds')
    require(np.any(labels>=0),'no_face_core','Two-pass segmentation retained no face core',
            segmentation=bands.report.get('segmentation'), exclusions=bands.report.get('exclusion_counts'))
    return labels,bands.report['segmentation']
