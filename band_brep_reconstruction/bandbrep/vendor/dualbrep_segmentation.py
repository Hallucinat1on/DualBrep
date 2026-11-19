"""Single-pass connected components adapted from clustering.py."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def segment_faces(mesh, face_udf, threshold, filter_size, avg_udf_threshold):
    """Label face cores and record why other triangles were excluded."""
    udf = np.abs(face_udf[:, 0]) if face_udf.ndim == 2 else np.abs(face_udf)
    n_faces = len(udf)

    reasons = np.ones(n_faces, dtype=np.int8)  # 0 core, 1 threshold, 3 size, 4 mean
    split_parents = []
    report = dict(split_parents=split_parents, pass1_components=0, pass1_valid=0,
                  second_pass_enabled=False)
    # --- Pass 1: threshold masking + connected components ---
    active = udf >= threshold
    labels = np.full(n_faces, -1, dtype=np.int32)

    active_indices = np.where(active)[0]
    n_active = len(active_indices)
    if n_active == 0:
        return 0, labels, reasons, report

    adj = mesh.face_adjacency  # (E, 2)
    both_active = active[adj[:, 0]] & active[adj[:, 1]]
    active_adj = adj[both_active]

    remap = np.full(n_faces, -1, dtype=np.int32)
    remap[active_indices] = np.arange(n_active)

    if len(active_adj) > 0:
        rows = remap[active_adj[:, 0]]
        cols = remap[active_adj[:, 1]]
        data = np.ones(len(rows) * 2, dtype=np.int8)
        graph = csr_matrix((data, (np.r_[rows, cols], np.r_[cols, rows])),
                           shape=(n_active, n_active))
    else:
        graph = csr_matrix((n_active, n_active), dtype=np.int8)

    n_comp, comp_labels = connected_components(graph, directed=False)
    labels[active_indices] = comp_labels
    reasons[active_indices] = 0

    # Filter by size and average UDF
    cluster_sizes = np.bincount(comp_labels, minlength=n_comp)
    cluster_udf_sums = np.bincount(comp_labels, weights=udf[active_indices], minlength=n_comp)
    cluster_avg_udf = np.where(cluster_sizes > 0, cluster_udf_sums / cluster_sizes, 0.0)

    valid_clusters = np.where(
        (cluster_sizes >= filter_size) & (cluster_avg_udf > avg_udf_threshold)
    )[0]
    valid_mask = np.isin(labels, valid_clusters)
    reasons[active_indices[cluster_sizes[comp_labels] < filter_size]] = 3
    mean_rejected = (cluster_sizes[comp_labels] >= filter_size) & (cluster_avg_udf[comp_labels] <= avg_udf_threshold)
    reasons[active_indices[mean_rejected]] = 4
    labels[~valid_mask] = -1
    report.update(pass1_components=int(len(cluster_sizes)), pass1_valid=int(len(valid_clusters)))
    labels[valid_mask] = np.searchsorted(valid_clusters, labels[valid_mask])
    return len(valid_clusters), labels, reasons, report
