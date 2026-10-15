"""Copied from repository clustering.py::hierarchical_segmentation.

Local standalone copy; only rejection provenance/report outputs were added.
Label decisions match the original two-pass function.
"""
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def hierarchical_segmentation(mesh, face_udf, threshold1=0.003, threshold2=0.01,
                              filter_size=5, avg_udf_threshold=0.003):
    """
    Two-pass hierarchical segmentation mirroring C++ segment_mode3.

    Pass 1: Mask faces with |UDF| < threshold1, region-grow connected components,
            filter small clusters and those with low average UDF.
    Pass 2: For each surviving cluster whose max UDF >= threshold2, attempt to
            split it by re-growing only through faces with UDF >= threshold2.
            If a cluster splits into >= 2 valid sub-clusters, replace it;
            otherwise keep the original.

    Parameters:
    - mesh: A trimesh.Trimesh object.
    - face_udf: np.ndarray of shape (n_faces,) or (n_faces, D); column 0 used if 2D.
    - threshold1: Float. First-pass mask threshold.
    - threshold2: Float. Second-pass split threshold.
    - filter_size: Int. Minimum cluster size.
    - avg_udf_threshold: Float. Minimum average UDF for a cluster to survive.

    Returns:
    - n_clusters: int, number of clusters found.
    - labels: np.ndarray of shape (n_faces,), cluster id per face (-1 if masked).
    """
    udf = np.abs(face_udf[:, 0]) if face_udf.ndim == 2 else np.abs(face_udf)
    n_faces = len(udf)

    reasons = np.ones(n_faces, dtype=np.int8)  # 0 core, 1 first threshold, 2 split threshold, 3 size, 4 mean, 5 sub-size
    split_parents = []
    # --- Pass 1: threshold masking + connected components ---
    active = udf >= threshold1
    labels = np.full(n_faces, -1, dtype=np.int32)

    active_indices = np.where(active)[0]
    n_active = len(active_indices)
    if n_active == 0:
        return 0, labels, reasons, dict(split_parents=[], pass1_components=0, pass1_valid=0)

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
    # Pass-1 labels retain original comp_labels ids (sparse); relabel once after pass 2.
    valid_set = set(valid_clusters.tolist())

    # --- Pass 2: split clusters with high max UDF using threshold2 ---
    # Compute per-cluster max UDF using original (sparse) cluster ids
    cluster_max_udf = np.full(n_comp, 0.0)
    np.maximum.at(cluster_max_udf, labels[labels >= 0], udf[labels >= 0])

    # Determine which valid clusters need splitting
    needs_split = np.zeros(n_comp, dtype=bool)
    for cid in valid_clusters:
        needs_split[cid] = cluster_max_udf[cid] >= threshold2

    # Faces eligible for pass-2: in a split-candidate cluster AND udf >= threshold2
    high_udf = udf >= threshold2
    labels_clipped = labels.clip(min=0)
    in_split_cluster = (labels >= 0) & needs_split[labels_clipped]
    eligible = in_split_cluster & high_udf

    eligible_indices = np.where(eligible)[0]
    n_eligible = len(eligible_indices)

    if n_eligible > 0:
        # Build sparse graph: both endpoints eligible AND in same pass-1 cluster
        both_eligible = eligible[adj[:, 0]] & eligible[adj[:, 1]]
        same_cluster = labels[adj[:, 0]] == labels[adj[:, 1]]
        p2_edges = adj[both_eligible & same_cluster]

        remap2 = np.full(n_faces, -1, dtype=np.int32)
        remap2[eligible_indices] = np.arange(n_eligible)

        if len(p2_edges) > 0:
            r2 = remap2[p2_edges[:, 0]]
            c2 = remap2[p2_edges[:, 1]]
            d2 = np.ones(len(r2) * 2, dtype=np.int8)
            g2 = csr_matrix((d2, (np.r_[r2, c2], np.r_[c2, r2])),
                            shape=(n_eligible, n_eligible))
        else:
            g2 = csr_matrix((n_eligible, n_eligible), dtype=np.int8)

        n_sub, sub_labels = connected_components(g2, directed=False)

        # Map each sub-component to its parent pass-1 cluster
        p1_of_eligible = labels[eligible_indices]
        sub_to_p1 = np.full(n_sub, -1, dtype=np.int32)
        sub_to_p1[sub_labels] = p1_of_eligible

        # Filter sub-clusters by size
        sub_sizes = np.bincount(sub_labels, minlength=n_sub)
        sub_valid = sub_sizes >= filter_size

        # Count valid sub-clusters per pass-1 cluster
        valid_per_p1 = defaultdict(int)
        for sid in range(n_sub):
            if sub_valid[sid]:
                valid_per_p1[sub_to_p1[sid]] += 1

        # For clusters that split into >= 2 valid sub-clusters,
        # replace pass-1 label with sub-cluster labels
        for cid in valid_clusters:
            if not needs_split[cid] or valid_per_p1.get(cid, 0) < 2:
                continue
            # Clear the original cluster labels; sub-clusters will overwrite
            parent_nodes = labels == cid
            reasons[parent_nodes & ~high_udf] = 2
            reasons[parent_nodes & high_udf] = 5
            split_parents.append(int(cid))
            labels[parent_nodes] = -1
            cid_mask = p1_of_eligible == cid
            cid_sub_ids = np.unique(sub_labels[cid_mask])
            for sid in cid_sub_ids:
                if not sub_valid[sid]:
                    continue
                # Assign a new unique label (above any existing comp_label)
                new_id = n_comp
                n_comp += 1
                face_idx = eligible_indices[sub_labels == sid]
                labels[face_idx] = new_id
                reasons[face_idx] = 0

    # --- Final relabel to continuous ids ---
    unique_labels = np.unique(labels[labels >= 0])
    relabel_map = np.full(labels.max() + 1 if labels.max() >= 0 else 1, -1, dtype=np.int64)
    relabel_map[unique_labels] = np.arange(len(unique_labels))
    valid_pos = labels >= 0
    labels[valid_pos] = relabel_map[labels[valid_pos]]
    n_clusters = len(unique_labels)

    return n_clusters, labels, reasons, dict(split_parents=split_parents, pass1_components=int(len(cluster_sizes)), pass1_valid=int(len(valid_clusters)))
