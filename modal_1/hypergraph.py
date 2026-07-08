"""
Hypergraph construction utilities for spatial multi-omics.

Provides:
  - Spatial hyperedges: Delaunay-star, multi-scale KNN
  - Feature hyperedges: DBSCAN with automatic eps
  - Incidence matrix assembly (multi-view, weighted)
  - Spectral clustering on the Zhou-normalized hypergraph Laplacian
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# --------------------------------------------------------------------------- #
#                          Spatial hyperedge builders                         #
# --------------------------------------------------------------------------- #

def delaunay_star_edges(
    coords: np.ndarray,
    s_min: int = 3,
    s_max: int = 80,
) -> List[np.ndarray]:
    """
    Build hyperedges from Delaunay triangulation.

    For each node, the hyperedge is the union of the node itself and all
    its Delaunay neighbors.  Hyperedges that are too small (<s_min) or
    too large (>s_max) are pruned.

    Parameters
    ----------
    coords : (N, 2) spatial coordinates
    s_min  : minimum hyperedge cardinality
    s_max  : maximum hyperedge cardinality

    Returns
    -------
    edges : list of 1-D int arrays (each array = one hyperedge)
    """
    n = coords.shape[0]
    coords2d = coords[:, :2] if coords.shape[1] > 2 else coords

    try:
        tri = Delaunay(coords2d)
    except Exception:
        warnings.warn("Delaunay failed, falling back to KNN star edges.")
        return multi_scale_knn_edges(coords, ks=[10, 20, 30], s_min=s_min, s_max=s_max)

    # Build adjacency from simplices
    adj: List[set] = [set() for _ in range(n)]
    for simplex in tri.simplices:
        for i in simplex:
            for j in simplex:
                if i != j:
                    adj[i].add(j)

    edges: List[np.ndarray] = []
    for node in range(n):
        nbrs = adj[node]
        star = np.array(sorted(nbrs | {node}), dtype=np.int64)
        if s_min <= len(star) <= s_max:
            edges.append(star)

    # Deduplicate (same set of nodes)
    seen = set()
    unique_edges = []
    for e in edges:
        key = tuple(e)
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges


def anchor_hypergraph(
    coords: np.ndarray,
    n_anchors: int = 150,
    cells_per_anchor: int = 20,
    s_min: int = 5,
    s_max: int = 100,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Anchor-based spatial hypergraph: region-centered instead of cell-centered.

    Instead of one hyperedge per cell (Delaunay-star, |E|~|V|),
    this places M anchor points via K-means and forms one spatial
    region hyperedge per anchor (|E|=M, M<<N).

    Each anchor's radius is set adaptively to its k-th nearest cell,
    with a hard cap at s_max cardinality to prevent overly large
    neighborhoods.

    Parameters
    ----------
    coords           : (N, 2) spatial coordinates
    n_anchors        : number of anchor points (K-means clusters)
    cells_per_anchor : target cells per anchor (for adaptive radius)
    s_min            : minimum hyperedge cardinality
    s_max            : maximum hyperedge cardinality (hard cap)
    seed             : random seed for K-means

    Returns
    -------
    edges   : list of 1-D int arrays (each array = one hyperedge)
    anchors : (M, 2) anchor point coordinates
    """
    n = coords.shape[0]
    coords2d = coords[:, :2] if coords.shape[1] > 2 else coords

    # Place anchors via K-means
    actual_anchors = min(n_anchors, n)
    km = KMeans(n_clusters=actual_anchors, n_init=10, random_state=seed)
    km.fit(coords2d)
    anchor_centers = km.cluster_centers_

    # Adaptive radius: for each anchor, find distance to k-th nearest cell
    k = min(cells_per_anchor, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords2d)
    _, knn_indices = nn.kneighbors(anchor_centers)

    # Build hyperedges using KNN (exact k neighbors per anchor)
    edges: List[np.ndarray] = []
    for i in range(len(anchor_centers)):
        members = knn_indices[i, 1:]  # exclude the anchor itself if it's a data point
        if len(members) > s_max:
            members = members[:s_max]
        if len(members) >= s_min:
            edges.append(np.sort(members).astype(np.int64))

    # Deduplicate
    seen = set()
    unique_edges = []
    for e in edges:
        key = tuple(e)
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    print(f"  Anchor hypergraph: {actual_anchors} anchors, "
          f"{len(unique_edges)} hyperedges, "
          f"avg cardinality={np.mean([len(e) for e in unique_edges]):.1f}")

    return unique_edges, anchor_centers


def multi_scale_knn_edges(
    coords: np.ndarray,
    ks: Sequence[int] = (6, 12, 24, 48),
    s_min: int = 3,
    s_max: int = 200,
) -> List[np.ndarray]:
    """
    Build multi-scale KNN hyperedges from spatial coordinates.

    For each k in *ks* and each node, the k nearest neighbors (plus the
    node itself) form one hyperedge.

    Parameters
    ----------
    coords : (N, 2) or (N, d) coordinates / embeddings
    ks     : sequence of neighbor counts
    s_min  : min cardinality
    s_max  : max cardinality

    Returns
    -------
    edges : list of 1-D int arrays
    """
    n = coords.shape[0]
    max_k = min(max(ks), n - 1)
    nn = NearestNeighbors(n_neighbors=min(max_k + 1, n), n_jobs=-1)
    nn.fit(coords)

    edges: List[np.ndarray] = []
    seen = set()
    for k in ks:
        k = min(k, n - 1)
        if k < 1:
            continue
        _, indices = nn.kneighbors(coords, n_neighbors=k + 1)
        for row in indices:
            e = np.sort(row)
            if s_min <= len(e) <= s_max:
                key = tuple(e)
                if key not in seen:
                    seen.add(key)
                    edges.append(e.astype(np.int64))
    return edges


def radius_edges(
    coords: np.ndarray,
    radii: Sequence[float] = (50, 100, 150, 200),
    s_min: int = 3,
    s_max: int = 300,
) -> List[np.ndarray]:
    """
    Build radius-based hyperedges.  For each radius and each node,
    all neighbors within that radius form a hyperedge.
    """
    n = coords.shape[0]
    tree = cKDTree(coords[:, :2] if coords.shape[1] > 2 else coords)
    edges: List[np.ndarray] = []
    seen = set()
    for r in radii:
        nbrs_list = tree.query_ball_point(coords[:, :2] if coords.shape[1] > 2 else coords, r=r)
        for idx, nbrs in enumerate(nbrs_list):
            e = np.sort(np.array(nbrs, dtype=np.int64))
            if s_min <= len(e) <= s_max:
                key = tuple(e)
                if key not in seen:
                    seen.add(key)
                    edges.append(e)
    return edges


# --------------------------------------------------------------------------- #
#                         Feature hyperedge builders                          #
# --------------------------------------------------------------------------- #

def kmeans_star_edges(
    X: np.ndarray,
    n_clusters: int = 50,
    k_neighbors: int = 30,
    s_min: int = 5,
    s_max: int = 200,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Build feature hyperedges via K-means clustering + star topology.

    For each K-means cluster, the hyperedge is the union of:
      - The cluster center's nearest neighbor in the data
      - The k nearest data points to the cluster center

    This creates one hyperedge per cluster, with controlled cardinality.

    Parameters
    ----------
    X           : (N, d) feature matrix
    n_clusters  : number of K-means clusters (= number of hyperedges)
    k_neighbors : max members per hyperedge
    s_min/s_max : cardinality bounds
    seed        : random seed

    Returns
    -------
    edges : list of 1-D int arrays (each = one hyperedge)
    """
    n = X.shape[0]
    k = min(n_clusters, n)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=100)
    km.fit(X)

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n), n_jobs=-1)
    nn.fit(X)
    _, knn_indices = nn.kneighbors(km.cluster_centers_)

    edges: List[np.ndarray] = []
    seen = set()
    for i in range(k):
        members = np.sort(knn_indices[i]).astype(np.int64)
        if s_min <= len(members) <= s_max:
            key = tuple(members)
            if key not in seen:
                seen.add(key)
                edges.append(members)

    return edges


def dbscan_edges(
    X: np.ndarray,
    min_samples: int = 10,
    s_min: int = 5,
    s_max: int = 600,
    eps: Optional[float] = None,
) -> Tuple[List[np.ndarray], float, np.ndarray]:
    """
    Build hyperedges from DBSCAN clustering on feature space.

    Each DBSCAN cluster becomes a hyperedge.  Noise points (label = -1)
    are assigned to the nearest cluster hyperedge.

    Parameters
    ----------
    X           : (N, d) feature matrix
    min_samples : DBSCAN min_samples
    s_min/s_max : cardinality bounds
    eps         : DBSCAN eps (auto-computed from k-distance knee if None)

    Returns
    -------
    edges    : list of hyperedge index arrays
    eps_used : the eps value that was used
    labels   : (N,) DBSCAN cluster labels
    """
    from sklearn.cluster import DBSCAN

    if eps is None:
        eps = _auto_eps_knee(X, k=min_samples)

    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)

    # Assign noise to nearest cluster
    cluster_ids = np.unique(labels[labels >= 0])
    if len(cluster_ids) == 0:
        # All noise — fall back to KNN edges
        warnings.warn("DBSCAN found 0 clusters; falling back to KNN feature edges.")
        return multi_scale_knn_edges(X, ks=[15, 30], s_min=s_min, s_max=s_max), eps, labels

    if np.any(labels < 0):
        noise_mask = labels < 0
        cent_rep = np.array([X[labels == c].mean(axis=0) for c in cluster_ids])
        nn = NearestNeighbors(n_neighbors=1).fit(cent_rep)
        _, near = nn.kneighbors(X[noise_mask])
        labels[noise_mask] = cluster_ids[near.ravel()]

    edges: List[np.ndarray] = []
    for c in cluster_ids:
        members = np.where(labels == c)[0]
        if s_min <= len(members) <= s_max:
            edges.append(members.astype(np.int64))

    return edges, eps, labels


def knn_feature_edges(
    X: np.ndarray,
    ks: Sequence[int] = (10, 20, 40),
    s_min: int = 5,
    s_max: int = 500,
) -> List[np.ndarray]:
    """KNN-based feature hyperedges (alternative to DBSCAN)."""
    return multi_scale_knn_edges(X, ks=ks, s_min=s_min, s_max=s_max)


# --------------------------------------------------------------------------- #
#              Expression-weighted incidence (CoMem-inspired)                 #
# --------------------------------------------------------------------------- #

def compute_expression_weighted_incidence(
    H: sp.csr_matrix,
    features: np.ndarray,
) -> sp.csr_matrix:
    """
    Weight spatial hyperedges by expression cosine similarity.

    Inspired by the CoMem-DIPHW paper (He et al., 2026): instead of
    binary incidence, each cell's weight in a spatial hyperedge is
    determined by how well its expression profile matches the mean
    expression of cells in that hyperedge.

    Parameters
    ----------
    H        : (N, E) binary sparse incidence matrix
    features : (N, d) expression features (e.g., PCA/LSI)

    Returns
    -------
    H_weighted : (N, E) sparse incidence with cosine similarity weights
    """
    H_csc = H.tocsc()
    n_cells, n_edges = H_csc.shape

    # L2-normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_norm = features / norms

    rows, cols, vals = [], [], []

    for e in range(n_edges):
        start, end = H_csc.indptr[e], H_csc.indptr[e + 1]
        members = H_csc.indices[start:end]
        if len(members) < 2:
            continue

        # Mean expression of cells in this hyperedge
        mean_expr = X_norm[members].mean(axis=0)
        mean_norm = np.linalg.norm(mean_expr)
        if mean_norm < 1e-8:
            continue
        mean_expr /= mean_norm

        # Cosine similarity: cell expression vs hyperedge mean
        sims = X_norm[members] @ mean_expr  # (n_members,)
        sims = np.clip(sims, 0.0, 1.0)

        for idx, node in enumerate(members):
            if sims[idx] > 0.01:
                rows.append(int(node))
                cols.append(int(e))
                vals.append(float(sims[idx]))

    H_weighted = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n_cells, n_edges), dtype=np.float32
    )
    return H_weighted


def _auto_eps_knee(X: np.ndarray, k: int = 10) -> float:
    """Estimate DBSCAN eps via the k-distance knee heuristic."""
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    k_dists = np.sort(dists[:, -1])  # k-th neighbor distances sorted ascending
    # Knee: largest gap in sorted distances
    diffs = np.diff(k_dists)
    knee_idx = np.argmax(diffs)
    eps = float(k_dists[knee_idx])
    return max(eps, 0.1)


# --------------------------------------------------------------------------- #
#                       Incidence matrix construction                         #
# --------------------------------------------------------------------------- #

def build_incidence(
    n: int,
    edge_lists: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Build a weighted incidence matrix H from multiple hyperedge views.

    Parameters
    ----------
    n          : number of nodes
    edge_lists : list of edge-lists, one per view
    weights    : per-view scalar weights (broadcast to hyperedges)

    Returns
    -------
    H     : (n, E) sparse incidence matrix  (csr)
    w     : (E,) hyperedge weights
    cards : (E,) hyperedge cardinalities
    edges : concatenated flat list of all hyperedge arrays
    """
    if weights is None:
        weights = [1.0] * len(edge_lists)

    rows, cols, vals = [], [], []
    w_list, card_list, all_edges = [], [], []
    col_offset = 0

    for view_idx, elist in enumerate(edge_lists):
        vw = weights[view_idx]
        for eidx, edge in enumerate(elist):
            c = col_offset + eidx
            for node in edge:
                rows.append(int(node))
                cols.append(c)
                vals.append(1.0)
            w_list.append(vw)
            card_list.append(len(edge))
            all_edges.append(edge)
        col_offset += len(elist)

    H = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n, col_offset),
        dtype=np.float32,
    )
    w = np.array(w_list, dtype=np.float32)
    cards = np.array(card_list, dtype=np.int32)
    return H, w, cards, all_edges


# --------------------------------------------------------------------------- #
#                      Hypergraph spectral clustering                         #
# --------------------------------------------------------------------------- #

def _zhou_laplacian(
    H: sp.csr_matrix,
    w: np.ndarray,
    cards: np.ndarray,
) -> sp.csr_matrix:
    """
    Compute the Zhou et al. (2006) normalized hypergraph Laplacian:
        L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}

    Parameters
    ----------
    H     : (n, E) incidence matrix
    w     : (E,) hyperedge weights
    cards : (E,) cardinalities (used for De)

    Returns
    -------
    L : (n, n) sparse Laplacian
    """
    n = H.shape[0]
    H_dense = H  # keep sparse for efficiency

    # Node degree: Dv = H @ w  (sum of weights of incident hyperedges)
    Dv = np.asarray(H.dot(w)).ravel()
    Dv = np.maximum(Dv, 1e-10)

    # Hyperedge degree: De = cardinality
    De = np.maximum(cards.astype(np.float64), 1.0)

    # Build diagonal matrices
    Dv_inv_sqrt = sp.diags(1.0 / np.sqrt(Dv))
    De_inv = sp.diags(1.0 / De)
    W_diag = sp.diags(w.astype(np.float64))

    # L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
    I = sp.eye(n, format="csr")
    Theta = Dv_inv_sqrt @ H @ W_diag @ De_inv @ H.T @ Dv_inv_sqrt
    L = I - Theta
    return L


def spectral_cluster_from_hypergraph(
    H: sp.csr_matrix,
    w: np.ndarray,
    cards: np.ndarray,
    n_clusters: Optional[int] = None,
    r_max: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Spectral clustering on the Zhou-normalized hypergraph Laplacian.

    If n_clusters is None, it is determined automatically via the
    eigengap heuristic (searching up to r_max eigenvalues).

    Returns
    -------
    labels : (n,) cluster assignments
    k_star : chosen number of clusters
    eigvals: sorted eigenvalues used for the gap analysis
    """
    L = _zhou_laplacian(H, w, cards)

    # Compute smallest eigenvalues
    r = min(r_max, L.shape[0] - 2)
    try:
        eigvals, eigvecs = eigsh(L.astype(np.float64), k=r + 1, which="SM", tol=1e-6)
    except Exception:
        # Fallback to dense eigendecomposition
        Ld = L.toarray() if sp.issparse(L) else L
        eigvals_all, eigvecs_all = np.linalg.eigh(Ld)
        eigvals = eigvals_all[: r + 1]
        eigvecs = eigvecs_all[:, : r + 1]

    # Sort
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if n_clusters is None:
        # Eigengap heuristic
        gaps = np.diff(eigvals[1:])  # skip the trivial 0 eigenvalue
        k_star = int(np.argmax(gaps) + 2)  # +2 because we skipped index 0
        k_star = max(2, min(k_star, r_max))
    else:
        k_star = n_clusters

    # K-means on the first k_star eigenvectors
    U = eigvecs[:, :k_star]
    # Normalize rows
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)
    U_norm = U / row_norms

    km = KMeans(n_clusters=k_star, n_init=20, random_state=seed, max_iter=500)
    labels = km.fit_predict(U_norm)

    return labels, k_star, eigvals


def adaptive_spectral_cluster(
    H: sp.csr_matrix,
    w: np.ndarray,
    cards: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    n_init: int = 30,
) -> np.ndarray:
    """
    Spectral clustering with a fixed number of clusters and multiple
    random restarts for better stability.
    """
    L = _zhou_laplacian(H, w, cards)

    r = min(n_clusters + 5, L.shape[0] - 2)
    try:
        eigvals, eigvecs = eigsh(L.astype(np.float64), k=r, which="SM", tol=1e-6)
    except Exception:
        Ld = L.toarray() if sp.issparse(L) else L
        eigvals_all, eigvecs_all = np.linalg.eigh(Ld)
        eigvals = eigvals_all[:r]
        eigvecs = eigvecs_all[:, :r]

    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]

    U = eigvecs[:, :n_clusters]
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)
    U_norm = U / row_norms

    best_ari = -1
    best_labels = None
    for _ in range(n_init):
        km = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed + _, max_iter=500)
        labels = km.fit_predict(U_norm)
        # Use silhouette-like internal score for selection when no ground truth
        from sklearn.metrics import silhouette_score
        try:
            score = silhouette_score(U_norm, labels)
        except Exception:
            score = 0
        if score > best_ari:
            best_ari = score
            best_labels = labels

    return best_labels if best_labels is not None else labels
