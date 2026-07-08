"""
Utility functions for evaluation, visualization, and preprocessing.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
    silhouette_score,
)


def evaluate_clustering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of clustering metrics.

    Returns
    -------
    metrics : dict with keys ari, nmi, ami, homogeneity, completeness, v_measure
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    return {
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"),
        "ami": adjusted_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred),
    }


def print_metrics(metrics: Dict[str, float], title: str = "Clustering Metrics"):
    """Pretty-print clustering metrics."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
    print(f"{'='*50}\n")


def compute_morans_i(
    coords: np.ndarray,
    values: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Parameters
    ----------
    coords : (N, 2) spatial coordinates
    values : (N,) values to test
    k      : number of neighbors for spatial weight matrix
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(values)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    # Build sparse weight matrix
    rows, cols, data = [], [], []
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    import scipy.sparse as sp
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W + W.T
    W.data[:] = 1.0
    S0 = W.sum()

    z = values - values.mean()
    Wz = W.dot(z)
    I = (n / S0) * (z @ Wz) / (z @ z + 1e-10)
    return float(I)


def label_encode(labels) -> np.ndarray:
    """Encode string/categorical labels as integers."""
    unique = sorted(set(labels))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[v] for v in labels], dtype=np.int64)


def setup_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
