"""
modal_1 — Dynamic Dual-View Hypergraph Neural Network
for Spatial Multi-Omics Clustering.

Provides:
  - hypergraph: spatial & feature hypergraph construction + spectral clustering
  - networks:   learnable dual-view HGNN with HSL-inspired structure refinement
  - model:      high-level model wrapper
  - trainer:    training loop with early stopping
  - utils:      evaluation metrics and helpers
"""

from modal_1.hypergraph import (
    delaunay_star_edges,
    multi_scale_knn_edges,
    dbscan_edges,
    build_incidence,
    spectral_cluster_from_hypergraph,
)
from modal_1.networks import DualBranchDHGNN
from modal_1.trainer import DHGNNTrainer

__all__ = [
    "delaunay_star_edges",
    "multi_scale_knn_edges",
    "dbscan_edges",
    "build_incidence",
    "spectral_cluster_from_hypergraph",
    "DualBranchDHGNN",
    "DHGNNTrainer",
]
