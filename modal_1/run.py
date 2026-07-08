"""
Entry point: run the DvDHGNN model on E18.5 mouse brain spatial multi-omics data.

Usage:
    python -m modal_1.run [--epochs 500 --lr 0.001 --seed 42]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import scanpy as sc
import torch

warnings.filterwarnings("ignore")

# Ensure modal_1 is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.preprocessing import pca, lsi, extract_coords
from modal_1.trainer import DHGNNTrainer
from modal_1.utils import evaluate_clustering, print_metrics, label_encode, setup_seed


def parse_args():
    p = argparse.ArgumentParser(description="DvDHGNN: Dual-View Dynamic Hypergraph NN")
    p.add_argument("--data_dir", type=str, default="data/E18.5_mouse_brain")
    p.add_argument("--rna_file", type=str, default="E18_adata_rna.h5ad")
    p.add_argument("--atac_file", type=str, default="E18_adata_atac.h5ad")
    p.add_argument("--label_col", type=str, default="Combined_Clusters_annotation")

    p.add_argument("--n_pca", type=int, default=50)
    p.add_argument("--n_lsi", type=int, default=51)
    p.add_argument("--n_top_genes", type=int, default=3000)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_feature_edges", type=int, default=60)
    p.add_argument("--k_nodes", type=int, default=15)
    p.add_argument("--k_edges", type=int, default=8)
    p.add_argument("--use_hsl_spatial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_dynamic_feature", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep dynamic prototype feature hypergraphs enabled; disabling the removed static gene-as-hyperedge path is unsupported")
    p.add_argument("--edge_adjust_interval", type=int, default=10)
    p.add_argument("--delta_edges", type=int, default=20)
    p.add_argument("--beta_saturation", type=float, default=0.90)
    p.add_argument("--gamma_saturation", type=float, default=0.98)
    p.add_argument("--topk_edges", type=int, default=3)
    p.add_argument("--min_edges", type=int, default=100)
    p.add_argument("--hsl_residual_strength", type=float, default=0.5)
    p.add_argument("--allow_edge_add", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_edges_after_warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=80)
    p.add_argument("--dec_stability_patience", type=int, default=3,
                   help="Enable DEC assignment stability stopping after this many stable checks; 0 disables it")
    p.add_argument("--dec_stability_tol", type=float, default=0.005,
                   help="Maximum fraction of changed DEC assignments considered stable")
    p.add_argument("--dec_stability_min_epochs", type=int, default=20,
                   help="Minimum number of DEC epochs before stability stopping can trigger")

    p.add_argument("--lambda_recon", type=float, default=0.5)
    p.add_argument("--lambda_cluster", type=float, default=1.0)
    p.add_argument("--lambda_smooth", type=float, default=0.001)
    p.add_argument("--max_spatial_edges", type=int, default=2129)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # --- Load data ---
    rna_path = os.path.join(args.data_dir, args.rna_file)
    atac_path = os.path.join(args.data_dir, args.atac_file)

    print(f"\nLoading RNA data from {rna_path}...")
    adata_rna = sc.read_h5ad(rna_path)

    print(f"Loading ATAC data from {atac_path}...")
    adata_atac = sc.read_h5ad(atac_path)

    # --- Preprocessing: RNA ---
    print("\nPreprocessing RNA...")
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=args.n_top_genes, check_values=False)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    rna_features = pca(adata_rna_high, n_comps=args.n_pca, random_state=args.seed)

    # --- Preprocessing: ATAC ---
    print("Preprocessing ATAC...")
    adata_atac = adata_atac[adata_rna.obs_names].copy()
    if "X_lsi" not in adata_atac.obsm.keys():
        sc.pp.highly_variable_genes(adata_atac, flavor="seurat_v3", n_top_genes=args.n_top_genes, check_values=False)
        lsi(adata_atac, use_highly_variable=False, n_components=args.n_lsi)
    atac_features = adata_atac.obsm["X_lsi"].copy()

    # --- Coordinates ---
    coords = extract_coords(adata_atac)

    # --- Labels ---
    label_col = args.label_col
    if label_col in adata_atac.obs.columns:
        raw_labels = adata_atac.obs[label_col].astype(str).values
        labels = label_encode(raw_labels)
        n_classes = len(np.unique(labels))
        print(f"\nLabels: {n_classes} classes from '{label_col}'")
    else:
        labels = None
        n_classes = 14
        print(f"\nNo label column '{label_col}' found; unsupervised mode.")

    print(f"RNA features: {rna_features.shape}")
    print(f"ATAC features: {atac_features.shape}")
    print(f"Coordinates: {coords.shape}")
    print(f"Nodes: {coords.shape[0]}")

    # --- Prepare modality data ---
    modality_data = [rna_features, atac_features]

    # --- Train ---
    trainer = DHGNNTrainer(
        coords=coords,
        modality_data=modality_data,
        labels=labels,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_feature_edges=args.n_feature_edges,
        k_nodes=args.k_nodes,
        k_edges=args.k_edges,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        dec_stability_patience=args.dec_stability_patience,
        dec_stability_tol=args.dec_stability_tol,
        dec_stability_min_epochs=args.dec_stability_min_epochs,
        seed=args.seed,
        device=device,
        lambda_cluster=args.lambda_cluster,
        lambda_smooth=args.lambda_smooth,
        lambda_recon=args.lambda_recon,
        max_spatial_edges=args.max_spatial_edges,
        use_hsl_spatial=args.use_hsl_spatial,
        use_dynamic_feature=args.use_dynamic_feature,
        edge_adjust_interval=args.edge_adjust_interval,
        delta_edges=args.delta_edges,
        beta_saturation=args.beta_saturation,
        gamma_saturation=args.gamma_saturation,
        topk_edges=args.topk_edges,
        min_edges=args.min_edges,
        max_edges=coords.shape[0],
        hsl_residual_strength=args.hsl_residual_strength,
        allow_edge_add=args.allow_edge_add,
        freeze_edges_after_warmup=args.freeze_edges_after_warmup,
    )

    metrics = trainer.fit()

    # --- Final evaluation ---
    if labels is not None:
        predictions = trainer.get_predictions()
        final_metrics = evaluate_clustering(labels, predictions)
        print_metrics(final_metrics, title="DvDHGNN Clustering (All Cells)")


if __name__ == "__main__":
    main()
