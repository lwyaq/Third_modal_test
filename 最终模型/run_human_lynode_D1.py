"""
Entry point: run DvDHGNN on Human Lymph Node D1 (RNA + Protein) dataset.

Dataset: data/human_lynode_D1/
  - adata_RNA.h5ad
  - adata_ADT.h5ad

The D1 dataset has no ground-truth labels. This script evaluates the learned
embedding with unsupervised metrics and saves both KMeans and mclust clustering
results.

Usage:
    python -m modal_1.run_human_lynode_D1
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.preprocessing import clustering, extract_coords, pca
from modal_1.trainer import DHGNNTrainer
from modal_1.utils import compute_morans_i, setup_seed


def parse_args():
    p = argparse.ArgumentParser(description="DvDHGNN on Human Lymph Node D1 (RNA + Protein)")

    # ============================================================
    # Data
    # ============================================================
    p.add_argument("--data_dir", type=str, default="data/human_lynode_D1")
    p.add_argument("--rna_file", type=str, default="adata_RNA.h5ad")
    p.add_argument("--prot_file", type=str, default="adata_ADT.h5ad")
    p.add_argument("--n_classes", type=int, default=11)
    p.add_argument("--output_dir", type=str, default="results/human_lynode_D1")

    # ============================================================
    # Preprocessing
    # ============================================================
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--n_pca_rna", type=int, default=64)
    p.add_argument("--n_pca_prot", type=int, default=32)

    # ============================================================
    # Model structure
    # ============================================================
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_feature_edges", type=int, default=60)
    p.add_argument("--k_nodes", type=int, default=15)
    p.add_argument("--k_edges", type=int, default=8)

    # ============================================================
    # Spatial hypergraph
    # ============================================================
    p.add_argument("--use_hsl_spatial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hsl_residual_strength", type=float, default=0.8)
    p.add_argument("--max_spatial_edges", type=int, default=3484)

    # ============================================================
    # Dynamic feature hypergraph
    # ============================================================
    p.add_argument("--use_dynamic_feature", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--edge_adjust_interval", type=int, default=10)
    p.add_argument("--delta_edges", type=int, default=50)
    p.add_argument("--beta_saturation", type=float, default=0.6)
    p.add_argument("--gamma_saturation", type=float, default=0.98)
    p.add_argument("--topk_edges", type=int, default=3)
    p.add_argument("--min_edges", type=int, default=100)
    p.add_argument("--allow_edge_add", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_edges_after_warmup", action=argparse.BooleanOptionalAction, default=True)

    # ============================================================
    # Training
    # ============================================================
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)

    # ============================================================
    # DEC
    # ============================================================
    p.add_argument("--warmup_epochs", type=int, default=60)
    p.add_argument("--dec_stability_patience", type=int, default=3)
    p.add_argument("--dec_stability_tol", type=float, default=0.005)
    p.add_argument("--dec_stability_min_epochs", type=int, default=20)

    # ============================================================
    # Loss
    # ============================================================
    p.add_argument("--lambda_recon", type=float, default=0.5)
    p.add_argument("--lambda_cluster", type=float, default=2.0)
    p.add_argument("--lambda_smooth", type=float, default=0.05)

    # ============================================================
    # Evaluation / Other
    # ============================================================
    p.add_argument("--morans_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


def evaluate_unsupervised(
    embedding: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    morans_k: int = 10,
) -> Dict[str, float]:
    """Evaluate clustering without ground-truth labels."""
    cluster_morans = [
        compute_morans_i(coords, (labels == cluster_id).astype(float), k=morans_k)
        for cluster_id in np.unique(labels)
    ]
    metrics = {
        "morans_i": float(np.mean(cluster_morans)),
        "silhouette": silhouette_score(embedding, labels),
        "dbi": davies_bouldin_score(embedding, labels),
    }
    return metrics


def print_unsupervised_metrics(metrics: Dict[str, float], title: str):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"  Moran's I           : {metrics['morans_i']:.4f}")
    print(f"  Silhouette          : {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['dbi']:.4f}")
    print("=" * 50 + "\n")


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  DvDHGNN on Human Lymph Node D1")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")

    # ============================================================
    # Load data
    # ============================================================
    rna_path = os.path.join(args.data_dir, args.rna_file)
    prot_path = os.path.join(args.data_dir, args.prot_file)

    print(f"\nLoading RNA data from {rna_path}...")
    adata_rna = sc.read_h5ad(rna_path)

    print(f"Loading Protein (ADT) data from {prot_path}...")
    adata_prot = sc.read_h5ad(prot_path)

    adata_rna.var_names_make_unique()
    adata_prot.var_names_make_unique()

    print("RNA:", adata_rna)
    print("Protein:", adata_prot)

    # ============================================================
    # Align RNA and Protein cells/spots
    # ============================================================
    if not np.array_equal(adata_rna.obs_names, adata_prot.obs_names):
        print("\nRNA and Protein obs_names are not aligned. Aligning by common obs_names...")
        common_obs = adata_rna.obs_names.intersection(adata_prot.obs_names)
        print(f"Common cells/spots: {len(common_obs)}")

        adata_rna = adata_rna[common_obs].copy()
        adata_prot = adata_prot[common_obs].copy()

    assert np.array_equal(adata_rna.obs_names, adata_prot.obs_names), \
        "RNA and Protein obs_names are not aligned!"

    # ============================================================
    # Preprocessing: RNA
    # ============================================================
    print("\nPreprocessing RNA...")

    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(
        adata_rna,
        flavor="seurat_v3",
        n_top_genes=args.n_top_genes,
        check_values=False,
    )
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)

    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    rna_features = pca(
        adata_rna_high,
        n_comps=args.n_pca_rna,
        random_state=args.seed,
    )

    # ============================================================
    # Preprocessing: Protein / ADT
    # ============================================================
    print("Preprocessing Protein (ADT)...")

    sc.pp.normalize_total(adata_prot, target_sum=1e4)
    sc.pp.log1p(adata_prot)
    sc.pp.scale(adata_prot)

    n_prot_pca = min(args.n_pca_prot, adata_prot.n_vars - 1)
    prot_features = pca(
        adata_prot,
        n_comps=n_prot_pca,
        random_state=args.seed,
    )

    # ============================================================
    # Coordinates and no-label setup
    # ============================================================
    coords = extract_coords(adata_prot)
    labels = None
    n_classes = args.n_classes
    print(f"\nNo ground-truth labels are used; unsupervised mode (n_classes={n_classes}).")

    # ============================================================
    # Print data summary
    # ============================================================
    print("\nData summary")
    print("-" * 70)
    print(f"RNA raw shape:        {adata_rna.shape}")
    print(f"Protein raw shape:    {adata_prot.shape}")
    print(f"RNA features:         {rna_features.shape}")
    print(f"Protein features:     {prot_features.shape}")
    print(f"Coordinates:          {coords.shape}")
    print(f"Nodes:                {coords.shape[0]}")
    print(f"Same obs_names:       {np.array_equal(adata_rna.obs_names, adata_prot.obs_names)}")
    print(f"RNA first 5:          {adata_rna.obs_names[:5].tolist()}")
    print(f"PROT first 5:         {adata_prot.obs_names[:5].tolist()}")
    print(f"Any NaN RNA:          {np.isnan(rna_features).any()}")
    print(f"Any NaN Protein:      {np.isnan(prot_features).any()}")
    print(f"Any NaN coords:       {np.isnan(coords).any()}")

    # ============================================================
    # Print parameter summary
    # ============================================================
    print("\nParameter summary")
    print("-" * 70)
    print(f"n_classes:                  {n_classes}")
    print(f"hidden_dim:                 {args.hidden_dim}")
    print(f"n_layers:                   {args.n_layers}")
    print(f"dropout:                    {args.dropout}")
    print(f"lr:                         {args.lr}")
    print(f"weight_decay:               {args.weight_decay}")
    print(f"epochs:                     {args.epochs}")
    print(f"patience:                   {args.patience}")
    print(f"warmup_epochs:              {args.warmup_epochs}")
    print(f"lambda_recon:               {args.lambda_recon}")
    print(f"lambda_cluster:             {args.lambda_cluster}")
    print(f"lambda_smooth:              {args.lambda_smooth}")
    print(f"topk_edges:                 {args.topk_edges}")
    print(f"edge_adjust_interval:       {args.edge_adjust_interval}")
    print(f"delta_edges:                {args.delta_edges}")
    print(f"beta_saturation:            {args.beta_saturation}")
    print(f"gamma_saturation:           {args.gamma_saturation}")
    print(f"min_edges:                  {args.min_edges}")
    print(f"hsl_residual_strength:      {args.hsl_residual_strength}")
    print(f"freeze_edges_after_warmup:  {args.freeze_edges_after_warmup}")

    # ============================================================
    # Prepare modality data and train model
    # ============================================================
    modality_data = [rna_features, prot_features]

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

    trainer.fit()
    embedding = trainer.get_embedding()

    # ============================================================
    # Final KMeans and mclust clustering on the learned embedding
    # ============================================================
    adata = anndata.AnnData(obs=adata_rna.obs.copy())
    adata.obsm["DvDHGNN"] = embedding
    adata.obsm["spatial"] = coords

    print("\nPerforming KMeans clustering on final embedding...")
    kmeans_labels = KMeans(
        n_clusters=n_classes,
        n_init=20,
        random_state=args.seed,
        max_iter=500,
    ).fit_predict(embedding)
    adata.obs["DvDHGNN_kmeans"] = pd.Categorical(kmeans_labels.astype(str))

    print("Performing mclust clustering on final embedding...")
    clustering(
        adata,
        key="DvDHGNN",
        add_key="DvDHGNN_mclust",
        n_clusters=n_classes,
        method="mclust",
        random_state=args.seed,
    )
    mclust_labels = adata.obs["DvDHGNN_mclust"].astype(int).values

    kmeans_metrics = evaluate_unsupervised(embedding, coords, kmeans_labels, args.morans_k)
    mclust_metrics = evaluate_unsupervised(embedding, coords, mclust_labels, args.morans_k)

    print_unsupervised_metrics(kmeans_metrics, "DvDHGNN Human Lymph Node D1 (KMeans)")
    print_unsupervised_metrics(mclust_metrics, "DvDHGNN Human Lymph Node D1 (mclust)")

    # ============================================================
    # Save outputs
    # ============================================================
    cluster_path = os.path.join(args.output_dir, "clusters.csv")
    metrics_path = os.path.join(args.output_dir, "unsupervised_metrics.csv")
    embedding_path = os.path.join(args.output_dir, "embedding.npy")
    adata_path = os.path.join(args.output_dir, "dvdhgnn_human_lynode_D1_results.h5ad")

    cluster_df = pd.DataFrame(
        {
            "obs_name": adata.obs_names,
            "DvDHGNN_kmeans": kmeans_labels.astype(int),
            "DvDHGNN_mclust": mclust_labels.astype(int),
        }
    )
    cluster_df.to_csv(cluster_path, index=False)

    metrics_df = pd.DataFrame(
        [
            {"method": "kmeans", **kmeans_metrics},
            {"method": "mclust", **mclust_metrics},
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    np.save(embedding_path, embedding)
    adata.write_h5ad(adata_path)

    print("\nSaved outputs")
    print("-" * 70)
    print(f"Clusters:  {cluster_path}")
    print(f"Metrics:   {metrics_path}")
    print(f"Embedding: {embedding_path}")
    print(f"AnnData:   {adata_path}")


if __name__ == "__main__":
    main()
