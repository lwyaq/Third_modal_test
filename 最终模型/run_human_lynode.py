"""
Entry point: run DvDHGNN on Human Lymph Node (CITE-seq spatial) dataset.

Dataset: data/human_lynode/
  - adata_RNA_with_annotation.h5ad
  - adata_ADT_with_annotation.h5ad

Usage:
    python -m modal_1.run_human_lynode
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import scanpy as sc
import torch
import anndata

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.preprocessing import pca, extract_coords, clustering
from modal_1.trainer import DHGNNTrainer
from modal_1.utils import evaluate_clustering, print_metrics, label_encode, setup_seed


def parse_args():
    p = argparse.ArgumentParser(description="DvDHGNN on Human Lymph Node (RNA + Protein)")

    # ============================================================
    # Data
    # ============================================================
    p.add_argument("--data_dir", type=str, default="data/human_lynode")
    p.add_argument("--rna_file", type=str, default="adata_RNA_with_annotation.h5ad")
    p.add_argument("--prot_file", type=str, default="adata_ADT_with_annotation.h5ad")
    p.add_argument("--label_col", type=str, default="LayerName")

    # ============================================================
    # Preprocessing
    # ============================================================
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--n_pca_rna", type=int, default=50)
    p.add_argument("--n_pca_prot", type=int, default=50)
    # Protein 原始只有 31 维，因此实际 PCA 后为 30 维。

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
    # Other
    # ============================================================
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  DvDHGNN on Human Lymph Node")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

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
    # Coordinates
    # ============================================================
    coords = extract_coords(adata_rna)

    # ============================================================
    # Labels
    # ============================================================
    label_col = args.label_col

    if label_col in adata_rna.obs.columns:
        raw_labels = adata_rna.obs[label_col].astype(str).values
        labels = label_encode(raw_labels)
        n_classes = len(np.unique(labels))
        print(f"\nLabels: {n_classes} classes from '{label_col}'")
    else:
        labels = None
        n_classes = 10
        print(f"\nNo label column '{label_col}' found; unsupervised mode (n_classes={n_classes}).")

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
    # Prepare modality data
    # ============================================================
    modality_data = [rna_features, prot_features]

    # ============================================================
    # Train model
    # ============================================================
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

    # ============================================================
    # Internal model prediction evaluation
    # ============================================================
    if labels is not None:
        print("\nEvaluating internal model predictions...")
        internal_predictions = trainer.get_predictions()
        internal_metrics = evaluate_clustering(labels, internal_predictions)
        print_metrics(internal_metrics, title="DvDHGNN Human Lymph Node (Internal Predictions)")

    # ============================================================
    # Final evaluation with mclust clustering
    # ============================================================
    embedding = trainer.get_embedding()

    adata = anndata.AnnData(obs=adata_rna.obs.copy())
    adata.obsm["DvDHGNN"] = embedding

    print("\nPerforming mclust clustering on final embedding...")

    clustering(
        adata,
        key="DvDHGNN",
        add_key="DvDHGNN_cluster",
        n_clusters=n_classes,
        method="mclust",
        random_state=args.seed,
    )

    predictions = adata.obs["DvDHGNN_cluster"].astype(int).values

    if labels is not None:
        final_metrics = evaluate_clustering(labels, predictions)

        for k in ("morans_i_cluster", "morans_i_embedding_mean"):
            if k in metrics:
                final_metrics[k] = metrics[k]

        print_metrics(final_metrics, title="DvDHGNN Human Lymph Node (mclust)")
    else:
        print("\n" + "=" * 50)
        print("  Unsupervised Evaluation (no labels)")
        print("=" * 50)
        for k in ("morans_i_cluster", "morans_i_embedding_mean"):
            if k in metrics:
                print(f"  {k:30s}: {metrics[k]:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()