"""
Parameter sweep runner for HSL-DvDHGNN on E18.5 mouse brain data.

This script loads/preprocesses RNA + ATAC once, then trains multiple
DHGNNTrainer configurations. It writes one CSV row per run so long sweeps can
be resumed or inspected even if a later configuration fails.

Example:
    python modal_1/param_sweep.py ^
      --warmup_epochs 120 ^
      --lambda_recon 0.5,0.4 ^
      --lambda_cluster 3.0,2.0 ^
      --hsl_residual_strength 0.3,0.5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback
import warnings
from typing import Iterable, List

import numpy as np
import scanpy as sc
import torch

warnings.filterwarnings("ignore")

# Ensure modal_1 is importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.preprocessing import extract_coords, lsi, pca
from modal_1.trainer import DHGNNTrainer
from modal_1.utils import label_encode, setup_seed


def _parse_list(value: str, cast):
    """Parse comma-separated CLI values, e.g. '0.5,0.4'."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [cast(v.strip()) for v in str(value).split(",") if v.strip() != ""]


def _bool_to_cli(value: bool) -> str:
    return "true" if value else "false"


def parse_args():
    p = argparse.ArgumentParser(description="Grid-search HSL-DvDHGNN parameters")

    # Data and preprocessing.
    p.add_argument("--data_dir", type=str, default="data/E18.5_mouse_brain")
    p.add_argument("--rna_file", type=str, default="E18_adata_rna.h5ad")
    p.add_argument("--atac_file", type=str, default="E18_adata_atac.h5ad")
    p.add_argument("--label_col", type=str, default="Combined_Clusters_annotation")
    p.add_argument("--n_pca", type=int, default=50)
    p.add_argument("--n_lsi", type=int, default=51)
    p.add_argument("--n_top_genes", type=int, default=3000)

    # Fixed model/training defaults.
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_spatial_edges", type=int, default=1500)
    p.add_argument("--min_edges", type=int, default=100)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_csv", type=str, default="sweep_results.csv")
    p.add_argument("--max_runs", type=int, default=0, help="0 means run the full grid")
    p.add_argument("--dry_run", action="store_true", help="Print the grid without training")

    # Boolean switches.
    p.add_argument("--use_hsl_spatial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_dynamic_feature", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow_edge_add", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_edges_after_warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_hsl_after_warmup", action=argparse.BooleanOptionalAction, default=True)

    # Sweep dimensions. Values are comma-separated lists.
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--lr", type=str, default="0.001")
    p.add_argument("--dropout", type=str, default="0.3")
    p.add_argument("--warmup_epochs", type=str, default="120")
    p.add_argument("--lambda_recon", type=str, default="0.5,0.4")
    p.add_argument("--lambda_cluster", type=str, default="3.0,2.0")
    p.add_argument("--lambda_smooth", type=str, default="0.05")
    p.add_argument("--topk_edges", type=str, default="12")
    p.add_argument("--edge_adjust_interval", type=str, default="10")
    p.add_argument("--delta_edges", type=str, default="50")
    p.add_argument("--beta_saturation", type=str, default="0.90")
    p.add_argument("--gamma_saturation", type=str, default="0.60")
    p.add_argument("--hsl_residual_strength", type=str, default="0.3,0.5")

    return p.parse_args()


def load_modalities(args):
    """Load and preprocess RNA/ATAC once for all sweep runs."""
    rna_path = os.path.join(args.data_dir, args.rna_file)
    atac_path = os.path.join(args.data_dir, args.atac_file)

    print(f"\nLoading RNA data from {rna_path}...")
    adata_rna = sc.read_h5ad(rna_path)

    print(f"Loading ATAC data from {atac_path}...")
    adata_atac = sc.read_h5ad(atac_path)

    print("\nPreprocessing RNA...")
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(
        adata_rna,
        flavor="seurat_v3",
        n_top_genes=args.n_top_genes,
        check_values=False,
    )
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    rna_features = pca(adata_rna_high, n_comps=args.n_pca, random_state=42)

    print("Preprocessing ATAC...")
    adata_atac = adata_atac[adata_rna.obs_names].copy()
    if "X_lsi" not in adata_atac.obsm.keys():
        sc.pp.highly_variable_genes(
            adata_atac,
            flavor="seurat_v3",
            n_top_genes=args.n_top_genes,
            check_values=False,
        )
        lsi(adata_atac, use_highly_variable=False, n_components=args.n_lsi)
    atac_features = adata_atac.obsm["X_lsi"].copy()

    coords = extract_coords(adata_atac)

    if args.label_col in adata_atac.obs.columns:
        raw_labels = adata_atac.obs[args.label_col].astype(str).values
        labels = label_encode(raw_labels)
        n_classes = len(np.unique(labels))
        print(f"\nLabels: {n_classes} classes from '{args.label_col}'")
    else:
        labels = None
        n_classes = 14
        print(f"\nNo label column '{args.label_col}' found; unsupervised mode.")

    print(f"RNA features: {rna_features.shape}")
    print(f"ATAC features: {atac_features.shape}")
    print(f"Coordinates: {coords.shape}")
    print(f"Nodes: {coords.shape[0]}")

    return [rna_features, atac_features], coords, labels, n_classes


def build_grid(args) -> List[dict]:
    dimensions = {
        "seed": _parse_list(args.seeds, int),
        "lr": _parse_list(args.lr, float),
        "dropout": _parse_list(args.dropout, float),
        "warmup_epochs": _parse_list(args.warmup_epochs, int),
        "lambda_recon": _parse_list(args.lambda_recon, float),
        "lambda_cluster": _parse_list(args.lambda_cluster, float),
        "lambda_smooth": _parse_list(args.lambda_smooth, float),
        "topk_edges": _parse_list(args.topk_edges, int),
        "edge_adjust_interval": _parse_list(args.edge_adjust_interval, int),
        "delta_edges": _parse_list(args.delta_edges, int),
        "beta_saturation": _parse_list(args.beta_saturation, float),
        "gamma_saturation": _parse_list(args.gamma_saturation, float),
        "hsl_residual_strength": _parse_list(args.hsl_residual_strength, float),
    }
    keys = list(dimensions.keys())
    grid = [dict(zip(keys, values)) for values in itertools.product(*(dimensions[k] for k in keys))]
    if args.max_runs and args.max_runs > 0:
        grid = grid[: args.max_runs]
    return grid


def append_result(path: str, row: dict, fieldnames: Iterable[str]):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    grid = build_grid(args)
    print(f"Prepared {len(grid)} sweep run(s). Device: {device}")
    for i, cfg in enumerate(grid, start=1):
        print(f"  [{i:03d}] {cfg}")
    if args.dry_run:
        return

    modality_data, coords, labels, n_classes = load_modalities(args)

    fieldnames = [
        "run_id", "status", "elapsed_sec", "error",
        "ari", "nmi", "ami", "silhouette",
        "best_observed_ari", "best_observed_nmi", "best_observed_epoch",
        "seed", "lr", "dropout", "warmup_epochs",
        "lambda_recon", "lambda_cluster", "lambda_smooth",
        "topk_edges", "edge_adjust_interval", "delta_edges",
        "beta_saturation", "gamma_saturation", "hsl_residual_strength",
        "use_hsl_spatial", "use_dynamic_feature", "allow_edge_add",
        "freeze_edges_after_warmup", "freeze_hsl_after_warmup",
    ]

    for run_id, cfg in enumerate(grid, start=1):
        print("\n" + "=" * 80)
        print(f"SWEEP RUN {run_id}/{len(grid)}")
        print(cfg)
        print("=" * 80)

        setup_seed(cfg["seed"])
        t0 = time.time()
        row = {
            "run_id": run_id,
            "status": "ok",
            "error": "",
            "use_hsl_spatial": _bool_to_cli(args.use_hsl_spatial),
            "use_dynamic_feature": _bool_to_cli(args.use_dynamic_feature),
            "allow_edge_add": _bool_to_cli(args.allow_edge_add),
            "freeze_edges_after_warmup": _bool_to_cli(args.freeze_edges_after_warmup),
            "freeze_hsl_after_warmup": _bool_to_cli(args.freeze_hsl_after_warmup),
            **cfg,
        }

        try:
            trainer = DHGNNTrainer(
                coords=coords,
                modality_data=modality_data,
                labels=labels,
                n_classes=n_classes,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                dropout=cfg["dropout"],
                lr=cfg["lr"],
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                warmup_epochs=cfg["warmup_epochs"],
                seed=cfg["seed"],
                device=device,
                lambda_recon=cfg["lambda_recon"],
                lambda_cluster=cfg["lambda_cluster"],
                lambda_smooth=cfg["lambda_smooth"],
                max_spatial_edges=args.max_spatial_edges,
                use_hsl_spatial=args.use_hsl_spatial,
                use_dynamic_feature=args.use_dynamic_feature,
                edge_adjust_interval=cfg["edge_adjust_interval"],
                delta_edges=cfg["delta_edges"],
                beta_saturation=cfg["beta_saturation"],
                gamma_saturation=cfg["gamma_saturation"],
                topk_edges=cfg["topk_edges"],
                min_edges=args.min_edges,
                max_edges=coords.shape[0],
                hsl_residual_strength=cfg["hsl_residual_strength"],
                allow_edge_add=args.allow_edge_add,
                freeze_edges_after_warmup=args.freeze_edges_after_warmup,
                freeze_hsl_after_warmup=args.freeze_hsl_after_warmup,
            )
            metrics = trainer.fit()
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    row[key] = float(value)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print("Run failed:")
            traceback.print_exc()
        finally:
            row["elapsed_sec"] = round(time.time() - t0, 2)
            append_result(args.output_csv, row, fieldnames)
            print(f"Saved result to {args.output_csv}")


if __name__ == "__main__":
    main()
