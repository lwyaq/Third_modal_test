"""
Small grid search for DvDHGNN on Human Lymph Node dataset.

Search parameters:
    lr
    dropout
    warmup_epochs
    lambda_cluster
    lambda_smooth

Usage:
    python -m modal_1.grid_search_human_lynode_small
"""

from __future__ import annotations

import itertools
import os
import sys
import time
import warnings

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.preprocessing import pca, extract_coords, clustering
from modal_1.trainer import DHGNNTrainer
from modal_1.utils import evaluate_clustering, label_encode, setup_seed


# ============================================================
# 1. Parameters to search
# ============================================================

PARAM_GRID = {
    "lr": [0.0005, 0.001],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "warmup_epochs": [60, 80],
    "lambda_cluster": [1.0, 1.5, 2.0],
    "lambda_smooth": [0.05, 0.1, 0.15],
}

# 完整遍历 144 组
MAX_TRIALS = None

OUTPUT_CSV = "results/grid_search_human_lynode_small.csv"


# ============================================================
# 2. Fixed parameters based on your current run_human_lynode.py
# ============================================================

FIXED_PARAMS = {
    # Data
    "data_dir": "data/human_lynode",
    "rna_file": "adata_RNA_with_annotation.h5ad",
    "prot_file": "adata_ADT_with_annotation.h5ad",
    "label_col": "LayerName",

    # Preprocessing
    "n_top_genes": 3000,
    "n_pca_rna": 50,
    "n_pca_prot": 50,  # Protein 原始 31 维，实际 PCA 后为 30 维

    # Model structure
    "hidden_dim": 128,
    "n_layers": 2,
    "n_feature_edges": 60,
    "k_nodes": 15,
    "k_edges": 8,

    # Spatial hypergraph
    "use_hsl_spatial": True,
    "hsl_residual_strength": 0.8,
    "max_spatial_edges": 3484,

    # Dynamic feature hypergraph
    "use_dynamic_feature": True,
    "edge_adjust_interval": 10,
    "delta_edges": 50,
    "beta_saturation": 0.6,
    "gamma_saturation": 0.98,
    "topk_edges": 3,
    "min_edges": 100,
    "allow_edge_add": True,
    "freeze_edges_after_warmup": True,

    # Training
    "epochs": 500,
    "patience": 50,
    "weight_decay": 1e-4,

    # Loss
    "lambda_recon": 0.5,

    # DEC stability
    "dec_stability_patience": 3,
    "dec_stability_tol": 0.005,
    "dec_stability_min_epochs": 20,

    # Seed
    "seed": 42,
}


# ============================================================
# 3. Grid utilities
# ============================================================

def sample_grid(grid, max_trials=None, seed=42):
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))

    if max_trials is not None and len(combos) > max_trials:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combos), size=max_trials, replace=False)
        combos = [combos[i] for i in idx]

    return keys, combos


# ============================================================
# 4. Data loading and preprocessing
# ============================================================

def load_and_preprocess(cfg):
    print("\nLoading data...")

    rna_path = os.path.join(cfg["data_dir"], cfg["rna_file"])
    prot_path = os.path.join(cfg["data_dir"], cfg["prot_file"])

    adata_rna = sc.read_h5ad(rna_path)
    adata_prot = sc.read_h5ad(prot_path)

    adata_rna.var_names_make_unique()
    adata_prot.var_names_make_unique()

    # Align RNA and Protein if necessary
    if not np.array_equal(adata_rna.obs_names, adata_prot.obs_names):
        print("RNA and Protein obs_names are not aligned. Aligning...")
        common_obs = adata_rna.obs_names.intersection(adata_prot.obs_names)
        adata_rna = adata_rna[common_obs].copy()
        adata_prot = adata_prot[common_obs].copy()

    assert np.array_equal(adata_rna.obs_names, adata_prot.obs_names), \
        "RNA and Protein obs_names are not aligned!"

    print("RNA shape:", adata_rna.shape)
    print("PROT shape:", adata_prot.shape)
    print("Same obs_names:", np.array_equal(adata_rna.obs_names, adata_prot.obs_names))

    # -------------------------
    # RNA preprocessing
    # -------------------------
    print("\nPreprocessing RNA...")
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(
        adata_rna,
        flavor="seurat_v3",
        n_top_genes=cfg["n_top_genes"],
        check_values=False,
    )
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)

    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]]
    rna_features = pca(
        adata_rna_high,
        n_comps=cfg["n_pca_rna"],
        random_state=cfg["seed"],
    )

    # -------------------------
    # Protein / ADT preprocessing
    # 保持你当前效果较好的 PCA 版本
    # -------------------------
    print("Preprocessing Protein (ADT)...")
    sc.pp.normalize_total(adata_prot, target_sum=1e4)
    sc.pp.log1p(adata_prot)
    sc.pp.scale(adata_prot)

    n_prot_pca = min(cfg["n_pca_prot"], adata_prot.n_vars - 1)
    prot_features = pca(
        adata_prot,
        n_comps=n_prot_pca,
        random_state=cfg["seed"],
    )

    # -------------------------
    # Coordinates and labels
    # -------------------------
    coords = extract_coords(adata_rna)

    label_col = cfg["label_col"]
    if label_col in adata_rna.obs.columns:
        raw_labels = adata_rna.obs[label_col].astype(str).values
        labels = label_encode(raw_labels)
        n_classes = len(np.unique(labels))

        print(f"\nLabels: {n_classes} classes from '{label_col}'")
        print("Label distribution:")
        print(pd.Series(raw_labels).value_counts())
    else:
        labels = None
        n_classes = 10
        print(f"\nNo label column '{label_col}' found; use n_classes={n_classes}")

    print("\nFeature summary:")
    print("RNA features:", rna_features.shape)
    print("Protein features:", prot_features.shape)
    print("Coordinates:", coords.shape)
    print("Nodes:", coords.shape[0])

    print("Any NaN RNA:", np.isnan(rna_features).any())
    print("Any NaN Protein:", np.isnan(prot_features).any())
    print("Any NaN coords:", np.isnan(coords).any())

    return rna_features, prot_features, coords, labels, n_classes, adata_rna


# ============================================================
# 5. Run one trial
# ============================================================

def run_trial(params, data_bundle):
    rna_features, prot_features, coords, labels, n_classes, adata_rna = data_bundle

    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(params["seed"])

    trainer = DHGNNTrainer(
        coords=coords,
        modality_data=[rna_features, prot_features],
        labels=labels,
        n_classes=n_classes,

        hidden_dim=params["hidden_dim"],
        n_layers=params["n_layers"],
        n_feature_edges=params["n_feature_edges"],
        k_nodes=params["k_nodes"],
        k_edges=params["k_edges"],

        dropout=params["dropout"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        epochs=params["epochs"],
        patience=params["patience"],
        warmup_epochs=params["warmup_epochs"],

        dec_stability_patience=params["dec_stability_patience"],
        dec_stability_tol=params["dec_stability_tol"],
        dec_stability_min_epochs=params["dec_stability_min_epochs"],

        seed=params["seed"],
        device=device,

        lambda_cluster=params["lambda_cluster"],
        lambda_smooth=params["lambda_smooth"],
        lambda_recon=params["lambda_recon"],

        max_spatial_edges=params["max_spatial_edges"],

        use_hsl_spatial=params["use_hsl_spatial"],
        use_dynamic_feature=params["use_dynamic_feature"],

        edge_adjust_interval=params["edge_adjust_interval"],
        delta_edges=params["delta_edges"],
        beta_saturation=params["beta_saturation"],
        gamma_saturation=params["gamma_saturation"],
        topk_edges=params["topk_edges"],
        min_edges=params["min_edges"],
        max_edges=coords.shape[0],

        hsl_residual_strength=params["hsl_residual_strength"],
        allow_edge_add=params["allow_edge_add"],
        freeze_edges_after_warmup=params["freeze_edges_after_warmup"],
    )

    train_metrics = trainer.fit()

    result = {}

    # -------------------------
    # Internal model prediction
    # -------------------------
    internal_preds = trainer.get_predictions()

    if labels is not None:
        internal_metrics = evaluate_clustering(labels, internal_preds)
        for k, v in internal_metrics.items():
            result[f"internal_{k}"] = v

    # -------------------------
    # mclust on final embedding
    # -------------------------
    embedding = trainer.get_embedding()

    adata_eval = anndata.AnnData(obs=adata_rna.obs.copy())
    adata_eval.obsm["DvDHGNN"] = embedding

    clustering(
        adata_eval,
        key="DvDHGNN",
        add_key="DvDHGNN_cluster",
        n_clusters=n_classes,
        method="mclust",
        random_state=params["seed"],
    )

    mclust_preds = adata_eval.obs["DvDHGNN_cluster"].astype(int).values

    if labels is not None:
        mclust_metrics = evaluate_clustering(labels, mclust_preds)
        for k, v in mclust_metrics.items():
            result[f"mclust_{k}"] = v

    # -------------------------
    # Extra trainer metrics
    # -------------------------
    result["morans_i_cluster"] = train_metrics.get("morans_i_cluster", np.nan)
    result["morans_i_embedding_mean"] = train_metrics.get("morans_i_embedding_mean", np.nan)
    result["best_observed_ari"] = train_metrics.get("best_observed_ari", -1)
    result["best_observed_nmi"] = train_metrics.get("best_observed_nmi", -1)
    result["best_observed_epoch"] = train_metrics.get("best_observed_epoch", -1)

    return result


# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 80)
    print("  Small Grid Search — DvDHGNN Human Lymph Node")
    print("=" * 80)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    data_bundle = load_and_preprocess(FIXED_PARAMS)

    keys, combos = sample_grid(
        PARAM_GRID,
        max_trials=MAX_TRIALS,
        seed=FIXED_PARAMS["seed"],
    )

    print("\nSearch parameters:")
    for k in keys:
        print(f"  {k}: {PARAM_GRID[k]}")

    print(f"\nTotal trials: {len(combos)}")
    print(f"Results will be saved to: {OUTPUT_CSV}")

    results = []
    best_ari = -1.0
    best_trial = -1

    for i, combo in enumerate(combos):
        params = dict(FIXED_PARAMS)
        searched = {}

        for k, v in zip(keys, combo):
            params[k] = v
            searched[k] = v

        print("\n" + "─" * 80)
        print(f"Trial {i + 1}/{len(combos)}")
        for k, v in searched.items():
            print(f"  {k}: {v}")
        print("─" * 80)

        t0 = time.time()

        try:
            result = run_trial(params, data_bundle)
            elapsed = time.time() - t0

            result["trial"] = i + 1
            result["status"] = "ok"
            result["elapsed_s"] = elapsed

            mclust_ari = result.get("mclust_ari", -1)
            mclust_nmi = result.get("mclust_nmi", -1)
            internal_ari = result.get("internal_ari", -1)

            if mclust_ari > best_ari:
                best_ari = mclust_ari
                best_trial = i + 1

            print(
                f"\n>>> Finished Trial {i + 1} "
                f"| internal ARI={internal_ari:.4f} "
                f"| mclust ARI={mclust_ari:.4f} "
                f"| mclust NMI={mclust_nmi:.4f} "
                f"| time={elapsed:.1f}s"
            )
            print(f"*** Current best mclust ARI={best_ari:.4f} at trial {best_trial}")

        except Exception as e:
            elapsed = time.time() - t0

            result = {
                "trial": i + 1,
                "status": f"error: {str(e)[:300]}",
                "elapsed_s": elapsed,
            }

            print(f"\nERROR in Trial {i + 1}: {e}")

        for k, v in searched.items():
            result[f"p_{k}"] = v

        results.append(result)

        # Save after every trial
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 80)
    print("  Grid Search Done")
    print("=" * 80)

    df = pd.DataFrame(results)
    df_ok = df[df["status"] == "ok"].copy()

    if not df_ok.empty and "mclust_ari" in df_ok.columns:
        best_row = df_ok.loc[df_ok["mclust_ari"].idxmax()]

        print(f"Best mclust ARI: {best_row['mclust_ari']:.4f}")
        print(f"Best mclust NMI: {best_row.get('mclust_nmi', np.nan):.4f}")
        print(f"Best internal ARI: {best_row.get('internal_ari', np.nan):.4f}")
        print(f"Trial: {int(best_row['trial'])}")

        print("\nBest params:")
        for k in keys:
            print(f"  {k}: {best_row[f'p_{k}']}")

        print("\nTop 10 by mclust ARI:")
        show_cols = (
            ["trial", "mclust_ari", "mclust_nmi", "internal_ari", "best_observed_ari"]
            + [f"p_{k}" for k in keys]
        )
        show_cols = [c for c in show_cols if c in df_ok.columns]

        print(
            df_ok.sort_values("mclust_ari", ascending=False)[show_cols]
            .head(10)
            .to_string(index=False)
        )

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()