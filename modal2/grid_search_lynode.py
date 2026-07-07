"""
Grid search for DvDHGNN on Human Lymph Node dataset.
Usage: python -m modal_1.grid_search_lynode
"""
from __future__ import annotations
import itertools, os, sys, time, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.preprocessing import pca, extract_coords
from modal_1.trainer import DHGNNTrainer
from modal_1.utils import evaluate_clustering, label_encode, setup_seed

PARAM_GRID = {
    "hidden_dim": [64, 128],
    "n_layers": [2, 3],
    "lr": [0.0005, 0.001, 0.002],
    "weight_decay": [5e-5, 1e-4],
    "dropout": [0.1, 0.2, 0.3],
    "lambda_cluster": [1.0, 1.5, 2.0],
    "lambda_smooth": [0.01, 0.05, 0.1],
    "lambda_recon": [0.3, 0.5],
    "warmup_epochs": [60, 80, 120],
    "topk_edges": [3, 5, 7],
    "hsl_residual_strength": [0.4, 0.6, 0.8],
    "beta_saturation": [0.6, 0.8, 0.9],
    "edge_adjust_interval": [10, 15, 20],
    "freeze_edges_after_warmup": [True, False],
}

FIXED_PARAMS = {
    "data_dir": "data/human_lynode",
    "rna_file": "adata_RNA_with_annotation.h5ad",
    "prot_file": "adata_ADT_with_annotation.h5ad",
    "label_col": "LayerName",
    "n_top_genes": 3000, "n_pca_rna": 30, "n_pca_prot": 50,
    "n_feature_edges": 60, "k_nodes": 15, "k_edges": 8,
    "epochs": 500, "patience": 80,
    "dec_stability_patience": 5, "dec_stability_tol": 0.003,
    "dec_stability_min_epochs": 50, "max_spatial_edges": 3484,
    "gamma_saturation": 0.99, "min_edges": 80, "delta_edges": 15,
    "allow_edge_add": True, "use_hsl_spatial": True,
    "use_dynamic_feature": True, "seed": 42,
}

MAX_TRIALS = 100
OUTPUT_CSV = "results/grid_search_lynode.csv"


def sample_grid(grid, max_trials=None, seed=42):
    keys = sorted(grid.keys())
    all_combos = list(itertools.product(*(grid[k] for k in keys)))
    if max_trials and len(all_combos) > max_trials:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_combos), size=max_trials, replace=False)
        all_combos = [all_combos[i] for i in idx]
    return keys, all_combos


def load_and_preprocess(cfg):
    adata_rna = sc.read_h5ad(os.path.join(cfg["data_dir"], cfg["rna_file"]))
    adata_prot = sc.read_h5ad(os.path.join(cfg["data_dir"], cfg["prot_file"]))
    adata_rna.var_names_make_unique()
    adata_prot.var_names_make_unique()
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=cfg["n_top_genes"])
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    rna_features = pca(adata_rna[:, adata_rna.var["highly_variable"]], n_comps=cfg["n_pca_rna"])
    sc.pp.normalize_total(adata_prot, target_sum=1e4)
    sc.pp.log1p(adata_prot)
    sc.pp.scale(adata_prot)
    prot_features = pca(adata_prot, n_comps=min(cfg["n_pca_prot"], adata_prot.n_vars - 1))
    coords = extract_coords(adata_rna)
    lbl_col = cfg["label_col"]
    if lbl_col in adata_rna.obs.columns:
        labels = label_encode(adata_rna.obs[lbl_col].astype(str).values)
        n_classes = len(np.unique(labels))
    else:
        labels, n_classes = None, 10
    return rna_features, prot_features, coords, labels, n_classes, adata_rna


def run_trial(params, data_bundle):
    rna_f, prot_f, coords, labels, n_cls, adata_rna = data_bundle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(params["seed"])
    trainer = DHGNNTrainer(
        coords=coords, modality_data=[rna_f, prot_f], labels=labels,
        n_classes=n_cls, hidden_dim=params["hidden_dim"],
        n_layers=params["n_layers"], dropout=params["dropout"],
        lr=params["lr"], weight_decay=params["weight_decay"],
        epochs=params["epochs"], patience=params["patience"],
        warmup_epochs=params["warmup_epochs"], seed=params["seed"],
        device=device, lambda_cluster=params["lambda_cluster"],
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
        dec_stability_patience=params["dec_stability_patience"],
        dec_stability_tol=params["dec_stability_tol"],
        dec_stability_min_epochs=params["dec_stability_min_epochs"],
        n_feature_edges=params.get("n_feature_edges"),
        k_nodes=params.get("k_nodes"), k_edges=params.get("k_edges"),
    )
    tm = trainer.fit()
    preds = trainer.get_predictions()
    res = {}
    if labels is not None:
        mc = evaluate_clustering(labels, preds)
        res.update(mc)  # ari, nmi, ami, homogeneity, completeness, v_measure
    res["morans_i_cluster"] = tm.get("morans_i_cluster", float("nan"))
    res["morans_i_emb_mean"] = tm.get("morans_i_embedding_mean", float("nan"))
    res["best_obs_ari"] = tm.get("best_observed_ari", -1)
    return res


def main():
    print("=" * 70)
    print("  DvDHGNN Grid Search — Human Lymph Node")
    print("=" * 70)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print("\nLoading and preprocessing data...")
    data_bundle = load_and_preprocess(FIXED_PARAMS)
    print("Data loaded.\n")

    keys, combos = sample_grid(PARAM_GRID, max_trials=MAX_TRIALS, seed=FIXED_PARAMS["seed"])
    n_trials = len(combos)
    total_grid = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    print(f"Total trials: {n_trials} (full grid: {total_grid})")
    print(f"Searched: {keys}\n")

    results = []
    best_ari, best_trial = -1.0, -1

    for i, combo in enumerate(combos):
        params = dict(FIXED_PARAMS)
        for k, v in zip(keys, combo):
            params[k] = v

        searched = {k: v for k, v in zip(keys, combo)}
        print(f"\n{'─'*70}")
        print(f"Trial {i+1}/{n_trials}  {searched}")
        print(f"{'─'*70}")

        t0 = time.time()
        try:
            result = run_trial(params, data_bundle)
            elapsed = time.time() - t0
            result["elapsed_s"] = elapsed
            result["trial"] = i + 1
            result["status"] = "ok"
            ari = result.get("ari", -1)
            if ari > best_ari:
                best_ari, best_trial = ari, i + 1
            print(f"  >>> ARI={result.get('ari',-1):.4f} "
                  f"NMI={result.get('nmi',-1):.4f} "
                  f"MI_c={result.get('morans_i_cluster',0):.4f} [{elapsed:.1f}s]")
            print(f"  *** BEST: ARI={best_ari:.4f} (trial {best_trial})")
        except Exception as e:
            elapsed = time.time() - t0
            result = {"trial": i+1, "status": f"error:{str(e)[:200]}", "elapsed_s": elapsed}
            print(f"  ERROR: {e}")

        for k, v in zip(keys, combo):
            result[f"p_{k}"] = v
        results.append(result)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Grid Search Done — {n_trials} trials")
    print(f"{'='*70}")
    df = pd.DataFrame(results)
    df_ok = df[df["status"] == "ok"]
    if not df_ok.empty and "ari" in df_ok.columns:
        br = df_ok.loc[df_ok["ari"].idxmax()]
        print(f"  Best ARI: {br['ari']:.4f}")
        print(f"  Best NMI: {br['nmi']:.4f}")
        print(f"  MI_cluster: {br['morans_i_cluster']:.4f}")
        print(f"  Trial: {int(br['trial'])}")
        print("  Params:")
        for k in keys:
            print(f"    {k}: {br[f'p_{k}']}")
    print(f"\n  Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
