"""
Preprocessing utilities for AGCLD model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sklearn.decomposition import PCA
import scipy
from scipy.sparse import coo_matrix
import scipy.sparse
import sklearn.utils.extmath
import sklearn.preprocessing


def pca(
    adata: sc.AnnData,
    n_comps: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Perform PCA on AnnData object."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n_comps = min(n_comps, X.shape[1] - 1, X.shape[0] - 1)
    if n_comps < 1:
        return X

    pca_model = PCA(n_components=n_comps, random_state=random_state)
    return pca_model.fit_transform(X).astype(np.float32)


def clustering(
    adata: sc.AnnData,
    key: str = "AGCLD",
    add_key: str = "AGCLD_cluster",
    n_clusters: int = 7,
    method: str = "mclust",
    random_state: int = 42,
) -> None:
    """
    Perform clustering on embeddings.

    Priority: mclust (R) → GaussianMixture (Python) → KMeans
    """
    embedding = adata.obsm[key]
    labels = None

    if method == "mclust":
        # Try R mclust first
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr

            numpy2ri.activate()
            mclust = importr("mclust")

            ro.r.assign("data", embedding)
            ro.r.assign("G", n_clusters)
            ro.r(
                """
                set.seed(42)
                fit <- Mclust(data, G=G)
                labels <- fit$classification
            """
            )
            labels = np.array(ro.r["labels"], dtype=int) - 1
            numpy2ri.deactivate()

        except Exception as e:
            print(f"R mclust unavailable ({e}), using GaussianMixture")

        # Fallback: GaussianMixture (Python mclust equivalent)
        if labels is None:
            try:
                from sklearn.mixture import GaussianMixture

                best_bic = np.inf
                best_labels = None
                for cov_type in ["full", "tied", "diag"]:
                    gm = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type=cov_type,
                        n_init=10,
                        max_iter=500,
                        random_state=random_state,
                    )
                    gm.fit(embedding)
                    if gm.bic(embedding) < best_bic:
                        best_bic = gm.bic(embedding)
                        best_labels = gm.predict(embedding)
                labels = best_labels
                print(f"  GaussianMixture best covariance: {cov_type}, BIC: {best_bic:.2f}")
            except Exception as e2:
                print(f"GaussianMixture failed ({e2}), using KMeans")

    # Final fallback: KMeans
    if labels is None:
        from sklearn.cluster import KMeans
        labels = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=20
        ).fit_predict(embedding)

    adata.obs[add_key] = pd.Categorical(labels.astype(str))


def extract_coords(adata: sc.AnnData) -> np.ndarray:
    """Extract spatial coordinates from AnnData."""
    candidate_keys = ["spatial", "coords", "X_spatial", "X_umap", "X_pca"]
    for key in candidate_keys:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(np.float32)

    cols = [c.lower() for c in adata.obs.columns]
    possible_pairs = [
        ("x", "y"),
        ("row", "col"),
        ("imagerow", "imagecol"),
        ("x_pos", "y_pos"),
        ("array_row", "array_col"),
    ]
    for x_name, y_name in possible_pairs:
        if x_name in cols and y_name in cols:
            x_col = adata.obs.columns[cols.index(x_name)]
            y_col = adata.obs.columns[cols.index(y_name)]
            return np.stack([adata.obs[x_col].values, adata.obs[y_col].values], axis=1).astype(np.float32)

    raise ValueError("Cannot find spatial coordinates in adata.obs/obsm")


def to_dense(X) -> np.ndarray:
    """Convert sparse matrix to dense array."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def lsi(
    adata: anndata.AnnData,
    n_components: int = 20,
    use_highly_variable: Optional[bool] = None,
    **kwargs,
) -> None:
    r"""LSI analysis (following the Seurat v3 approach)."""
    import scipy.sparse as sp
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)

    # L1 normalize (stays sparse)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)

    # log1p transform — need to handle sparse carefully
    if sp.issparse(X_norm):
        X_norm = X_norm.tocsr()
        X_norm.data = np.log1p(X_norm.data * 1e4)
    else:
        X_norm = np.log1p(X_norm * 1e4)

    # Randomized SVD
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""TF-IDF normalization (following the Seurat v3 approach)."""
    import scipy.sparse as sp
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    n_cells = X.shape[0]

    # idf: n_cells / column sums
    col_sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    col_sums[col_sums == 0] = 1.0
    idf = n_cells / col_sums

    # tf: row-normalize each cell by its total counts
    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    inv_row = sp.diags(1.0 / row_sums)
    tf = inv_row @ X

    # Apply idf via diagonal matrix multiplication
    idf_diag = sp.diags(idf)
    result = tf @ idf_diag
    return result
