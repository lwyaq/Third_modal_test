"""
Training pipeline for DvDHGNN v9b — Per-Modality Feature Hypergraphs.
"""

from __future__ import annotations

import time
import copy
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, silhouette_score,
)

from modal_1.networks import DualBranchDHGNN, compute_total_loss
from modal_1.hypergraph import (
    delaunay_star_edges, gene_as_hyperedge, build_incidence,
    compute_expression_weighted_incidence,
)


def mclust_via_r(embedding, n_clusters, seed=42):
    import subprocess, tempfile, os, shutil
    tmpdir = tempfile.mkdtemp(prefix="mclust_")
    data_path = os.path.join(tmpdir, "embedding.csv")
    out_path = os.path.join(tmpdir, "labels.csv")
    script_path = os.path.join(tmpdir, "run_mclust.R")
    np.savetxt(data_path, embedding, delimiter=",")
    data_r = data_path.replace("\\", "/")
    out_r = out_path.replace("\\", "/")
    r_script = f"""
library(mclust)
set.seed({seed})
data <- as.matrix(read.csv("{data_r}", header=FALSE))
fit <- Mclust(data, G={n_clusters})
write.csv(fit$classification, "{out_r}", row.names=FALSE)
"""
    with open(script_path, "w") as f:
        f.write(r_script)
    try:
        result = subprocess.run(["Rscript", script_path], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"mclust failed: {result.stderr[:300]}")
        return np.loadtxt(out_path, delimiter=",", skiprows=1, dtype=int) - 1
    except FileNotFoundError:
        return KMeans(n_clusters=n_clusters, n_init=20, random_state=seed, max_iter=500).fit_predict(embedding)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _prune_hyperedges(H, max_edges=2000, min_cardinality=3):
    n, E = H.shape
    if E <= max_edges:
        return H
    cards = np.asarray(H.sum(axis=0)).ravel()
    card_score = np.exp(-0.5 * ((cards - 15.0) / 15.0) ** 2)
    nnz_score = np.asarray(H.astype(bool).sum(axis=0)).ravel()
    nnz_score = nnz_score / (nnz_score.max() + 1e-8)
    combined = 0.6 * card_score + 0.4 * nnz_score
    combined[cards < min_cardinality] = -1
    top_idx = np.argsort(combined)[-max_edges:]
    return H[:, np.sort(top_idx)]


def _incidence_to_sparse_tensors(H, device):
    H_coo = H.tocoo()
    return (
        torch.LongTensor(H_coo.row).to(device),
        torch.LongTensor(H_coo.col).to(device),
        torch.FloatTensor(H_coo.data).to(device),
    )


class DHGNNTrainer:
    """Trainer for DvDHGNN v9b: Per-Modality Feature Hypergraphs."""

    def __init__(
        self,
        coords, modality_data, labels=None, n_classes=None,
        hidden_dim=128, n_layers=3,
        dropout=0.2, lr=0.001, weight_decay=1e-4,
        epochs=500, patience=50, warmup_epochs=80,
        seed=42, device="cpu",
        lambda_cluster=0.5, lambda_smooth=0.1,
        lambda_recon=0.5, lambda_contrast=0.0,
        max_spatial_edges=2000,
        gene_expression_matrices=None,
        expression_features=None, expression_weight=True,
    ):
        self.coords = coords
        self.modality_data = modality_data
        self.labels = labels
        self.seed = seed
        self.device = torch.device(device)

        self.X = np.hstack(modality_data).astype(np.float32)
        self.n_nodes = self.X.shape[0]
        self.input_dims = [d.shape[1] for d in modality_data]

        self.n_classes = n_classes or (len(np.unique(labels)) if labels is not None else 14)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.lambda_cluster = lambda_cluster
        self.lambda_smooth = lambda_smooth
        self.lambda_recon = lambda_recon
        self.lambda_contrast = lambda_contrast
        self.max_spatial_edges = max_spatial_edges
        self.gene_expression_matrices = gene_expression_matrices or []
        self.expression_features = expression_features
        self.expression_weight = expression_weight

        self._build_spatial_hypergraph()
        self._build_feature_hypergraphs()

    def _build_spatial_hypergraph(self):
        print("Building spatial hypergraph (Delaunay-star)...")
        E_del = delaunay_star_edges(self.coords, s_min=3, s_max=60)
        print(f"  Delaunay edges: {len(E_del)}")
        H_spatial, _, _, _ = build_incidence(self.n_nodes, [E_del], [1.0])
        if H_spatial.shape[1] > self.max_spatial_edges:
            H_spatial = _prune_hyperedges(H_spatial, max_edges=self.max_spatial_edges)
            print(f"  Pruned to {H_spatial.shape[1]} edges")
        if self.expression_weight and self.expression_features is not None:
            print("  Applying expression-weighted incidence...")
            H_spatial = compute_expression_weighted_incidence(H_spatial, self.expression_features)
            print(f"  nnz={H_spatial.nnz}")
        self.sp_rows, self.sp_cols, self.sp_vals = _incidence_to_sparse_tensors(H_spatial, self.device)
        self.n_spatial_edges = H_spatial.shape[1]
        print(f"  Final: {self.n_spatial_edges} edges, {len(self.sp_rows)} nnz")

    def _build_feature_hypergraphs(self):
        """Build one gene-as-hyperedge feature hypergraph per modality."""
        modality_names = ["RNA", "ATAC"]
        self.feat_tensors = []  # list of (ft_rows, ft_cols, ft_vals, n_feat_edges)

        for i, gene_mat in enumerate(self.gene_expression_matrices):
            name = modality_names[i] if i < len(modality_names) else f"Modality_{i}"
            print(f"Building {name} feature hypergraph (gene-as-hyperedge)...")
            H_gene, n_genes = gene_as_hyperedge(
                gene_mat,
                min_cells_pct=0.05,
                max_cells_pct=0.40,
                max_genes=3000,
                use_expression_weights=True,
            )
            print(f"  {name} gene hyperedges: {n_genes}")
            print(f"  H_gene shape: {H_gene.shape}, nnz={H_gene.nnz}")
            ft_rows, ft_cols, ft_vals = _incidence_to_sparse_tensors(H_gene, self.device)
            n_feat_edges = H_gene.shape[1]
            print(f"  Final: {n_feat_edges} edges, {len(ft_rows)} nnz")
            self.feat_tensors.append((ft_rows, ft_cols, ft_vals, n_feat_edges))

    def _forward(self, model, X_tensor):
        return model(
            X_tensor,
            self.sp_rows, self.sp_cols, self.sp_vals, self.n_spatial_edges,
            self.feat_tensors,
        )

    def fit(self):
        X_tensor = torch.FloatTensor(self.X).to(self.device)

        model = DualBranchDHGNN(
            input_dims=self.input_dims,
            hidden_dim=self.hidden_dim,
            n_classes=self.n_classes,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        best_ari = -1
        best_state = None
        patience_counter = 0
        best_epoch = 0
        dec_initialized = False

        print(f"\nTraining DvDHGNN v9b ({n_params:,} params)")
        print(f"  Device: {self.device}, Nodes: {self.n_nodes}, Classes: {self.n_classes}")
        for i, dim in enumerate(self.input_dims):
            n_feat = self.feat_tensors[i][3] if i < len(self.feat_tensors) else 0
            print(f"  Modality {i}: input_dim={dim}, encoder → {self.hidden_dim}, "
                  f"feature_hypergraph={n_feat} edges")
        print(f"  Spatial: {self.n_spatial_edges} edges (shared Delaunay-star)")
        print(f"  Losses: recon({self.lambda_recon}) + cluster({self.lambda_cluster}) "
              f"+ smooth({self.lambda_smooth})")
        print(f"  Fusion: intra-modal gate → cross-modal gate")
        print(f"  Warmup: {self.warmup_epochs} → DEC KL")
        print()

        for epoch in range(self.epochs):
            t0 = time.time()
            model.train()

            if epoch == self.warmup_epochs and not dec_initialized:
                print(f"\n>>> Warmup complete. Initializing DEC...")
                model.eval()
                with torch.no_grad():
                    emb = self._forward(model, X_tensor)["embedding"].cpu().numpy()
                km = KMeans(n_clusters=self.n_classes, n_init=20, random_state=self.seed, max_iter=300)
                km.fit(emb)
                model.cluster_centers.data.copy_(torch.FloatTensor(km.cluster_centers_).to(self.device))
                dec_initialized = True
                init_ari = adjusted_rand_score(self.labels, km.labels_) if self.labels is not None else -1
                print(f"  KMeans init ARI: {init_ari:.4f}")
                model.train()

            outputs = self._forward(model, X_tensor)
            loss, loss_dict = compute_total_loss(
                outputs, self.sp_rows, self.sp_cols, self.sp_vals,
                self.n_nodes, self.n_spatial_edges, self.input_dims,
                lambda_cluster=self.lambda_cluster,
                lambda_smooth=self.lambda_smooth,
                lambda_recon=self.lambda_recon,
                lambda_contrast=self.lambda_contrast,
                dec_phase=dec_initialized,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    metrics = self._evaluate(model, X_tensor)

                current_ari = metrics.get("ari", -1)
                if current_ari > best_ari:
                    best_ari = current_ari
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                phase = "DEC" if dec_initialized else "warmup"
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    print(
                        f"Epoch {epoch+1:4d} [{phase:7s}] | "
                        f"Loss {loss_dict['total']:.4f} "
                        f"(recon={loss_dict.get('recon', 0):.3f}, "
                        f"clust={loss_dict.get('cluster', 0):.3f}, "
                        f"sm_s={loss_dict.get('smooth_s', 0):.3f}) | "
                        f"ARI {current_ari:.4f} (best {best_ari:.4f}@{best_epoch+1}) | "
                        f"{time.time()-t0:.1f}s"
                    )

                if patience_counter >= self.patience // 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            metrics = self._evaluate(model, X_tensor)

        print(f"\nBest ARI: {best_ari:.4f} at epoch {best_epoch+1}")
        if "ari" in metrics:
            print(f"Final: ARI={metrics['ari']:.4f}, NMI={metrics['nmi']:.4f}")

        self.model = model
        self.embeddings = metrics.get("embedding", None)
        self.predictions = metrics.get("predictions", None)
        return metrics

    def _evaluate(self, model, X_tensor):
        outputs = self._forward(model, X_tensor)
        embedding = outputs["embedding"].cpu().numpy()
        km = KMeans(n_clusters=self.n_classes, n_init=20, random_state=self.seed, max_iter=500)
        predictions = km.fit_predict(embedding)
        metrics = {"embedding": embedding, "predictions": predictions}
        if self.labels is not None:
            metrics["ari"] = adjusted_rand_score(self.labels, predictions)
            metrics["nmi"] = normalized_mutual_info_score(self.labels, predictions, average_method="arithmetic")
            metrics["ami"] = adjusted_mutual_info_score(self.labels, predictions)
            try:
                metrics["silhouette"] = silhouette_score(embedding, predictions)
            except Exception:
                metrics["silhouette"] = 0.0
        return metrics

    def get_embedding(self):
        if self.embeddings is not None: return self.embeddings
        raise RuntimeError("Call .fit() first.")

    def get_predictions(self):
        if self.predictions is not None: return self.predictions
        raise RuntimeError("Call .fit() first.")

    def recluster(self, method="mclust", n_clusters=None):
        if self.embeddings is None: raise RuntimeError("Call .fit() first.")
        n_cls = n_clusters or self.n_classes
        if method == "mclust":
            predictions = mclust_via_r(self.embeddings, n_cls, self.seed)
        else:
            predictions = KMeans(n_clusters=n_cls, n_init=20, random_state=self.seed).fit_predict(self.embeddings)
        self.predictions = predictions
        metrics = {"embedding": self.embeddings, "predictions": predictions}
        if self.labels is not None:
            metrics["ari"] = adjusted_rand_score(self.labels, predictions)
            metrics["nmi"] = normalized_mutual_info_score(self.labels, predictions, average_method="arithmetic")
        return metrics
