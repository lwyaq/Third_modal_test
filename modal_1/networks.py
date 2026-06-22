"""
DvDHGNN v9b — Per-Modality Encoders + Feature Hypergraphs

Each modality (RNA, ATAC) has:
  - Its own encoder (MLP)
  - Its own spatial HGNN (operates on shared H_spatial)
  - Its own feature HGNN (operates on its own H_feature)

Architecture:
  Input → per-modality encoder → per-modality dual HGNN → cross-modal fusion → output
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# ====================================================================== #
#                     Sparse Incidence Message Passing                    #
# ====================================================================== #

class SparseHGNNConv(nn.Module):
    """Sparse hypergraph convolution: V->E->V with skip connection."""

    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.W_v2e = nn.Linear(in_dim, out_dim)
        self.W_e2v = nn.Linear(out_dim, out_dim)
        self.W_skip = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = dropout

    def forward(self, x, H_rows, H_cols, H_vals, n_nodes, n_edges):
        Dv = torch.zeros(n_nodes, device=x.device)
        Dv.scatter_add_(0, H_rows, H_vals)
        Dv = Dv.clamp(min=1.0)

        De = torch.zeros(n_edges, device=x.device)
        De.scatter_add_(0, H_cols, H_vals)
        De = De.clamp(min=1.0)

        node_feats = self.act(self.W_v2e(x))
        gathered_v = node_feats[H_rows]
        weighted_v = gathered_v * H_vals.unsqueeze(1)
        edge_feats = torch.zeros(n_edges, node_feats.shape[1], device=x.device)
        edge_feats.scatter_add_(0, H_cols.unsqueeze(1).expand_as(weighted_v), weighted_v)
        edge_feats = edge_feats / De.unsqueeze(1)

        edge_proj = self.act(self.W_e2v(edge_feats))
        gathered_e = edge_proj[H_cols]
        weighted_e = gathered_e * H_vals.unsqueeze(1)
        node_msg = torch.zeros(n_nodes, edge_proj.shape[1], device=x.device)
        node_msg.scatter_add_(0, H_rows.unsqueeze(1).expand_as(weighted_e), weighted_e)
        node_msg = node_msg / Dv.unsqueeze(1)

        node_msg = self.norm1(node_msg)
        node_msg = F.dropout(node_msg, p=self.dropout, training=self.training)
        x_out = node_msg + self.W_skip(x)
        x_out = self.norm2(x_out)
        return x_out


class HSLSpatialRefiner(nn.Module):
    """Learn incidence weights on a fixed spatial hypergraph topology."""

    def __init__(self, hidden_dim: int, residual_strength: float = 0.5):
        super().__init__()
        self.residual_strength = residual_strength
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.edge_gate[-1].weight)
        nn.init.zeros_(self.edge_gate[-1].bias)
        self.last_stats: Dict[str, float] = {}

    def forward(self, h_s, sp_rows, sp_cols, sp_vals, n_nodes, n_spatial_edges):
        De = torch.zeros(n_spatial_edges, device=h_s.device)
        De.scatter_add_(0, sp_cols, sp_vals)
        De = De.clamp(min=1.0)
        weighted_nodes = h_s[sp_rows] * sp_vals.unsqueeze(1)
        edge_feats = torch.zeros(n_spatial_edges, h_s.shape[1], device=h_s.device)
        edge_feats.scatter_add_(0, sp_cols.unsqueeze(1).expand_as(weighted_nodes), weighted_nodes)
        edge_feats = edge_feats / De.unsqueeze(1)

        logits = self.edge_gate(torch.cat([h_s[sp_rows], edge_feats[sp_cols]], dim=1)).squeeze(-1)
        learned = 2.0 * torch.sigmoid(logits)
        scale = (1.0 - self.residual_strength) + self.residual_strength * learned
        refined = sp_vals * scale.clamp(0.1, 2.0)
        with torch.no_grad():
            self.last_stats = {
                "min": float(refined.min().detach().cpu()),
                "max": float(refined.max().detach().cpu()),
                "mean": float(refined.mean().detach().cpu()),
            }
        return refined


class VariableBioDynamicFeatureHypergraph(nn.Module):
    """Biologically initialized dynamic feature hypergraph with variable edge count."""

    def __init__(self, init_node_features, hidden_dim: int, topk_edges: int = 3,
                 min_edges: int = 100, max_edges: Optional[int] = None):
        super().__init__()
        init = torch.as_tensor(init_node_features, dtype=torch.float32).detach().clone()
        if init.shape[1] == hidden_dim:
            prototypes = init
        else:
            projection = torch.randn(init.shape[1], hidden_dim) / (init.shape[1] ** 0.5)
            prototypes = init @ projection
        self.edge_features = nn.Parameter(prototypes)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.topk_edges = topk_edges
        self.min_edges = min_edges
        self.max_edges = max_edges or init.shape[0]
        self.last_edge_usage: Optional[torch.Tensor] = None
        self.last_saturation = 0.0

    @property
    def n_edges(self) -> int:
        return int(self.edge_features.shape[0])

    def forward(self, node_embeddings):
        n_nodes = node_embeddings.shape[0]
        n_edges = self.n_edges
        k = min(self.topk_edges, n_edges)
        scores = self.query(node_embeddings) @ self.key(self.edge_features).T
        scores = scores / (node_embeddings.shape[1] ** 0.5)
        attn = torch.softmax(scores, dim=1)
        vals, cols = torch.topk(attn, k=k, dim=1)
        rows = torch.arange(n_nodes, device=node_embeddings.device).repeat_interleave(k)
        cols = cols.reshape(-1)
        vals = vals.reshape(-1)
        usage = torch.bincount(cols.detach(), minlength=n_edges).to(node_embeddings.device)
        self.last_edge_usage = usage
        self.last_saturation = float((usage > 0).float().mean().detach().cpu())
        return rows, cols, vals, n_edges

    @torch.no_grad()
    def adjust_edges(self, beta=0.90, gamma=0.98, delta_edges=20, allow_add=True):
        device = self.edge_features.device
        usage = self.last_edge_usage
        n_edges = self.n_edges
        if usage is None:
            return {"action": "skip", "saturation": 0.0, "empty": n_edges, "n_edges": n_edges}
        non_empty = int((usage > 0).sum().item())
        empty = n_edges - non_empty
        saturation = non_empty / max(n_edges, 1)
        action = "keep"
        if saturation < beta and n_edges > self.min_edges:
            keep_count = max(self.min_edges, n_edges - delta_edges)
            keep_idx = torch.argsort(usage, descending=True)[:keep_count].sort().values
            self.edge_features = nn.Parameter(self.edge_features.data[keep_idx].clone().to(device))
            action = "prune"
        elif saturation > gamma and allow_add and n_edges < self.max_edges:
            add_count = min(delta_edges, self.max_edges - n_edges)
            source = torch.randint(0, n_edges, (add_count,), device=device)
            noise = 0.01 * torch.randn(add_count, self.edge_features.shape[1], device=device)
            added = self.edge_features.data[source] + noise
            self.edge_features = nn.Parameter(torch.cat([self.edge_features.data, added], dim=0))
            action = "add"
        return {"action": action, "saturation": saturation, "empty": empty, "n_edges": self.n_edges}


# ====================================================================== #
#              Per-Modality Dual-Hypergraph Model (v9b)                   #
# ====================================================================== #

class DualBranchDHGNN(nn.Module):
    """
    DvDHGNN v9b: Per-modality encoders + feature hypergraphs.

    For each modality m in {RNA, ATAC}:
      - Encoder: x_m -> h_m (hidden_dim)
      - Spatial HGNN: message passing on shared H_spatial
      - Feature HGNN: message passing on modality-specific H_feature_m
      - Intra-modal fusion: gate * spatial + (1-gate) * feature

    Cross-modal fusion: gate_rna * h_rna + (1-gate_rna) * h_atac
    """

    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 128,
        n_classes: int = 14,
        n_layers: int = 3,
        dropout: float = 0.2,
        init_edge_features: Optional[List[torch.Tensor]] = None,
        use_hsl_spatial: bool = False,
        use_dynamic_feature: bool = False,
        topk_edges: int = 3,
        min_edges: int = 100,
        max_edges: Optional[int] = None,
        hsl_residual_strength: float = 0.5,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.n_modalities = len(input_dims)
        self.n_layers = n_layers
        self.use_hsl_spatial = use_hsl_spatial
        self.use_dynamic_feature = use_dynamic_feature

        # ---- Per-modality encoders ----
        self.encoders = nn.ModuleList()
        for dim in input_dims:
            self.encoders.append(nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ))

        # ---- Per-modality spatial HGNNs (operate on shared H_spatial) ----
        self.spatial_convs = nn.ModuleList()
        for _ in range(self.n_modalities):
            self.spatial_convs.append(nn.ModuleList([
                SparseHGNNConv(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(n_layers)
            ]))

        self.spatial_refiners = nn.ModuleList()
        for _ in range(self.n_modalities):
            self.spatial_refiners.append(nn.ModuleList([
                HSLSpatialRefiner(hidden_dim, residual_strength=hsl_residual_strength)
                for _ in range(n_layers)
            ]))

        # ---- Per-modality feature HGNNs (operate on modality-specific H_feature) ----
        self.feature_convs = nn.ModuleList()
        for _ in range(self.n_modalities):
            self.feature_convs.append(nn.ModuleList([
                SparseHGNNConv(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(n_layers)
            ]))

        self.dynamic_feature_builders = nn.ModuleList()
        if use_dynamic_feature:
            if init_edge_features is None:
                raise ValueError("init_edge_features is required when use_dynamic_feature=True")
            for feats in init_edge_features:
                self.dynamic_feature_builders.append(VariableBioDynamicFeatureHypergraph(
                    feats, hidden_dim, topk_edges=topk_edges, min_edges=min_edges, max_edges=max_edges
                ))

        # ---- Per-modality intra-modal fusion gates ----
        self.intra_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(self.n_modalities)
        ])
        self.intra_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.n_modalities)
        ])

        # ---- Cross-modal fusion ----
        self.cross_gate = nn.Parameter(torch.tensor(0.0))
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ---- Unsupervised DEC cluster centers ----
        self.cluster_centers = Parameter(torch.randn(n_classes, hidden_dim) * 0.01)

        # ---- Per-modality decoders ----
        self.decoders = nn.ModuleList()
        for dim in input_dims:
            self.decoders.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            ))

    def adjust_dynamic_feature_edges(self, beta=0.90, gamma=0.98, delta_edges=20, allow_add=True):
        if not self.use_dynamic_feature:
            return []
        logs = []
        for m, builder in enumerate(self.dynamic_feature_builders):
            log = builder.adjust_edges(beta=beta, gamma=gamma, delta_edges=delta_edges, allow_add=allow_add)
            log["modality"] = m
            logs.append(log)
        return logs

    def hsl_stats(self):
        stats = []
        for m, refiners in enumerate(self.spatial_refiners):
            for layer_i, refiner in enumerate(refiners):
                if refiner.last_stats:
                    stats.append({"modality": m, "layer": layer_i, **refiner.last_stats})
        return stats

    def forward(self, x, sp_rows, sp_cols, sp_vals, n_spatial_edges,
                feat_tensors):
        """
        Parameters
        ----------
        x : list of per-modality tensors, each shaped (N, input_dim_m)
        sp_rows/cols/vals : spatial hypergraph sparse tensors
        n_spatial_edges : number of spatial edges
        feat_tensors : list of (ft_rows, ft_cols, ft_vals, n_feat_edges) per modality
        """
        if not isinstance(x, (list, tuple)):
            raise TypeError("DualBranchDHGNN.forward expects a list/tuple of per-modality tensors; "
                            "do not concatenate RNA and ATAC before the model.")
        if len(x) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modality tensors, got {len(x)}")
        x_raw = list(x)
        n_nodes = x_raw[0].shape[0]

        # ---- Per-modality encoding (no cross-omics concatenation at input) ----
        mod_h = []
        for i in range(self.n_modalities):
            mod_h.append(self.encoders[i](x_raw[i]))

        # ---- Per-modality dual HGNN message passing ----
        mod_embeddings = []
        mod_spatial_pre = []
        mod_feature_pre = []
        dynamic_feat_tensors = [None for _ in range(self.n_modalities)]

        for m in range(self.n_modalities):
            h_s = mod_h[m]
            h_f = mod_h[m]

            for layer_i in range(self.n_layers):
                # Spatial conv on shared H_spatial with optional HSL incidence refinement
                if self.use_hsl_spatial:
                    sp_vals_layer = self.spatial_refiners[m][layer_i](
                        h_s, sp_rows, sp_cols, sp_vals, n_nodes, n_spatial_edges
                    )
                else:
                    sp_vals_layer = sp_vals
                h_s_new = self.spatial_convs[m][layer_i](
                    h_s, sp_rows, sp_cols, sp_vals_layer, n_nodes, n_spatial_edges
                )
                # Feature conv on modality-specific static or dynamic H_feature
                if self.use_dynamic_feature:
                    ft_rows, ft_cols, ft_vals, n_feat_edges = self.dynamic_feature_builders[m](h_f)
                    dynamic_feat_tensors[m] = (ft_rows, ft_cols, ft_vals, n_feat_edges)
                else:
                    ft_rows, ft_cols, ft_vals, n_feat_edges = feat_tensors[m]
                h_f_new = self.feature_convs[m][layer_i](
                    h_f, ft_rows, ft_cols, ft_vals, n_nodes, n_feat_edges
                )

                if layer_i == self.n_layers - 1:
                    mod_spatial_pre.append(h_s_new)
                    mod_feature_pre.append(h_f_new)

                # Intra-modal fusion
                gate = torch.sigmoid(self.intra_gates[m])
                merged = gate * h_s_new + (1 - gate) * h_f_new
                h_s = self.intra_norms[m](merged)
                h_f = h_s  # share for next layer

            mod_embeddings.append(h_s)

        # ---- Cross-modal fusion ----
        gate = torch.sigmoid(self.cross_gate)
        fused = gate * mod_embeddings[0] + (1 - gate) * mod_embeddings[1]
        embedding = self.cross_norm(fused)

        dist = torch.sum(
            (embedding.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2
        )
        cluster_logits = -dist

        # Per-modality reconstruction
        mod_recons = [self.decoders[i](embedding) for i in range(self.n_modalities)]

        return {
            "embedding": embedding,
            "cluster_logits": cluster_logits,
            "mod_recons": mod_recons,
            "x_raw": x_raw,
            "mod_spatial_pre": mod_spatial_pre,
            "mod_feature_pre": mod_feature_pre,
            "feat_tensors": dynamic_feat_tensors if self.use_dynamic_feature else feat_tensors,
        }


# ====================================================================== #
#                          Loss Functions                                  #
# ====================================================================== #

def smoothness_loss_sparse(x, rows, cols, vals, n_nodes, n_edges):
    De = torch.zeros(n_edges, device=x.device)
    De.scatter_add_(0, cols, vals)
    De = De.clamp(min=1.0)
    gathered = x[rows] * vals.unsqueeze(1)
    edge_feats = torch.zeros(n_edges, x.shape[1], device=x.device)
    edge_feats.scatter_add_(0, cols.unsqueeze(1).expand_as(gathered), gathered)
    edge_feats = edge_feats / De.unsqueeze(1)
    Dv = torch.zeros(n_nodes, device=x.device)
    Dv.scatter_add_(0, rows, vals)
    Dv = Dv.clamp(min=1.0)
    gathered_e = edge_feats[cols] * vals.unsqueeze(1)
    x_recon = torch.zeros(n_nodes, x.shape[1], device=x.device)
    x_recon.scatter_add_(0, rows.unsqueeze(1).expand_as(gathered_e), gathered_e)
    x_recon = x_recon / Dv.unsqueeze(1)
    return F.mse_loss(x, x_recon)


def compute_total_loss(
    outputs, sp_rows, sp_cols, sp_vals,
    n_nodes, n_spatial_edges, input_dims,
    lambda_cluster=1.0, lambda_smooth=0.1,
    lambda_recon=0.5, lambda_contrast=0.0,
    dec_phase=True, temperature=0.5,
):
    embedding = outputs["embedding"]
    cluster_logits = outputs["cluster_logits"]

    # Per-modality reconstruction loss
    x_raw = outputs["x_raw"]
    recon_loss = torch.tensor(0.0, device=embedding.device)
    for i, _dim in enumerate(input_dims):
        recon_loss = recon_loss + F.mse_loss(outputs["mod_recons"][i], x_raw[i])

    # No supervised or semi-supervised objective is used in this unsupervised setting.
    contrast_loss = torch.tensor(0.0, device=embedding.device)

    # DEC KL loss
    if dec_phase:
        dist = -cluster_logits
        t_weights = 1.0 / (1.0 + dist).clamp(min=1e-8)
        q = t_weights / t_weights.sum(dim=1, keepdim=True)
        f = q.sum(dim=0, keepdim=True).clamp(min=1e-8)
        p = (q ** 2) / f
        p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_q = torch.log(q.clamp(min=1e-10))
        log_p = torch.log(p.clamp(min=1e-10))
        clust_loss = (p.detach() * (log_p - log_q)).sum(dim=1).mean()
    else:
        clust_loss = torch.tensor(0.0, device=embedding.device)

    # Smoothness on spatial hypergraph
    sm_spatial = smoothness_loss_sparse(
        embedding, sp_rows, sp_cols, sp_vals, n_nodes, n_spatial_edges
    )

    # Smoothness on each modality's feature hypergraph
    sm_feat_total = torch.tensor(0.0, device=embedding.device)
    for ft_rows, ft_cols, ft_vals, n_feat_edges in outputs["feat_tensors"]:
        sm_feat_total = sm_feat_total + smoothness_loss_sparse(
            embedding, ft_rows, ft_cols, ft_vals, n_nodes, n_feat_edges
        )

    total = (
        lambda_recon * recon_loss
        + lambda_cluster * clust_loss
        + lambda_smooth * (sm_spatial + sm_feat_total)
    )
    loss_dict = {
        "total": total.item(), "recon": recon_loss.item(),
        "contrast": contrast_loss.item(), "cluster": clust_loss.item(),
        "smooth_s": sm_spatial.item(), "smooth_f": sm_feat_total.item(),
    }
    return total, loss_dict
