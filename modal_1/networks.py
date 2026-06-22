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
    ):
        super().__init__()
        self.input_dims = input_dims
        self.n_modalities = len(input_dims)
        self.n_layers = n_layers

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

        # ---- Per-modality feature HGNNs (operate on modality-specific H_feature) ----
        self.feature_convs = nn.ModuleList()
        for _ in range(self.n_modalities):
            self.feature_convs.append(nn.ModuleList([
                SparseHGNNConv(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(n_layers)
            ]))

        # ---- Per-modality intra-modal fusion gates ----
        self.intra_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_modalities)
        ])
        self.intra_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.n_modalities)
        ])

        # ---- Cross-modal fusion ----
        self.cross_gate = nn.Parameter(torch.tensor(0.5))
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ---- Output heads ----
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.cluster_centers = Parameter(torch.randn(n_classes, hidden_dim) * 0.01)

        # ---- Per-modality decoders ----
        self.decoders = nn.ModuleList()
        for dim in input_dims:
            self.decoders.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            ))

    def forward(self, x, sp_rows, sp_cols, sp_vals, n_spatial_edges,
                feat_tensors):
        """
        Parameters
        ----------
        x : (N, sum(input_dims)) raw concatenated features
        sp_rows/cols/vals : spatial hypergraph sparse tensors
        n_spatial_edges : number of spatial edges
        feat_tensors : list of (ft_rows, ft_cols, ft_vals, n_feat_edges) per modality
        """
        x_raw = x
        n_nodes = x.shape[0]

        # ---- Per-modality encoding ----
        mod_h = []
        offset = 0
        for i in range(self.n_modalities):
            d = self.input_dims[i]
            mod_h.append(self.encoders[i](x[:, offset:offset + d]))
            offset += d

        # ---- Per-modality dual HGNN message passing ----
        mod_embeddings = []
        mod_spatial_pre = []
        mod_feature_pre = []

        for m in range(self.n_modalities):
            ft_rows, ft_cols, ft_vals, n_feat_edges = feat_tensors[m]
            h_s = mod_h[m]
            h_f = mod_h[m]

            for layer_i in range(self.n_layers):
                # Spatial conv on shared H_spatial
                h_s_new = self.spatial_convs[m][layer_i](
                    h_s, sp_rows, sp_cols, sp_vals, n_nodes, n_spatial_edges
                )
                # Feature conv on modality-specific H_feature
                h_f_new = self.feature_convs[m][layer_i](
                    h_f, ft_rows, ft_cols, ft_vals, n_nodes, n_feat_edges
                )

                if layer_i == self.n_layers - 1:
                    mod_spatial_pre.append(h_s_new)
                    mod_feature_pre.append(h_f_new)

                # Intra-modal fusion
                gate = self.intra_gates[m]
                merged = gate * h_s_new + (1 - gate) * h_f_new
                h_s = self.intra_norms[m](merged)
                h_f = h_s  # share for next layer

            mod_embeddings.append(h_s)

        # ---- Cross-modal fusion ----
        gate = self.cross_gate
        fused = gate * mod_embeddings[0] + (1 - gate) * mod_embeddings[1]
        embedding = self.cross_norm(fused)

        logits = self.classifier(embedding)

        dist = torch.sum(
            (embedding.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2
        )
        cluster_logits = -dist

        # Per-modality reconstruction
        mod_recons = [self.decoders[i](embedding) for i in range(self.n_modalities)]

        return {
            "logits": logits,
            "embedding": embedding,
            "cluster_logits": cluster_logits,
            "mod_recons": mod_recons,
            "x_raw": x_raw,
            "mod_spatial_pre": mod_spatial_pre,
            "mod_feature_pre": mod_feature_pre,
            "feat_tensors": feat_tensors,
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
    offset = 0
    for i, dim in enumerate(input_dims):
        recon_loss = recon_loss + F.mse_loss(outputs["mod_recons"][i], x_raw[:, offset:offset + dim])
        offset += dim

    # Contrastive loss (optional)
    contrast_loss = torch.tensor(0.0, device=embedding.device)
    if lambda_contrast > 0 and len(outputs["mod_spatial_pre"]) >= 2:
        x_s = F.normalize(outputs["mod_spatial_pre"][0], dim=1)
        x_f = F.normalize(outputs["mod_feature_pre"][0], dim=1)
        sim_sf = x_s @ x_f.T / temperature
        sim_ss = x_s @ x_s.T / temperature
        sim_ff = x_f @ x_f.T / temperature
        mask = ~torch.eye(n_nodes, dtype=torch.bool, device=embedding.device)
        pos_scores = torch.diag(sim_sf)
        log_sum_neg = torch.logsumexp(
            torch.cat([sim_sf, sim_ss.masked_fill(~mask, float('-inf')),
                       sim_ff.masked_fill(~mask, float('-inf'))], dim=1), dim=1
        )
        contrast_loss = (-pos_scores + log_sum_neg).mean()

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
        + lambda_contrast * contrast_loss
        + lambda_cluster * clust_loss
        + lambda_smooth * (sm_spatial + sm_feat_total)
    )
    loss_dict = {
        "total": total.item(), "recon": recon_loss.item(),
        "contrast": contrast_loss.item(), "cluster": clust_loss.item(),
        "smooth_s": sm_spatial.item(), "smooth_f": sm_feat_total.item(),
    }
    return total, loss_dict
