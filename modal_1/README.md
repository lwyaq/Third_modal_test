# HSL-DvDHGNN (`modal_1`)

This folder contains the current unsupervised spatial multi-omics model implementation.
The model is designed for paired RNA/ATAC features and spatial coordinates without
concatenating the two omics inputs before the model.

## Current model flow

```text
RNA features X_RNA          ATAC features X_ATAC          spatial coordinates C
        |                            |                              |
        v                            v                              v
  Encoder_RNA                  Encoder_ATAC              Delaunay-star H_spatial^0
        |                            |                              |
        +------------+---------------+------------------------------+
                     |
                     v
        Per-modality dual-branch hypergraph learning

For each modality m in {RNA, ATAC}:
  1. Spatial branch:
     - Use the fixed Delaunay-star spatial topology.
     - Apply HSLSpatialRefiner to learn incidence weights only.
     - Do not add or remove spatial hyperedges.

  2. Feature branch:
     - Initialize dynamic feature hyperedge prototypes from that modality's
       biological feature distribution.
     - Build a dynamic feature hypergraph with top-k node-to-hyperedge attention.
     - Optionally adjust feature hyperedge count using saturation-guided pruning
       and expansion.

  3. Intra-modal fusion:
     Z_m = sigmoid(alpha_m) * Z_m_spatial + (1 - sigmoid(alpha_m)) * Z_m_feature

Cross-modal fusion:
  Z = sigmoid(beta) * Z_RNA + (1 - sigmoid(beta)) * Z_ATAC

Unsupervised optimization:
  reconstruction loss + DEC cluster loss + spatial/feature smoothness loss
```

## Important implementation details

### No RNA/ATAC input concatenation

The trainer keeps RNA and ATAC features as a list of per-modality arrays. During
training, the model receives a list of tensors:

```python
[RNA_tensor, ATAC_tensor]
```

`DualBranchDHGNN.forward()` raises an error if a concatenated tensor is passed.
This is intentional: each modality must be encoded independently.

### HSL spatial refinement

`HSLSpatialRefiner` refines only the incidence values of the fixed spatial
hypergraph:

```text
H_spatial_refined = H_spatial^0 ⊙ G_theta(Z_m)
```

The spatial topology is not reconstructed during training.

### Dynamic biological feature hypergraphs

`VariableBioDynamicFeatureHypergraph` initializes feature hyperedge prototypes
from the corresponding preprocessed modality features (RNA PCA / ATAC LSI).
The initial number of feature hyperedges is the cell count (`M0 = N`), because
each cell's preprocessed feature vector is used as one candidate biological
prototype. The raw prototypes remain in the biological feature space and are
encoded into `hidden_dim` only when attention is computed; they are not sampled
from hidden embeddings or random Gaussian hyperedges.

### Unsupervised objective

The current loss is unsupervised:

```text
L = lambda_recon * L_recon
  + lambda_cluster * L_DEC
  + lambda_smooth * (L_smooth_spatial + L_smooth_feature)
```

There is no supervised or semi-supervised classifier loss.

### Early stopping

Model selection and early stopping use total training loss improvement, not ARI.
ARI/NMI are only reported when labels are available.

## Running experiment variants

Run commands from the repository root.

### Version 1: HSL spatial + static feature hypergraph

```bash
python -m modal_1.run \
  --use_hsl_spatial \
  --no-use_dynamic_feature
```

### Version 2: HSL spatial + dynamic feature hypergraph with fixed edge count

```bash
python -m modal_1.run \
  --use_hsl_spatial \
  --use_dynamic_feature \
  --edge_adjust_interval 0
```

### Version 3: HSL spatial + dynamic feature hypergraph with pruning only

```bash
python -m modal_1.run \
  --use_hsl_spatial \
  --use_dynamic_feature \
  --edge_adjust_interval 10 \
  --delta_edges 20 \
  --beta_saturation 0.90 \
  --gamma_saturation 0.98 \
  --no-allow_edge_add
```

### Version 4: HSL spatial + dynamic feature hypergraph with add/prune

```bash
python -m modal_1.run \
  --use_hsl_spatial \
  --use_dynamic_feature \
  --edge_adjust_interval 10 \
  --delta_edges 20 \
  --beta_saturation 0.90 \
  --gamma_saturation 0.98 \
  --allow_edge_add
```

By default, dynamic feature hyperedge add/prune runs only during warmup.  After
DEC is initialized, the learned biological hyperedge set is frozen so the DEC
cluster refinement phase does not keep changing the feature-hypergraph topology.
Use `--no-freeze_edges_after_warmup` only when you explicitly want the older
behavior where dynamic edge adjustment continues inside DEC.

### Parameter sweep

Use `param_sweep.py` to load/preprocess the data once and run a comma-separated
grid of parameter values. The script writes one row per run to `sweep_results.csv`
by default.

```bash
python -m modal_1.param_sweep \
  --warmup_epochs 120 \
  --lambda_recon 0.5,0.4 \
  --lambda_cluster 3.0,2.0 \
  --lambda_smooth 0.05 \
  --topk_edges 12 \
  --edge_adjust_interval 10 \
  --delta_edges 50 \
  --beta_saturation 0.90 \
  --gamma_saturation 0.60 \
  --hsl_residual_strength 0.3,0.5 \
  --output_csv sweep_results.csv
```

For a quick check of the generated grid without training, add `--dry_run`. When
labels are available, the CSV includes both the final metrics selected by
unsupervised loss and label-based `best_observed_ari`/`best_observed_epoch` for
debugging only.

### Recommended 48-run sweep shortcut

`run_sweep_48.py` runs the recommended 48-combination sweep without typing the
long command:

```bash
python -m modal_1.run_sweep_48
```

To check the 48 generated configurations without training:

```bash
python -m modal_1.run_sweep_48 --dry_run
```

The shortcut still accepts overrides, for example:

```bash
python -m modal_1.run_sweep_48 --epochs 300 --output_csv my_sweep.csv
```

### Refined 48-run sweep around run34

`run_sweep_48_refine_run34.py` runs a focused 48-combination local search around
the best setting observed in the previous sweep (`warmup=120`,
`lambda_recon=0.45`, `lambda_cluster=4.0`, `topk_edges=8`, `delta_edges=50`):

```bash
python -m modal_1.run_sweep_48_refine_run34
```

To check the generated configurations without training:

```bash
python -m modal_1.run_sweep_48_refine_run34 --dry_run
```

## Key CLI arguments

| Argument | Meaning |
| --- | --- |
| `--use_hsl_spatial` / `--no-use_hsl_spatial` | Enable/disable HSL refinement of spatial incidence weights. |
| `--use_dynamic_feature` / `--no-use_dynamic_feature` | Enable/disable dynamic biological feature hypergraphs. |
| `--edge_adjust_interval` | Epoch interval for feature hyperedge count adjustment; use `0` for fixed M. |
| `--delta_edges` | Number of feature hyperedges to add/prune at each adjustment. |
| `--beta_saturation` | Prune threshold. If saturation is below this value, redundant edges are removed. |
| `--gamma_saturation` | Add threshold. If saturation is above this value, capacity can be expanded. |
| `--topk_edges` | Number of feature hyperedges each node connects to during dynamic construction. |
| `--min_edges` | Minimum number of feature hyperedges after pruning. |
| `--allow_edge_add` / `--no-allow_edge_add` | Allow or forbid feature hyperedge expansion. |
| `--freeze_edges_after_warmup` / `--no-freeze_edges_after_warmup` | Freeze dynamic feature hyperedge add/prune after DEC starts; enabled by default. |
| `--hsl_residual_strength` | Residual strength for HSL incidence refinement. |
| `--warmup_epochs` | Number of reconstruction/smoothness warmup epochs before DEC cluster loss starts. |
| `--lambda_recon` | Weight for reconstruction loss. |
| `--lambda_cluster` | Weight for DEC cluster loss after warmup. |
| `--lambda_smooth` | Weight for spatial/feature smoothness loss. |
| `modal_1/param_sweep.py --output_csv` | CSV file receiving one result row per parameter combination. |
| `modal_1/param_sweep.py --max_runs` | Limit the sweep to the first N grid combinations; `0` means all. |
| `modal_1/param_sweep.py --dry_run` | Print parameter combinations without training. |

## Notes

- Dynamic feature hypergraphs are currently modality-level, not layer-level, to
  keep computation manageable.
- Spatial topology is always built outside the network and remains fixed during
  training; HSL only changes the weights.
- If `use_dynamic_feature=True`, static feature hypergraph construction is
  skipped and feature hypergraphs are generated inside the model during forward.
