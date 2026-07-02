"""
Run the focused 72-configuration local refinement sweep for HSL-DvDHGNN.

This wrapper targets the stable neighborhood suggested by recent E18.5 mouse
brain runs: weak smoothness around 0.01, full 2129 spatial Delaunay-star edges,
DEC assignment-stability stopping enabled, and dynamic feature-edge adjustment
restricted to warmup.

Default 72-run grid:
    lambda_smooth:          0.005, 0.01, 0.02
    warmup_epochs:          80, 100
    edge_adjust_interval:   10, 20
    beta_saturation:        0.85, 0.90
    hsl_residual_strength:  0.3, 0.5, 0.7

Fixed defaults:
    max_spatial_edges:      2129
    lambda_recon:           0.5
    lambda_cluster:         1.0
    topk_edges:             3
    delta_edges:            20
    gamma_saturation:       0.98
    dropout:                0.3
    lr:                     0.001
    seed:                   42
    DEC stability:          patience=3, tol=0.005, min_dec_epochs=20

Total combinations:
    3 * 2 * 2 * 2 * 3 = 72

Usage:
    python -m modal_1.run_sweep_72_local_refine

Check the generated grid without training:
    python -m modal_1.run_sweep_72_local_refine --dry_run

You can still override any default by passing the same argument after the script
name. For example:
    python -m modal_1.run_sweep_72_local_refine --epochs 300 --output_csv my_sweep.csv
"""

from __future__ import annotations

import os
import sys


# Ensure modal_1 is importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.param_sweep import main as sweep_main


DEFAULT_SWEEP_ARGS = [
    "--output_csv", "sweep_72_local_refine.csv",
    "--max_spatial_edges", "2129",
    "--epochs", "500",
    "--patience", "50",
    "--dec_stability_patience", "3",
    "--dec_stability_tol", "0.005",
    "--dec_stability_min_epochs", "20",
    "--seeds", "42",
    "--lr", "0.001",
    "--dropout", "0.3",
    "--warmup_epochs", "80,100",
    "--lambda_recon", "0.5",
    "--lambda_cluster", "1.0",
    "--lambda_smooth", "0.005,0.01,0.02",
    "--topk_edges", "3",
    "--edge_adjust_interval", "10,20",
    "--delta_edges", "20",
    "--beta_saturation", "0.85,0.90",
    "--gamma_saturation", "0.98",
    "--hsl_residual_strength", "0.3,0.5,0.7",
]


def main():
    user_args = sys.argv[1:]
    # User-provided arguments come after defaults so argparse keeps the user's
    # value if the same option is specified twice.
    sys.argv = [sys.argv[0], *DEFAULT_SWEEP_ARGS, *user_args]
    sweep_main()


if __name__ == "__main__":
    main()
