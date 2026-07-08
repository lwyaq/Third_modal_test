"""
Run the 48-configuration local refinement sweep around the best observed run34.

This wrapper is focused on the parameter neighborhood that produced the best
result in the previous 48-run sweep:

    warmup_epochs = 120
    lambda_recon = 0.45
    lambda_cluster = 4.0
    topk_edges = 8
    delta_edges = 50
    hsl_residual_strength = 0.3

New 48-run local grid:
    warmup_epochs:          110, 120, 130
    lambda_recon:           0.40, 0.45
    lambda_cluster:         3.5, 4.0
    lambda_smooth:          0.05
    topk_edges:             6, 8
    edge_adjust_interval:   10
    delta_edges:            40, 50
    beta_saturation:        0.90
    gamma_saturation:       0.60
    hsl_residual_strength:  0.3
    dropout:                0.3

Total combinations:
    3 * 2 * 2 * 1 * 2 * 1 * 2 * 1 * 1 * 1 * 1 = 48

Usage:
    python modal_1/run_sweep_48_refine_run34.py

Check the generated grid without training:
    python modal_1/run_sweep_48_refine_run34.py --dry_run

You can still override any default by passing the same argument after the script
name. For example:
    python modal_1/run_sweep_48_refine_run34.py --epochs 300 --output_csv my_sweep.csv
"""

from __future__ import annotations

import os
import sys


# Ensure modal_1 is importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.param_sweep import main as sweep_main


DEFAULT_SWEEP_ARGS = [
    "--warmup_epochs", "110,120,130",
    "--lambda_recon", "0.40,0.45",
    "--lambda_cluster", "3.5,4.0",
    "--lambda_smooth", "0.05",
    "--topk_edges", "6,8",
    "--edge_adjust_interval", "10",
    "--delta_edges", "40,50",
    "--beta_saturation", "0.90",
    "--gamma_saturation", "0.60",
    "--hsl_residual_strength", "0.3",
    "--dropout", "0.3",
    "--output_csv", "sweep_48_refine_around_run34.csv",
]


def main():
    user_args = sys.argv[1:]
    # User-provided arguments come after defaults so argparse keeps the user's
    # value if the same option is specified twice.
    sys.argv = [sys.argv[0], *DEFAULT_SWEEP_ARGS, *user_args]
    sweep_main()


if __name__ == "__main__":
    main()
