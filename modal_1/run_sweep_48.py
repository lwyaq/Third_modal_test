"""
Run the recommended 48-configuration parameter sweep for HSL-DvDHGNN.

This is a convenience wrapper around ``modal_1.param_sweep`` so you do not need
to type the long 48-run command every time.

Default 48-run grid:
    warmup_epochs:          100, 120
    lambda_recon:           0.45, 0.5
    lambda_cluster:         2.0, 3.0, 4.0
    lambda_smooth:          0.05
    topk_edges:             8, 12
    edge_adjust_interval:   10
    delta_edges:            30, 50
    beta_saturation:        0.90
    gamma_saturation:       0.60
    hsl_residual_strength:  0.3
    dropout:                0.3

Total combinations:
    2 * 2 * 3 * 1 * 2 * 1 * 2 * 1 * 1 * 1 * 1 = 48

Usage:
    python modal_1/run_sweep_48.py

Check the generated grid without training:
    python modal_1/run_sweep_48.py --dry_run

You can still override any default by passing the same argument after the script
name. For example:
    python modal_1/run_sweep_48.py --epochs 300 --output_csv my_sweep.csv
"""

from __future__ import annotations

import os
import sys


# Ensure modal_1 is importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_1.param_sweep import main as sweep_main


DEFAULT_SWEEP_ARGS = [
    "--warmup_epochs", "100,120",
    "--lambda_recon", "0.45,0.5",
    "--lambda_cluster", "2.0,3.0,4.0",
    "--lambda_smooth", "0.05",
    "--topk_edges", "8,12",
    "--edge_adjust_interval", "10",
    "--delta_edges", "30,50",
    "--beta_saturation", "0.90",
    "--gamma_saturation", "0.60",
    "--hsl_residual_strength", "0.3",
    "--dropout", "0.3",
    "--output_csv", "sweep_48.csv",
]


def main():
    user_args = sys.argv[1:]
    # User-provided arguments come after defaults so argparse keeps the user's
    # value if the same option is specified twice.
    sys.argv = [sys.argv[0], *DEFAULT_SWEEP_ARGS, *user_args]
    sweep_main()


if __name__ == "__main__":
    main()
