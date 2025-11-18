#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
from dbsi_toolbox.utils import load_dwi_data_dipy, save_parameter_maps
from dbsi_toolbox.twostep import DBSI_TwoStep
from dbsi_toolbox.calibration import optimize_dbsi_params
from dbsi_toolbox.deep_learning import DBSI_DeepSolver

def parse_arguments():
    parser = argparse.ArgumentParser(description="DBSI Toolbox CLI: Run Standard or Deep Learning DBSI.")

    # --- INPUT / OUTPUT ---
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', '-i', required=True, help='Path to input NIfTI file (4D DWI)')
    io_group.add_argument('--bval', '-b', required=True, help='Path to .bval file')
    io_group.add_argument('--bvec', '-v', required=True, help='Path to .bvec file')
    io_group.add_argument('--mask', '-m', required=True, help='Path to brain mask NIfTI file')
    io_group.add_argument('--out_dir', '-o', required=True, help='Directory to save output maps')

    # --- METHOD SELECTION ---
    parser.add_argument('--method', choices=['twostep', 'dl'], required=True, 
                        help="Choose 'twostep' for Classic Optimized or 'dl' for Self-Supervised Deep Learning")
    parser.add_argument('--snr', type=float, default=30.0, help='Estimated SNR for calibration/noise injection (Default: 30)')

    # --- TWO-STEP SPECIFIC ---
    ts_group = parser.add_argument_group('Two-Step Parameters')
    ts_group.add_argument('--no_calibration', action='store_true', 
                          help='Skip Monte Carlo calibration and use manual bases/lambda')
    ts_group.add_argument('--manual_bases', type=int, default=20, help='Manual n_iso_bases (if calibration skipped)')
    ts_group.add_argument('--manual_lambda', type=float, default=0.01, help='Manual reg_lambda (if calibration skipped)')
    ts_group.add_argument('--mc_iter', type=int, default=500, help='Monte Carlo iterations for calibration (Default: 500)')

    # --- DEEP LEARNING SPECIFIC ---
    dl_group = parser.add_argument_group('Deep Learning Parameters')
    dl_group.add_argument('--epochs', type=int, default=100, help='Training epochs (Default: 100)')
    dl_group.add_argument('--batch_size', type=int, default=2048, help='Batch size (Default: 2048)')
    dl_group.add_argument('--lr', type=float, default=1e-3, help='Learning Rate (Default: 0.001)')
    dl_group.add_argument('--dl_bases', type=int, default=50, help='Isotropic bases resolution for DL (Default: 50)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Setup Output Directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"[CLI] Created output directory: {args.out_dir}")

    # 2. Load Data
    print(f"[CLI] Loading data: {args.input}")
    try:
        dwi_data, affine, gtab, mask = load_dwi_data_dipy(
            args.input, args.bval, args.bvec, args.mask
        )
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        sys.exit(1)

    real_bvals = gtab.bvals
    real_bvecs = gtab.bvecs
    print(f"[CLI] Protocol: {len(real_bvals)} volumes detected.")

    maps = {}

    # 3. Branch: TWO-STEP
    if args.method == 'twostep':
        print("\n=== Method: Two-Step DBSI ===")
        
        n_bases = args.manual_bases
        reg_lambda = args.manual_lambda

        if not args.no_calibration:
            print(f"[CLI] Running Auto-Calibration (SNR={args.snr}, Iter={args.mc_iter})...")
            best_params = optimize_dbsi_params(
                real_bvals, real_bvecs,
                snr_estimate=args.snr,
                n_monte_carlo=args.mc_iter,
                bases_grid=[20, 50, 75],
                lambdas_grid=[0.0, 0.01, 0.1, 0.5],
                verbose=True
            )
            n_bases = best_params['n_bases']
            reg_lambda = best_params['reg_lambda']
            print(f"[CLI] Calibration Result -> Bases: {n_bases}, Lambda: {reg_lambda}")
        else:
            print(f"[CLI] Using Manual Parameters -> Bases: {n_bases}, Lambda: {reg_lambda}")

        # Init & Fit
        model = DBSI_TwoStep(
            n_iso_bases=n_bases,
            reg_lambda=reg_lambda,
            iso_diffusivity_range=(0.0, 3.0e-3)
        )
        maps = model.fit_volume(dwi_data, real_bvals, real_bvecs, mask=mask)

    # 4. Branch: DEEP LEARNING
    elif args.method == 'dl':
        print("\n=== Method: Self-Supervised Deep Learning ===")
        print(f"[CLI] Config: Epochs={args.epochs}, Batch={args.batch_size}, Bases={args.dl_bases}")
        
        model = DBSI_DeepSolver(
            n_iso_bases=args.dl_bases,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            noise_injection_level=1.0/args.snr # Use SNR arg to determine noise injection
        )
        maps = model.fit_volume(dwi_data, real_bvals, real_bvecs, mask=mask)

    # 5. Save Results
    print(f"\n[CLI] Saving results to {args.out_dir}...")
    save_parameter_maps(maps, affine, output_dir=args.out_dir)
    print("[CLI] Done.")

if __name__ == "__main__":
    main()