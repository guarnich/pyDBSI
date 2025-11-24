# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple
from scipy.ndimage import binary_erosion # Usato solo per pulire il background

# Import DIPY functions
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable
    from dipy.segment.mask import median_otsu
except (ImportError, AttributeError):
    print("WARNING: DIPY not found. Some utility functions may not work.")
    GradientTable = type("GradientTable", (object,), {})

def load_dwi_data_dipy(
    f_nifti: str, 
    f_bval: str, 
    f_bvec: str, 
    f_mask: str 
) -> Tuple[np.ndarray, np.ndarray, 'GradientTable', np.ndarray]:
    """
    Loads DWI data, bvals, bvecs, and the MANDATORY brain mask.
    """
    # --- STRICT INPUT CHECK ---
    if not f_mask:
        raise ValueError("\n[CRITICAL] Brain Mask is MISSING. Please provide it.")

    print(f"[Utils] Loading data from: {f_nifti}")
    data, affine = load_nifti(f_nifti)
    
    print(f"[Utils] Loading bvals/bvecs...")
    bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    print(f"[Utils] Loading mask...")
    mask_data, _ = load_nifti(f_mask)
    mask_data = mask_data.astype(bool)
    
    if mask_data.shape != data.shape[:3]:
        raise ValueError(f"Mask shape {mask_data.shape} mismatch data {data.shape[:3]}")
    
    return data, affine, gtab, mask_data

def estimate_snr(
    data: np.ndarray, 
    gtab: 'GradientTable', 
    affine: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Estimates SNR using a Robust & Simple approach.
    
    Method 1 (Temporal): If >= 3 b0s, uses voxel-wise temporal stability.
    Method 2 (Spatial): Uses Median(Brain) / Std(Background_Air).
    """
    print("\n[Utils] Estimating SNR (Robust Method)...")
    
    # 1. Extract b0 volumes
    b0_data = data[..., gtab.b0s_mask]
    n_b0 = b0_data.shape[-1]
    
    if n_b0 == 0:
        print("  ! No b=0 volumes. Defaulting to SNR=30.0")
        return 30.0

    snr_est = 0.0

    # --- METHOD 1: TEMPORAL (Best if possible) ---
    if n_b0 >= 3:
        print(f"  ✓ Method: Temporal ({n_b0} volumes)")
        mean_b0 = np.mean(b0_data, axis=-1)
        std_b0 = np.std(b0_data, axis=-1)
        std_b0[std_b0 == 0] = 1e-10
        
        # SNR Map
        snr_map = mean_b0 / std_b0
        
        # Robust Metric: Median SNR inside the Brain Mask
        if np.sum(mask) > 0:
            snr_est = np.median(snr_map[mask])
            print(f"  ✓ Median Voxel-wise SNR: {snr_est:.2f}")
        else:
            snr_est = 30.0

    # --- METHOD 2: SPATIAL (Simple & Robust) ---
    else:
        print(f"  ✓ Method: Spatial (Background Statistics)")
        mean_b0 = np.mean(b0_data, axis=-1)
        
        # A. SIGNAL: Median of the Brain (Robust to lesions/CSF)
        if np.sum(mask) == 0:
            return 30.0
        signal_val = np.median(mean_b0[mask])
        
        # B. NOISE: Automatic Background Detection (Intensity-based)
        # Use Otsu to find a threshold that separates "Signal" (Head) from "Noise" (Air)
        # We don't care about anatomy here, just intensity statistics.
        otsu_thresh, _ = median_otsu(mean_b0, median_radius=2, numpass=1)
        
        # Define Background: Everything well below the signal threshold
        # Using 0.5 * Otsu ensures we stay deep in the noise floor
        noise_thresh = otsu_thresh * 0.5
        background_mask = mean_b0 < noise_thresh
        
        # Optional: Erode background to stay away from scalp edges/ghosting
        background_mask = binary_erosion(background_mask, iterations=1)
        
        # Calculate Noise Std
        noise_vals = mean_b0[background_mask]
        if len(noise_vals) > 100:
            noise_std = np.std(noise_vals)
            # Rician Correction
            noise_corr = noise_std / 0.655
            
            if noise_corr > 0:
                snr_est = signal_val / noise_corr
            else:
                snr_est = 30.0
        else:
            print("  ! Not enough background voxels found.")
            snr_est = 30.0
            
        print(f"  ✓ Signal (Median Brain): {signal_val:.2f}")
        print(f"  ✓ Noise (Std Background): {noise_corr:.2f}")
        print(f"  ✓ Estimated SNR: {snr_est:.2f}")

    # Safety Clamping
    snr_est = np.clip(snr_est, 5.0, 100.0)
    return float(snr_est)

def save_parameter_maps(param_maps, affine, output_dir, prefix='dbsi'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Utils] Saving {len(param_maps)} maps to: {output_dir}")
    for k, v in param_maps.items():
        try:
            nib.save(nib.Nifti1Image(v.astype(np.float32), affine), 
                     os.path.join(output_dir, f'{prefix}_{k}.nii.gz'))
        except Exception as e:
            print(f"  ! Error saving {k}: {e}")
    print("  ✓ Done.")