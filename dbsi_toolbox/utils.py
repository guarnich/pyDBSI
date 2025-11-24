# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple
from scipy.ndimage import binary_dilation

# Import DIPY functions for compatibility
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable
except (ImportError, AttributeError):
    print("WARNING: DIPY not found. Some utility functions may not work.")
    print("Install with: pip install dipy")
    
    # Create a dummy type to avoid import errors if dipy isn't present
    GradientTable = type("GradientTable", (object,), {})


def load_dwi_data_dipy(
    f_nifti: str, 
    f_bval: str, 
    f_bvec: str, 
    f_mask: str 
) -> Tuple[np.ndarray, np.ndarray, 'GradientTable', np.ndarray]: #type: ignore
    """
    Loads DWI data, bvals, bvecs, and the MANDATORY brain mask using DIPY.
    
    Args:
        f_nifti: Path to the 4D NIfTI file.
        f_bval: Path to the .bval file.
        f_bvec: Path to the .bvec file.
        f_mask: Path to the 3D NIfTI mask file (REQUIRED).
        
    Returns:
        A tuple containing:
        - data (np.ndarray): 4D DWI data
        - affine (np.ndarray): Affine matrix
        - gtab (GradientTable): DIPY gradient table object
        - mask (np.ndarray): 3D boolean mask
        
    Raises:
        ValueError: If f_mask is not provided or file logic fails.
    """
    # --- 1. STRICT INPUT CHECK (BLOCKING) ---
    if not f_mask:
        raise ValueError(
            "\n[CRITICAL ERROR] Brain Mask is MISSING.\n"
            "The DBSI pipeline requires a valid brain mask to accurately estimate SNR "
            "and perform voxel-wise fitting.\n"
            "Execution stopped. Please provide a mask using the --mask argument."
        )

    print(f"[Utils] Loading data from: {f_nifti}")
    data, affine = load_nifti(f_nifti)
    
    print(f"[Utils] Loading bvals/bvecs from: {f_bval}, {f_bvec}")
    bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec)
    
    # Pass 'bvecs' as keyword argument to avoid Warning
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    print(f"  ✓ Volume: {data.shape}, Bvals: {len(gtab.bvals)}, Bvecs: {gtab.bvecs.shape}")
    print(f"  ✓ No. of b=0 volumes: {np.sum(gtab.b0s_mask)}")
    
    print(f"[Utils] Loading MANDATORY mask from: {f_mask}")
    mask_data, mask_affine = load_nifti(f_mask)
    mask_data = mask_data.astype(bool)
    
    # Dimensionality Validation
    if mask_data.shape != data.shape[:3]:
        raise ValueError(
            f"Mask shape {mask_data.shape} does not match "
            f"data shape {data.shape[:3]}"
        )
    print(f"  ✓ Mask loaded successfully: {np.sum(mask_data):,} voxels.")
    
    return data, affine, gtab, mask_data


def _get_signal_bbox(data: np.ndarray, margin: int = 20) -> Tuple[slice, slice, slice]:
    """
    Calculates a bounding box around non-zero signal (Automatic Signal ROI).
    Adds a safety margin to ensure background air is included for noise estimation.
    """
    # Identify all pixels with signal > 0 (The Head/Object)
    coords = np.array(np.nonzero(data))
    
    if coords.size == 0:
        # Fallback to full image if data is empty
        return (slice(None), slice(None), slice(None))

    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1 # slice exclusive
    
    # Expand by margin (e.g. 20 voxels) to keep air around the head
    min_coords = np.maximum(0, min_coords - margin)
    max_coords = np.minimum(data.shape, max_coords + margin)
    
    return tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))


def estimate_snr(
    data: np.ndarray, 
    gtab: 'GradientTable', #type: ignore
    affine: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Estimates SNR (Signal-to-Noise Ratio) using b=0 images on an optimized FOV.
    
    Strategy:
    1. ROI Optimization: Computes a bounding box around non-zero signal (+ margin)
       to exclude far-field zeros/artifacts (Automatic Signal ROI).
    2. If >= 3 b=0 volumes: Use 'temporal' method (Voxel-wise Mean/Std).
    3. If < 3 b=0 volumes: Use 'spatial' method (Signal ROI / Background Noise).
       Spatial method expands the mask by ~15mm to exclude scalp/eyes from background.
    
    Args:
        data: 4D DWI volume (X, Y, Z, N)
        gtab: DIPY GradientTable
        affine: 4x4 Affine matrix to determine voxel size in mm
        mask: 3D binary mask of the brain (MANDATORY)
        
    Returns:
        float: Estimated SNR.
    """
    print("\n[Utils] Automatically estimating SNR...")
    
    # STRICT CHECK
    if mask is None:
        raise ValueError("[CRITICAL] SNR estimation stopped: No brain mask provided.")

    # 1. Extract b=0 volumes
    b0_mask = gtab.b0s_mask
    b0_data = data[..., b0_mask]
    n_b0 = b0_data.shape[-1]
    
    if n_b0 == 0:
        print("  ! WARNING: No b=0 volumes found. Returning default SNR = 30.0")
        return 30.0

    # 2. Calculate Mean b0 for ROI definition
    mean_b0_full = np.mean(b0_data, axis=-1)

    # 3. Define Automatic Signal FOV (Bounding Box of non-zero signal + margin)
    # We exclude pure zeros (padding) but keep air around the head for noise estimation.
    bbox = _get_signal_bbox(mean_b0_full, margin=20)
    
    # Crop data to this optimized FOV
    b0_data_cropped = b0_data[bbox]
    mask_cropped = mask[bbox]
    mean_b0_cropped = mean_b0_full[bbox]
    
    vol_reduction = 100 * (1 - (b0_data_cropped.size / b0_data.size))
    print(f"  ✓ Signal ROI Optimization: Cropped FOV to valid signal area (Volume reduced by {vol_reduction:.1f}%).")

    snr_est = 0.0

    # --- METHOD 1: Temporal SNR (Preferred if enough b0s) ---
    if n_b0 >= 3:
        print(f"  ✓ Method: Temporal (based on {n_b0} b=0 volumes)")
        
        # Calculate voxel-wise stats on cropped data
        std_b0 = np.std(b0_data_cropped, axis=-1)
        
        # Avoid division by zero
        std_b0[std_b0 == 0] = 1e-10
        
        # Voxel-wise SNR
        snr_map = mean_b0_cropped / std_b0
        
        # Median SNR inside the brain mask
        if np.sum(mask_cropped) > 0:
            snr_est = np.median(snr_map[mask_cropped])
            print(f"  ✓ Calculated SNR (Voxel-wise Median): {snr_est:.2f}")
        else:
            print("  ! Warning: Cropped mask is empty. Defaulting to 30.0")
            snr_est = 30.0

    # --- METHOD 2: Spatial SNR (Signal/Background) ---
    else:
        print(f"  ✓ Method: Spatial (few b0s available: {n_b0})")
        
        # Calculate voxel dimensions (mm)
        voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        mean_vox_dim = np.mean(voxel_sizes)
        
        # Dilation Strategy: 15mm to exclude scalp/eyes/fat
        target_dist_mm = 15.0
        n_iter = int(np.ceil(target_dist_mm / mean_vox_dim))
        
        print(f"  ✓ Expanding mask by {target_dist_mm}mm ({n_iter} voxels) to identify safe background...")
        
        # Create Dilated Mask (Brain + Scalp + Margin) within the cropped FOV
        dilated_mask = binary_dilation(mask_cropped, iterations=n_iter)
        
        # Background is everything OUTSIDE the dilated mask (True Air)
        # inside our Optimized Signal FOV
        background_mask = ~dilated_mask
        
        # Signal: Mean intensity inside the ORIGINAL brain mask
        signal_mean = np.mean(mean_b0_cropped[mask_cropped])
        
        # Noise: Standard deviation in the SAFE background (Air)
        noise_data = mean_b0_cropped[background_mask]
        
        # Remove artifacts (zero padding/NaNs) if any remain
        noise_data = noise_data[noise_data > 0] 
        
        if len(noise_data) < 100:
             print("  ! CRITICAL: Optimized FOV has insufficient background for noise estimation.")
             print("  ! Returning default SNR = 30.0")
             return 30.0
             
        noise_std = np.std(noise_data)
        
        # Correction for Rician/Rayleigh noise in magnitude images
        # Real_SD = Background_SD / 0.655
        noise_std_corrected = noise_std / 0.655
        
        if noise_std_corrected == 0:
            snr_est = 30.0
        else:
            snr_est = signal_mean / noise_std_corrected
            
        print(f"  ✓ Mean Signal (Brain): {signal_mean:.2f}")
        print(f"  ✓ Noise Std (Air, corrected): {noise_std_corrected:.2f}")
        print(f"  ✓ Calculated SNR: {snr_est:.2f}")

    # Safety limits (Sanity Check)
    if snr_est < 5.0:
        print("  ! Very low SNR detected (<5). Might be an error. Clamping to 5.0.")
        snr_est = 5.0
    elif snr_est > 100.0:
        print("  ! Very high SNR detected (>100). Possible synthetic data. Clamping to 100.0.")
        snr_est = 100.0
        
    return float(snr_est)


def save_parameter_maps(
    param_maps: dict, 
    affine: np.ndarray, 
    output_dir: str, 
    prefix: str = 'dbsi'
):
    """
    Saves parameter maps as NIfTI files.
    
    Args:
        param_maps: Dictionary with the parameter maps
        affine: Affine matrix from the original volume
        output_dir: Output directory
        prefix: Prefix for the output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    print(f"\n[Utils] Saving {len(param_maps)} maps to: {output_dir}")
    
    for param_name, param_data in param_maps.items():
        try:
            # Ensure data is float32 for saving
            img = nib.Nifti1Image(param_data.astype(np.float32), affine)
            filename = os.path.join(output_dir, f'{prefix}_{param_name}.nii.gz')
            nib.save(img, filename)
            saved_count += 1
        except Exception as e:
            print(f"  ! Error saving {param_name}: {e}")
    
    print(f"  ✓ Saved {saved_count} parameter maps.")