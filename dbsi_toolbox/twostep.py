# dbsi_toolbox/twostep.py

import numpy as np
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .common import DBSIParams

class DBSI_TwoStep(BaseDBSI):
    """
    Standard DBSI Implementation (Two-Step Approach).
    1. Linear Basis Spectrum (NNLS) -> Estimates fractions and directions.
    2. Non-Linear Tensor Fit (NLLS) -> Refines diffusivities (AD/RD).
    """
    
    def __init__(self, 
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=20,
                 reg_lambda=0.01,
                 filter_threshold=0.01,
                 axial_diff_basis=1.5e-3,
                 radial_diff_basis=0.3e-3):
        
        # Initialize the two internal models
        self.spectrum_model = DBSI_BasisSpectrum(
            iso_diffusivity_range=iso_diffusivity_range,
            n_iso_bases=n_iso_bases,
            axial_diff_basis=axial_diff_basis,
            radial_diff_basis=radial_diff_basis,
            reg_lambda=reg_lambda,
            filter_threshold=filter_threshold
        )
        
        self.fitting_model = DBSI_TensorFit()
        
    def fit_volume(self, volume, bvals, bvecs, **kwargs):
        """
        Initial setup before launching the voxel-wise loop defined in BaseDBSI.
        """
        print("[DBSI] Initializing Standard Two-Step Fit...")
        
        # Standardize Inputs
        flat_bvals = np.array(bvals).flatten()
        N = len(flat_bvals)
        
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T
        else:
            current_bvecs = bvecs
            
        # Pre-calculate the Design Matrix (only once for the entire volume)
        print(f"[DBSI] Pre-calculating Design Matrix ({self.spectrum_model.n_iso_bases} isotropic bases)...")
        self.spectrum_model.design_matrix = self.spectrum_model._build_design_matrix(flat_bvals, current_bvecs)
        
        # Share gradient vectors with sub-models
        self.spectrum_model.current_bvecs = current_bvecs
        self.fitting_model.current_bvecs = current_bvecs
        
        # Launch the voxel-by-voxel fit managed by BaseDBSI
        return super().fit_volume(volume, bvals, bvecs, **kwargs)

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        # --- STEP 1: Basis Spectrum (NNLS) ---
        # Quickly find fractions and principal directions using the linear solver
        spectrum_result = self.spectrum_model.fit_voxel(signal, bvals)
        
        # If the linear fit finds nothing or fails (e.g., empty voxel), return what we have
        if spectrum_result.f_fiber == 0 and spectrum_result.f_iso_total == 0:
            return spectrum_result
            
        # --- STEP 2: Tensor Fit (NLLS) ---
        # Use the spectrum result as an "initial guess" to refine parameters (specifically AD and RD)
        # Note: To speed up processing, this step could be skipped if standard AD/RD values are sufficient.
        try:
            final_result = self.fitting_model.fit_voxel(signal, bvals, initial_guess=spectrum_result)
            return final_result
        except Exception:
            # Fallback to linear result if non-linear optimization fails
            return spectrum_result