# dbsi_toolbox/twostep.py

import numpy as np
from typing import Dict, Optional
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .common import DBSIParams

class DBSI_TwoStep(BaseDBSI):
    """
    Orchestrator class that implements the DBSI Two-Step approach:
    1. Basis Spectrum (NNLS) -> Find active compartments.
    2. Tensor Fit (NLLS) -> Refine parameters.
    """
    
    def __init__(self, 
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=20,
                 reg_lambda=0.01,
                 filter_threshold=0.01,
                 axial_diff_basis=1.5e-3,
                 radial_diff_basis=0.3e-3):
        
        # --- AGGIORNAMENTO CLASSI ---
        self.spectrum_model = DBSI_BasisSpectrum(  # Ex linear_model
            iso_diffusivity_range=iso_diffusivity_range,
            n_iso_bases=n_iso_bases,
            axial_diff_basis=axial_diff_basis,
            radial_diff_basis=radial_diff_basis,
            reg_lambda=reg_lambda,
            filter_threshold=filter_threshold
        )
        
        self.fitting_model = DBSI_TensorFit()      # Ex nonlinear_model
        
    def fit_volume(self, volume, bvals, bvecs, **kwargs):
        # ... (codice di setup bvals/bvecs invariato) ...
            
        print("Step 1/2: Pre-calculating Basis Spectrum Matrix...", end="")
        # Usa self.spectrum_model invece di self.linear_model
        self.spectrum_model.design_matrix = self.spectrum_model._build_design_matrix(flat_bvals, current_bvecs)
        
        self.spectrum_model.current_bvecs = current_bvecs
        self.fitting_model.current_bvecs = current_bvecs
        print(" Done.")
        
        print("Step 2/2: Running Two-Step DBSI...")
        return super().fit_volume(volume, bvals, bvecs, **kwargs)

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        # --- STEP 1: Basis Spectrum ---
        spectrum_result = self.spectrum_model.fit_voxel(signal, bvals)
        
        if spectrum_result.f_fiber == 0 and spectrum_result.f_iso_total == 0:
            return spectrum_result
            
        # --- STEP 2: Tensor Fit ---
        final_result = self.fitting_model.fit_voxel(signal, bvals, initial_guess=spectrum_result)
        
        return final_result