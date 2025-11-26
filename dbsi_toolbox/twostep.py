# dbsi_toolbox/twostep.py

import numpy as np
from tqdm import tqdm
import sys
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .numba_backend import build_design_matrix_numba, fit_volume_numba

class DBSI_TwoStep(BaseDBSI):
    """
    High-Performance DBSI solver.
    Uses Numba for parallelized volumetric fitting of the Basis Spectrum.
    """
    
    def __init__(self, 
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=20,
                 reg_lambda=0.01,
                 filter_threshold=0.01,
                 axial_diff_basis=1.5e-3,
                 radial_diff_basis=0.3e-3):
        
        self.iso_range = iso_diffusivity_range
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.filter_threshold = filter_threshold
        self.axial_diff = axial_diff_basis
        self.radial_diff = radial_diff_basis
        
        # Mantieni compatibilità con vecchie chiamate
        self.spectrum_model = DBSI_BasisSpectrum(
            iso_diffusivity_range, n_iso_bases, axial_diff_basis, radial_diff_basis, reg_lambda
        )

    def fit_volume(self, volume, bvals, bvecs, mask=None, show_progress=True, **kwargs):
        """
        Esegue il fit su tutto il volume usando il backend accelerato Numba.
        Ignora il ciclo lento di BaseDBSI.
        """
        X, Y, Z, N = volume.shape
        n_voxels = X * Y * Z
        
        # 1. Preparazione Dati (Flattening)
        print(f"[DBSI-Fast] Preparing data for hardware acceleration...")
        data_flat = volume.reshape(n_voxels, N).astype(np.float64)
        
        if mask is not None:
            mask_flat = mask.flatten().astype(bool)
        else:
            # Auto-masking semplice se non fornita
            mask_flat = np.any(data_flat > 0, axis=1)
            
        flat_bvals = np.array(bvals).flatten().astype(np.float64)
        
        # Gestione bvecs (N, 3)
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T.astype(np.float64)
        else:
            current_bvecs = bvecs.astype(np.float64)

        # 2. Costruzione Matrice di Design (Numba)
        print(f"[DBSI-Fast] Building Design Matrix ({len(flat_bvals)} meas x {len(flat_bvals) + self.n_iso_bases} bases)...")
        
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        # Definisci soglie per categorizzare le frazioni isotrope
        # Restricted: <= 0.3 | Hindered: 0.3 < D <= 2.0 | Water: > 2.0
        # Troviamo gli indici nello spettro isotropo
        idx_res_end = np.sum(iso_diffs <= 0.3e-3)
        idx_hin_end = np.sum(iso_diffs <= 2.0e-3)
        
        design_matrix = build_design_matrix_numba(
            flat_bvals, 
            current_bvecs, 
            iso_diffs, 
            self.axial_diff, 
            self.radial_diff
        )
        
        # Salva per visualizzazione esterna (come nel tuo script)
        self.spectrum_model.design_matrix = design_matrix 

        # 3. Fitting Parallelo (Numba)
        print(f"[DBSI-Fast] Fitting {np.sum(mask_flat)} voxels using Numba Parallel backend...")
        
        # Questa funzione usa tutti i core della CPU
        raw_results = fit_volume_numba(
            data_flat, 
            flat_bvals, 
            design_matrix, 
            self.reg_lambda, 
            mask_flat,
            len(current_bvecs), # n_aniso
            idx_res_end,
            idx_hin_end
        )
        
        # 4. Ricostruzione Mappe 3D
        print(f"[DBSI-Fast] Reconstructing parameter maps...")
        
        maps = {
            'fiber_fraction': raw_results[:, 0].reshape(X, Y, Z),
            'restricted_fraction': raw_results[:, 1].reshape(X, Y, Z),
            'hindered_fraction': raw_results[:, 2].reshape(X, Y, Z),
            'water_fraction': raw_results[:, 3].reshape(X, Y, Z),
            'r_squared': raw_results[:, 4].reshape(X, Y, Z),
            # Placeholder per diffusività (usiamo quelle standard se non facciamo step 2 non-lineare)
            'axial_diffusivity': np.full((X, Y, Z), self.axial_diff),
            'radial_diffusivity': np.full((X, Y, Z), self.radial_diff),
        }
        
        return maps