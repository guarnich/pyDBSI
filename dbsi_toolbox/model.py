# dbsi_toolbox/model.py

import numpy as np
from scipy.optimize import nnls
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from tqdm import tqdm


@dataclass
class DBSIParams:
    """
    Dataclass for holding DBSI results for a single voxel.
    Aggregates weights from the spectral solution into clinical metrics.
    """
    # Isotropic Fractions (derived from the spectrum)
    f_restricted: float      # Restricted fraction (cells/inflammation, D <= 0.3)
    f_hindered: float        # Hindered fraction (extracellular space/edema, 0.3 < D <= 2.0)
    f_water: float           # Free water fraction (CSF, D > 2.0)
    
    # Anisotropic Fractions
    f_fiber: float           # Total anisotropic fraction (sum of all fiber weights)
    
    # Fiber Properties
    fiber_dir: np.ndarray    # Dominant fiber direction vector (x, y, z)
    axial_diffusivity: float # Axial diffusivity of the basis tensor (fixed in this implementation)
    radial_diffusivity: float # Radial diffusivity of the basis tensor (fixed in this implementation)
    
    # Fitting Quality
    r_squared: float         # Coefficient of determination (R^2)

    @property
    def f_iso_total(self) -> float:
        """Total isotropic fraction."""
        return self.f_restricted + self.f_hindered + self.f_water

    @property
    def fiber_density(self) -> float:
        """Normalized fiber density (Fiber Fraction)."""
        total = self.f_fiber + self.f_iso_total
        return self.f_fiber / total if total > 0 else 0.0

class DBSIModel:
    """
    Implementation of Diffusion Basis Spectrum Imaging (DBSI) using 
    Non-Negative Least Squares (NNLS).
    
    The model solves the equation: S = A * x
    Where:
        S = Measured signal vector
        A = Design Matrix (Dictionary of basis functions)
        x = Weights (fractions) to be solved (x >= 0)
    """
    
    def __init__(self, 
                 bvals: np.ndarray, 
                 bvecs: np.ndarray, 
                 iso_diffusivity_range: Tuple[float, float] = (0.0, 3.0e-3),
                 n_iso_bases: int = 20,
                 axial_diff_basis: float = 1.5e-3,
                 radial_diff_basis: float = 0.3e-3):
        """
        Initialize the DBSI Model and pre-calculate the Design Matrix.

        Args:
            bvals: Array of b-values (N,) in s/mm^2.
            bvecs: Array of gradient directions (N, 3). Normalized.
            iso_diffusivity_range: Min and Max diffusivity for the isotropic spectrum (mm^2/s).
            n_iso_bases: Number of isotropic basis tensors to generate.
            axial_diff_basis: Assumed axial diffusivity for the fiber basis functions (mm^2/s).
            radial_diff_basis: Assumed radial diffusivity for the fiber basis functions (mm^2/s).
        """
        # Ensure inputs are numpy arrays
        self.bvals = np.array(bvals)
        self.bvecs = np.array(bvecs)
        
        # Validate dimensions
        if self.bvals.shape[0] != self.bvecs.shape[0]:
            raise ValueError("bvals and bvecs must have the same length.")

        # 1. Generate Isotropic Basis (The Spectrum)
        # Creates a range of diffusivities from Restricted (0) to Free Water (3.0)
        self.iso_diffusivities = np.linspace(
            iso_diffusivity_range[0], 
            iso_diffusivity_range[1], 
            n_iso_bases
        )
        
        # 2. Fiber Basis Parameters
        # In standard DBSI, we use fixed diffusivities for the basis set.
        # The NNLS solver determines the weight of these fibers.
        self.axial_diff_basis = axial_diff_basis
        self.radial_diff_basis = radial_diff_basis
        
        # 3. Pre-calculate Design Matrix A
        print("Building DBSI Design Matrix...", end="")
        self.design_matrix = self._build_design_matrix()
        print(" Done.")
        
    def _build_design_matrix(self) -> np.ndarray:
        """
        Constructs the dictionary matrix A.
        Rows: Measurements (N_bvals)
        Cols: Basis functions (N_aniso_directions + N_iso)
        """
        n_meas = len(self.bvals)
        n_iso = len(self.iso_diffusivities)
        n_aniso = len(self.bvecs) # We use acquisition directions as potential fiber orientations
        
        # Initialize Matrix A
        A = np.zeros((n_meas, n_aniso + n_iso))
        
        # --- Part 1: Anisotropic Basis Functions (Fibers) ---
        # Signal = exp(-b * g^T * D * g)
        # We assume a cylindrical tensor D oriented along each bvec direction
        for j in range(n_aniso):
            fiber_dir = self.bvecs[j]
            # Normalize fiber direction
            norm = np.linalg.norm(fiber_dir)
            if norm > 0:
                fiber_dir = fiber_dir / norm
                
            # Calculate predicted signal for this specific fiber orientation across all measurements
            # Simplified formula for cylindrical tensor projection:
            # D_app = D_rad + (D_ax - D_rad) * (g . fiber_dir)^2
            
            # Dot product of all gradient directions (g) with the current fiber basis (fiber_dir)
            cos_angles = np.dot(self.bvecs, fiber_dir)
            
            # Apparent diffusion coefficient for this specific basis tensor
            D_app = self.radial_diff_basis + (self.axial_diff_basis - self.radial_diff_basis) * (cos_angles**2)
            
            # Populate column j
            A[:, j] = np.exp(-self.bvals * D_app)
            
        # --- Part 2: Isotropic Basis Functions (Spectrum) ---
        # Signal = exp(-b * D_iso)
        for i, D_iso in enumerate(self.iso_diffusivities):
            # Populate the columns after the anisotropic ones
            A[:, n_aniso + i] = np.exp(-self.bvals * D_iso)
            
        return A

    def fit_voxel(self, signal: np.ndarray) -> DBSIParams:
        """
        Fits a single voxel using Non-Negative Least Squares (NNLS).
        
        Args:
            signal: 1D array of signal intensities (N_measurements).
            
        Returns:
            DBSIParams object containing fitted fractions and metrics.
        """
        # Signal Normalization (S / S0)
        # We assume the first volume is b0, or calculate mean of b<50
        if np.any(self.bvals < 50):
            S0 = np.mean(signal[self.bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6 or np.any(np.isnan(signal)):
            return self._get_empty_params()
            
        y = signal / S0
        
        # --- SOLVER: NNLS ---
        # Solve argmin_x || Ax - y ||_2 subject to x >= 0
        weights, residual = nnls(self.design_matrix, y)
        
        # --- Extract Metrics from Weights ---
        n_aniso = len(self.bvecs)
        
        # 1. Anisotropic Weights (Fibers)
        aniso_weights = weights[:n_aniso]
        f_fiber = np.sum(aniso_weights)
        
        # Determine dominant fiber direction (index of max weight)
        if f_fiber > 0:
            dom_idx = np.argmax(aniso_weights)
            main_dir = self.bvecs[dom_idx]
        else:
            main_dir = np.array([0.0, 0.0, 0.0])
        
        # 2. Isotropic Weights (Spectrum)
        iso_weights = weights[n_aniso:]
        
        # Aggregate spectrum into clinical compartments based on diffusivity thresholds
        # Thresholds are in mm^2/s. Note: 0.3e-3 mm^2/s = 0.3 um^2/ms
        
        # Restricted (Cells): D <= 0.3 um^2/ms
        mask_res = self.iso_diffusivities <= 0.3e-3
        f_restricted = np.sum(iso_weights[mask_res])
        
        # Hindered (Edema/Tissue): 0.3 < D <= 2.0 um^2/ms
        mask_hin = (self.iso_diffusivities > 0.3e-3) & (self.iso_diffusivities <= 2.0e-3)
        f_hindered = np.sum(iso_weights[mask_hin])
        
        # Free Water (CSF): D > 2.0 um^2/ms
        mask_wat = self.iso_diffusivities > 2.0e-3
        f_water = np.sum(iso_weights[mask_wat])
        
        # Calculate R-Squared
        predicted_signal = self.design_matrix @ weights
        ss_res = np.sum((y - predicted_signal)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0

        return DBSIParams(
            f_restricted=float(f_restricted),
            f_hindered=float(f_hindered),
            f_water=float(f_water),
            f_fiber=float(f_fiber),
            fiber_dir=main_dir,
            axial_diffusivity=self.axial_diff_basis,
            radial_diffusivity=self.radial_diff_basis,
            r_squared=float(r2)
        )

    def fit_volume(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Fits the DBSI model to an entire 4D volume.
        
        Args:
            volume: 4D numpy array (X, Y, Z, N_meas).
            mask: Optional 3D boolean mask (X, Y, Z).
            
        Returns:
            Dictionary of 3D parameter maps.
        """
        if volume.shape[-1] != len(self.bvals):
            raise ValueError(f"Volume last dimension ({volume.shape[-1]}) does not match bvals length ({len(self.bvals)})")
            
        X, Y, Z, _ = volume.shape
        
        if mask is None:
            mask = np.ones((X, Y, Z), dtype=bool)
            
        # Initialize output maps
        maps = {
            'fiber_fraction': np.zeros((X, Y, Z)),
            'restricted_fraction': np.zeros((X, Y, Z)),
            'hindered_fraction': np.zeros((X, Y, Z)),
            'water_fraction': np.zeros((X, Y, Z)),
            'fiber_dir_x': np.zeros((X, Y, Z)),
            'fiber_dir_y': np.zeros((X, Y, Z)),
            'fiber_dir_z': np.zeros((X, Y, Z)),
            'r_squared': np.zeros((X, Y, Z)),
        }
        
        n_voxels = np.sum(mask)
        
        # Iterate with progress bar
        with tqdm(total=n_voxels, desc="Fitting DBSI Volume", unit="vox") as pbar:
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):
                        if not mask[x, y, z]:
                            continue
                            
                        signal = volume[x, y, z, :]
                        params = self.fit_voxel(signal)
                        
                        # Store results
                        maps['fiber_fraction'][x, y, z] = params.fiber_density
                        maps['restricted_fraction'][x, y, z] = params.f_restricted
                        maps['hindered_fraction'][x, y, z] = params.f_hindered
                        maps['water_fraction'][x, y, z] = params.f_water
                        maps['fiber_dir_x'][x, y, z] = params.fiber_dir[0]
                        maps['fiber_dir_y'][x, y, z] = params.fiber_dir[1]
                        maps['fiber_dir_z'][x, y, z] = params.fiber_dir[2]
                        maps['r_squared'][x, y, z] = params.r_squared
                        
                        pbar.update(1)
                        
        return maps

    def _get_empty_params(self) -> DBSIParams:
        """Returns a zero-filled DBSIParams object for background/error voxels."""
        return DBSIParams(
            f_restricted=0.0, f_hindered=0.0, f_water=0.0, f_fiber=0.0,
            fiber_dir=np.zeros(3), axial_diffusivity=0.0, radial_diffusivity=0.0,
            r_squared=0.0
        )