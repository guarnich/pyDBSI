# dbsi_toolbox/base.py
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple
import sys
from .common import DBSIParams

class BaseDBSI:
    """
    Abstract base class for DBSI models. 
    Handles volume iteration, masking, and I/O. 
    Do not instantiate directly.
    """
    
    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        """
        Fits a single voxel. Must be implemented by subclasses.
        """
        raise NotImplementedError("Use a specific model implementation (Linear or NonLinear).")

    def fit_volume(self, 
                   volume: np.ndarray, 
                   bvals: np.ndarray, 
                   bvecs: np.ndarray, 
                   mask: Optional[np.ndarray] = None,
                   show_progress: bool = True,
                   **kwargs) -> Dict[str, np.ndarray]:
        """
        Fits the model to a 4D volume.
        
        Args:
            volume: 4D numpy array (X, Y, Z, N)
            bvals: 1D array of b-values
            bvecs: Array of gradient directions (N, 3) or (3, N)
            mask: Optional 3D boolean mask
            show_progress: Whether to display the TQDM progress bar
            
        Returns:
            Dictionary of 3D parameter maps.
        """
        # 1. Validate Dimensions
        if volume.ndim != 4:
            raise ValueError(f"Volume must be 4D, got {volume.ndim}")
        
        X, Y, Z, N = volume.shape
        
        # 2. Standardize bvals and bvecs
        flat_bvals = np.array(bvals).flatten()
        
        if bvecs.shape == (3, N):
            self.current_bvecs = bvecs.T
        elif bvecs.shape == (N, 3):
            self.current_bvecs = bvecs
        else:
             raise ValueError(f"bvecs shape {bvecs.shape} mismatch. Expected ({N}, 3)")
             
        # 3. Prepare Mask
        if mask is None:
            # Simple auto-masking strategy if none provided
            b0_mean = np.mean(volume[..., flat_bvals < 50], axis=-1)
            mask = b0_mean > np.percentile(b0_mean, 10)
            
        # 4. Initialize Output Maps
        maps = {
            'fiber_fraction': np.zeros((X, Y, Z)),
            'restricted_fraction': np.zeros((X, Y, Z)),
            'hindered_fraction': np.zeros((X, Y, Z)),
            'water_fraction': np.zeros((X, Y, Z)),
            'fiber_dir_x': np.zeros((X, Y, Z)),
            'fiber_dir_y': np.zeros((X, Y, Z)),
            'fiber_dir_z': np.zeros((X, Y, Z)),
            'r_squared': np.zeros((X, Y, Z)),
            'axial_diffusivity': np.zeros((X, Y, Z)),
            'radial_diffusivity': np.zeros((X, Y, Z)),
        }
        
        # 5. Iteration Loop
        n_voxels = np.sum(mask)
        
        # Using sys.stdout ensures it works well in both terminals and notebooks
        with tqdm(total=n_voxels, desc="Fitting DBSI", unit="vox", 
                  disable=not show_progress, file=sys.stdout) as pbar:
            
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):
                        if not mask[x, y, z]:
                            continue
                        
                        # Extract single voxel signal
                        signal = volume[x, y, z, :]
                        
                        # Polymorphism: calls the specific implementation
                        params = self.fit_voxel(signal, flat_bvals)
                        
                        # Store results
                        maps['fiber_fraction'][x, y, z] = params.fiber_density
                        maps['restricted_fraction'][x, y, z] = params.f_restricted
                        maps['hindered_fraction'][x, y, z] = params.f_hindered
                        maps['water_fraction'][x, y, z] = params.f_water
                        maps['fiber_dir_x'][x, y, z] = params.fiber_dir[0]
                        maps['fiber_dir_y'][x, y, z] = params.fiber_dir[1]
                        maps['fiber_dir_z'][x, y, z] = params.fiber_dir[2]
                        maps['r_squared'][x, y, z] = params.r_squared
                        maps['axial_diffusivity'][x, y, z] = params.axial_diffusivity
                        maps['radial_diffusivity'][x, y, z] = params.radial_diffusivity
                        
                        pbar.update(1)
                        
        return maps
    
    def _get_empty_params(self) -> DBSIParams:
        """Returns zero-filled parameters for failed fits or empty voxels."""
        return DBSIParams(
            f_restricted=0.0, f_hindered=0.0, f_water=0.0, f_fiber=0.0,
            fiber_dir=np.zeros(3), axial_diffusivity=0.0, radial_diffusivity=0.0,
            r_squared=0.0
        )