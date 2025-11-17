# dbsi_toolbox/nonlinear.py
import numpy as np
from scipy.optimize import least_squares
from .base import BaseDBSI
from .common import DBSIParams

class DBSI_NonLinear(BaseDBSI):
    """
    DBSI implementation using Non-Linear Least Squares (NLLS).
    Optimizes fiber angle and diffusivities explicitly. Slower than Linear.
    """
    def __init__(self):
        self.n_params = 10 # 3 iso pairs + 1 fiber (f, Dax, Drad, theta, phi)
    
    def _predict_signal(self, params, bvals, bvecs):
        # params unpacking: [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
        f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi = params
        
        D_water = 3.0e-3 # Fixed
        
        # Normalize fractions
        f_total = f_res + f_hin + f_wat + f_fib + 1e-10
        
        # Fiber direction from spherical coords
        fiber_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Simplified tensor projection
        cos_angles = np.dot(bvecs, fiber_dir)
        # D_app = D_rad + (D_ax - D_rad) * (g . v)^2
        D_app_fiber = D_rad + (D_ax - D_rad) * (cos_angles**2)
        
        signal = (
            (f_res / f_total) * np.exp(-bvals * D_res) +
            (f_hin / f_total) * np.exp(-bvals * D_hin) +
            (f_wat / f_total) * np.exp(-bvals * D_water) +
            (f_fib / f_total) * np.exp(-bvals * D_app_fiber)
        )
        return signal

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        # Signal Normalization
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6: return self._get_empty_params()
        y = signal / S0
        
        # Initial guess
        p0 = [0.1, 0.0002, 0.2, 0.001, 0.1, 0.6, 0.0015, 0.0003, np.pi/4, np.pi/4]
        bounds_lower = [0.0, 0.0,     0.0, 0.0003, 0.0, 0.0, 0.0005, 0.0,     0.0,   0.0]
        bounds_upper = [1.0, 0.0003,  1.0, 0.0015, 1.0, 1.0, 0.003,  0.0015, np.pi, 2*np.pi]
        
        def objective(p):
            return self._predict_signal(p, bvals, self.current_bvecs) - y
        
        try:
            res = least_squares(objective, p0, bounds=(bounds_lower, bounds_upper), method='trf')
            p = res.x
            
            # Calculate Fiber Direction
            fiber_dir = np.array([
                np.sin(p[8]) * np.cos(p[9]),
                np.sin(p[8]) * np.sin(p[9]),
                np.cos(p[8])
            ])
            
            # Normalize fractions for output
            f_total = p[0] + p[2] + p[4] + p[5] + 1e-10
            
            # R-Squared
            ss_res = np.sum(res.fun**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else 0
            
            return DBSIParams(
                f_restricted=p[0]/f_total,
                f_hindered=p[2]/f_total,
                f_water=p[4]/f_total,
                f_fiber=p[5]/f_total,
                fiber_dir=fiber_dir,
                axial_diffusivity=p[6],
                radial_diffusivity=p[7],
                r_squared=r2
            )
        except:
            return self._get_empty_params()