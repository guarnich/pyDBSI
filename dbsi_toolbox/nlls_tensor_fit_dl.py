import numpy as np
from scipy.optimize import least_squares
from .nlls_tensor_fit import DBSI_TensorFit
from .common import DBSIParams
from typing import Optional

class DBSI_TensorFit_DL(DBSI_TensorFit):
    def __init__(self, alpha_reg: float = 0.5, d_res_threshold: float = 0.3e-3):
        super().__init__()
        self.alpha_reg = alpha_reg
        self.d_res_threshold = d_res_threshold

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray, 
                  initial_guess: Optional[DBSIParams] = None,
                  dl_priors: Optional[np.ndarray] = None) -> DBSIParams:
        
        # Normalizzazione S0
        b0_idx = bvals < 50
        if b0_idx.sum() > 0: s0 = np.mean(signal[b0_idx])
        else: s0 = signal[0]
        
        if s0 <= 1e-6: return self._get_empty_params()
        y = signal / s0

        # --- 1. Initial Guess (Cruciale!) ---
        # Se abbiamo i priors, USIAMOLI come punto di partenza.
        if dl_priors is not None:
            # Assicuriamoci che non siano tutti zero o invalidi
            if np.sum(dl_priors) > 0.1:
                p0_f = dl_priors
            else:
                p0_f = [0.1, 0.2, 0.1, 0.6]
        else:
            p0_f = [0.1, 0.2, 0.1, 0.6]

        # [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
        p0 = [
            p0_f[0], 0.0001,            # Res
            p0_f[1], 0.0015,            # Hin
            p0_f[2], p0_f[3],           # Wat, Fib
            1.5e-3, 0.3e-3,             # D
            0.0, 0.0                    # Angles
        ]
        
        # Bounds
        lb = [0, 0,                  0, self.d_res_threshold, 0, 0, 0.5e-3, 0,      -np.pi, -np.pi]
        ub = [1, self.d_res_threshold, 1, 3.0e-3,               1, 1, 3.0e-3, 1.5e-3,  np.pi,  2*np.pi]

        # --- 2. Cost Function Bilanciata ---
        # Fattore di scala per rendere confrontabili i residui
        n_meas = len(y)
        n_priors = 4
        # Vogliamo che alpha=1 significhi "pari importanza per elemento"
        reg_scale = np.sqrt(self.alpha_reg * (n_meas / n_priors))

        def objective(p):
            # A. Errore Segnale (Fisica)
            y_pred = self._predict_signal(p, bvals, self.current_bvecs)
            res_sig = y_pred - y
            
            if dl_priors is None: return res_sig
            
            # B. Errore Regolarizzazione (Priors)
            f_tot = p[0] + p[2] + p[4] + p[5] + 1e-9
            f_curr = np.array([p[0], p[2], p[4], p[5]]) / f_tot
            
            res_reg = (f_curr - dl_priors) * reg_scale
            
            return np.concatenate([res_sig, res_reg])

        try:
            # Usa 'trf' (Trust Region Reflective) che rispetta i bounds robustamente
            res = least_squares(objective, p0, bounds=(lb, ub), method='trf', max_nfev=100)
            p = res.x
            
            # Ricostruzione Output
            f_tot = p[0] + p[2] + p[4] + p[5] + 1e-9
            
            # R2 solo sul segnale
            y_fin = self._predict_signal(p, bvals, self.current_bvecs)
            ss_res = np.sum((y - y_fin)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else 0

            fib_dir = np.array([np.sin(p[8])*np.cos(p[9]), np.sin(p[8])*np.sin(p[9]), np.cos(p[8])])

            return DBSIParams(
                f_restricted=p[0]/f_tot, f_hindered=p[2]/f_tot, 
                f_water=p[4]/f_tot, f_fiber=p[5]/f_tot,
                fiber_dir=fib_dir, axial_diffusivity=p[6], radial_diffusivity=p[7],
                r_squared=r2
            )
        except:
            return self._get_empty_params()