# dbsi_toolbox/calibration.py

import numpy as np
from typing import List, Dict, Optional
from .twostep import DBSI_TwoStep

def generate_synthetic_signal_rician(
    bvals: np.ndarray, 
    bvecs: np.ndarray, 
    f_fiber: float, 
    f_cell: float, 
    f_water: float, 
    snr: float
) -> np.ndarray:
    """
    Generates a synthetic diffusion signal with Rician noise.
    Based on physiological parameters from Wang et al., 2011.
    """
    # 1. Standard physiological parameters
    D_fiber_ax = 1.5e-3
    D_fiber_rad = 0.3e-3
    D_cell = 0.0e-3      # Restricted diffusion
    D_water = 3.0e-3     # Free diffusion
    
    # Arbitrary fiber direction along X-axis
    fiber_dir = np.array([1.0, 0.0, 0.0])
    
    # 2. Forward Model Calculation
    n_meas = len(bvals)
    signal = np.zeros(n_meas)
    
    for i in range(n_meas):
        cos_angle = np.dot(bvecs[i], fiber_dir)
        D_app = D_fiber_rad + (D_fiber_ax - D_fiber_rad) * (cos_angle**2)
        
        s_fiber = np.exp(-bvals[i] * D_app)
        s_cell = np.exp(-bvals[i] * D_cell)
        s_water = np.exp(-bvals[i] * D_water)
        
        signal[i] = (f_fiber * s_fiber) + (f_cell * s_cell) + (f_water * s_water)
    
    # 3. Add Rician Noise
    # Generate Gaussian noise on real and imaginary channels
    sigma = 1.0 / snr
    noise_real = np.random.normal(0, sigma, n_meas)
    noise_imag = np.random.normal(0, sigma, n_meas)
    
    # MRI signal is the magnitude
    signal_noisy = np.sqrt((signal + noise_real)**2 + noise_imag**2)
    
    return signal_noisy


def optimize_dbsi_params(
    real_bvals: np.ndarray,
    real_bvecs: np.ndarray,
    snr_estimate: float = 30.0,
    n_monte_carlo: int = 100,
    bases_grid: List[int] = [20, 30, 50, 75],
    lambdas_grid: List[float] = [0.0, 0.01, 0.1, 0.2],
    ground_truth: Dict[str, float] = {'f_fiber': 0.5, 'f_cell': 0.3, 'f_water': 0.2},
    verbose: bool = True
) -> Dict:
    """
    Performs a Monte Carlo calibration to find the optimal hyperparameters
    (n_iso_bases, reg_lambda) for a specific acquisition protocol.
    
    Args:
        real_bvals: Array of b-values from the real protocol.
        real_bvecs: Array of b-vecs from the real protocol (N, 3).
        snr_estimate: Estimated SNR of the real images (default 30).
        n_monte_carlo: Number of iterations per configuration.
        bases_grid: List of n_iso_bases to test.
        lambdas_grid: List of reg_lambda to test.
        ground_truth: Dictionary containing "true" fractions to simulate.
        
    Returns:
        A dictionary containing the best parameters and error statistics.
    """
    
    if verbose:
        print(f"\n[Calibration] Starting optimization for protocol with {len(real_bvals)} volumes.")
        print(f"[Calibration] Target: Cell Fraction = {ground_truth['f_cell']}")
        print("-" * 75)
        print(f"{'Bases':<6} | {'Lambda':<8} | {'Avg Cell':<10} | {'Avg Error':<10} | {'Std Dev':<10}")
        print("-" * 75)

    results = []

    # Standardize vectors for calculation
    flat_bvals = np.array(real_bvals).flatten()
    if real_bvecs.shape[0] == 3:
        clean_bvecs = real_bvecs.T
    else:
        clean_bvecs = real_bvecs

    # Grid Search Loop
    for n_bases in bases_grid:
        for reg in lambdas_grid:
            
            errors = []
            estimates = []
            
            # Initialize Model (Once per configuration)
            # Note: We use TwoStep but focus on the Linear Spectrum part
            # because that is where lambda and n_bases are critical.
            model = DBSI_TwoStep(
                n_iso_bases=n_bases,
                reg_lambda=reg,
                iso_diffusivity_range=(0.0, 3.0e-3)
            )
            
            # Manual Matrix Setup (Bypass fit_volume for speed)
            model.spectrum_model.design_matrix = model.spectrum_model._build_design_matrix(flat_bvals, clean_bvecs)
            model.spectrum_model.current_bvecs = clean_bvecs
            model.fitting_model.current_bvecs = clean_bvecs
            
            # Monte Carlo Loop
            for _ in range(n_monte_carlo):
                # Generate new signal with new noise instance
                sig = generate_synthetic_signal_rician(
                    flat_bvals, clean_bvecs,
                    ground_truth['f_fiber'], ground_truth['f_cell'], ground_truth['f_water'],
                    snr=snr_estimate
                )
                
                try:
                    res = model.fit_voxel(sig, flat_bvals)
                    estimates.append(res.f_restricted)
                    errors.append(abs(res.f_restricted - ground_truth['f_cell']))
                except Exception:
                    pass # Ignore failed fits

            if not errors: continue

            avg_error = np.mean(errors)
            avg_estimate = np.mean(estimates)
            std_dev = np.std(errors)
            
            if verbose:
                print(f"{n_bases:<6} | {reg:<8} | {avg_estimate:.4f}     | {avg_error*100:.2f}%      | {std_dev*100:.2f}%")
            
            results.append({
                'n_bases': n_bases,
                'reg_lambda': reg,
                'avg_error': avg_error,
                'std_dev': std_dev,
                'avg_estimate': avg_estimate
            })

    # Selection Criteria: Minimize average error. 
    best_config = min(results, key=lambda x: x['avg_error'])
    
    if verbose:
        print("-" * 75)
        print(f"[Calibration] WINNER: {best_config['n_bases']} Bases, Lambda {best_config['reg_lambda']}")
        print(f"              Avg Error: {best_config['avg_error']*100:.2f}%")
        print("-" * 75)
        
    return best_config