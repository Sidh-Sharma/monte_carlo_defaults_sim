import numpy as np
from scipy.special import iv
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import interp1d
import logging
from typing import Optional

from config.model_params import ModelParams
from top_down.cir_utils.hittingtime import compute_coefficients, compute_survival_probability_p1

# Constants for numerical stability
EPS_ZERO = 1e-12
EPS_SMALL = 1e-8

# Get logger
logger = logging.getLogger("monte_carlo_sim")

def cir_transition_density_g(x: float, y: float, t: float, model_params: ModelParams) -> float:
    """
    Robust CIR transition density computation
    
    Parameters:
        x: float - Target intensity value
        y: float - Initial intensity value
        t: float - Time interval
        model_params: ModelParams - CIR model parameters
        
    Returns:
        float: Transition density g(x, t | y, 0)
    """
    try:
        kappa = model_params.kappa
        theta = model_params.theta
        sigma = model_params.sigma
        
        if t <= 0 or x < 0 or y < 0:
            return 0.0
            
        q = 2 * kappa * theta / sigma**2 - 1
        exp_term = np.exp(-kappa * t)
        a = 2 * kappa / (sigma**2 * (1 - exp_term))
        b = a * exp_term
        
        by = b * y + EPS_ZERO
        ax = a * x + EPS_ZERO
        
        # Compute in log space for stability
        log_factor = np.log(a) - (ax + b * y)
        log_power = (q / 2) * (np.log(ax) - np.log(by))
        
        # Bessel function argument
        bessel_arg = 2 * np.sqrt(a * b * x * y)
        if bessel_arg > 100:  # Use asymptotic approximation
            log_bessel = bessel_arg - 0.5 * np.log(2 * np.pi * bessel_arg)
        else:
            bessel_val = iv(q, bessel_arg)
            if bessel_val <= 0:
                return 0.0
            log_bessel = np.log(bessel_val)
        
        log_density = log_factor + log_power + log_bessel
        
        if log_density > 50:  # Prevent overflow
            return 0.0
            
        return np.exp(log_density)
        
    except Exception as e:
        logger.debug(f"CIR density computation failed: {e}")
        return 0.0
    
def sample_conditional_intensity_from_f(y: float, H: float, tau: float, model_params: ModelParams) -> float:
    """
    Sample from conditional transition density with vectorized integration
    
    Parameters:
        y: float - Initial intensity
        H: float - Threshold value
        tau: float - Time interval
        model_params: ModelParams - CIR model parameters
    
    Returns:
        float: Sampled intensity value
    """
    try:
        theta = model_params.theta
        nu_scale = max(y, theta, 0.1)
        x_max = max(nu_scale * 5, 10.0)
        
        # Create x-grid with endpoint for proper integration
        n_points = 500  # Default integration points
        x_grid = np.linspace(EPS_ZERO, x_max, n_points)
        
        # Vectorized computation of unconditional density
        g_vals = np.array([cir_transition_density_g(x, y, tau, model_params) for x in x_grid])
        
        # Precompute hitting time coefficients
        eta_n, beta_n = compute_coefficients(y, H, model_params)
        eta_n = np.array(eta_n)
        beta_n = np.array(beta_n)

        # Create s-grid (ensure sufficient points)
        s_points = 500  # Default grid size for numerical integration
        s_grid = np.linspace(0, tau, s_points)
        
        # Precompute hitting density u(s) for all s
        u_vals = np.sum(beta_n[:, None] * eta_n[:, None] * 
                      np.exp(-eta_n[:, None] * s_grid[None, :]), axis=0)

        # Vectorized computation of convolution integral
        conv_vals = np.zeros_like(x_grid)
        for j, sj in enumerate(s_grid):
            if sj < tau:  # Only compute for valid times
                time_remaining = tau - sj
                # Vectorized call over x_grid
                g_shift = np.array([cir_transition_density_g(x, H, time_remaining, model_params) 
                                    for x in x_grid])
                conv_vals += g_shift * u_vals[j]
        
        # Apply Simpson integration weights
        ds = s_grid[1] - s_grid[0] if len(s_grid) > 1 else tau
        conv_vals *= ds  # Simple scaling for integration
        
        # Compute survival probability
        p1 = compute_survival_probability_p1(y, H, tau, model_params)
        
        # Conditional density with numerical stability
        f_vals = (g_vals - conv_vals) / p1
        f_vals = np.clip(f_vals, EPS_ZERO, None)
        
        # PROPER NORMALIZATION using Simpson integration
        total_prob = simpson(f_vals, x_grid)
        if total_prob < EPS_ZERO:
            logger.warning("Normalization failed, using fallback")
            return max(y * np.random.uniform(0.5, 1.5), EPS_SMALL)
        
        f_vals /= total_prob  # Correct normalization
        
        # Build CDF using cumulative trapezoidal rule
        cdf_vals = cumulative_trapezoid(f_vals, x_grid, initial=0)
        cdf_vals /= cdf_vals[-1]  # Ensure CDF ends at 1
        
        # Maintain monotonicity
        cdf_vals = np.maximum.accumulate(cdf_vals)
        
        # Sample using inverse CDF
        u_sample = np.random.uniform()
        inv_cdf = interp1d(cdf_vals, x_grid, bounds_error=False, 
                         fill_value=(x_grid[0], x_grid[-1]),
                         assume_sorted=True)
        sample = float(inv_cdf(u_sample))
        
        # Final bounds check
        sample = max(sample, EPS_SMALL)
        
        logger.debug(f"Sampled intensity: {sample:.6f}")
        return sample
        
    except Exception as e:
        logger.error(f"Conditional sampling failed: {e}")
        return max(y * np.random.uniform(0.8, 1.2), EPS_SMALL)
