import time
import numpy as np
from scipy.optimize import brentq
from typing import Tuple, List, Optional
import logging

from config.model_params import ModelParams
from utils.numerical import _hyp1f1, hyp1f1_derivative_a, validate_parameters

# Constants for numerical stability
EPS_ZERO = 1e-12
EPS_SMALL = 1e-8

# Get logger
logger = logging.getLogger("monte_carlo_sim")

def compute_coefficients(y: float, H: float, model_params: ModelParams, n_roots: int = 15) -> Tuple[List[float], List[float]]:
    """Compute eigenfunction expansion coefficients as given by eqn 20 in section 4.2. root finding given in appendix B of the paper"""
    
    try:
        kappa = model_params.kappa
        theta = model_params.theta
        sigma = model_params.sigma
        
        c_bar = 2 * kappa * theta / sigma**2
        H_bar = 2 * kappa * H / sigma**2
        y_bar = 2 * kappa * y / sigma**2

        if c_bar < 1 or H_bar < c_bar:
            logger.error(f"Invalid parameters: c_bar={c_bar:.6f}, H_bar={H_bar:.6f}, y_bar={y_bar:.6f}")
            return [], []
        
        logger.debug(f"Computing coefficients: c_bar={c_bar:.6f}, H_bar={H_bar:.6f}, y_bar={y_bar:.6f}")
        
        def objective(alpha):
            return _hyp1f1(alpha, c_bar, H_bar, logger)
        
        roots = []
        n = 0
        alpha_prev = 0

        # Find the first root in (-c_bar / H_bar, 0)
        left = -c_bar / H_bar
        right = 0
        try:
            alpha_curr = brentq(objective, left, right)
            roots.append(alpha_curr)
        except Exception as e:
            logger.error(f"Failed to find first root: {e}")
            return [], []

        # Find subsequent roots
        while len(roots) < n_roots:
            n += 1
            # Predict the next root using extrapolation
            alpha_hat = 2 * alpha_curr - alpha_prev

            # Inner loop to bracket the root
            bracket_found = False
            for attempt in range(20):  
                try:
                    test_val = objective(alpha_hat - 1)
                    if np.sign(test_val) == (-1)**n:
                        alpha_hat = alpha_hat - 1
                    else:
                        bracket_found = True
                        break
                except:
                    alpha_hat = alpha_hat - 1
                    if attempt > 10:  # Give up after many attempts
                        break
            
            if not bracket_found:
                logger.warning(f"Could not bracket root {n+1}, stopping at {len(roots)} roots")
                break
                
            try:
                alpha_next = brentq(objective, alpha_hat - 1, alpha_hat)
                roots.append(alpha_next)
                alpha_prev = alpha_curr
                alpha_curr = alpha_next
            except Exception as e:
                logger.warning(f"Failed to find root {n+1}: {e}, stopping at {len(roots)} roots")
                break
        
        if len(roots) < 5:
            logger.error(f"Insufficient roots found: {len(roots)}")
            return [], []
        
        # Compute coefficients
        eta_n, beta_n = [], []
        for alpha in roots:
            try:
                eta = -kappa * alpha
                eta_n.append(eta)
                
                numerator = _hyp1f1(alpha, c_bar, y_bar, logger)
                derivative = hyp1f1_derivative_a(alpha, c_bar, H_bar, logger)
                
                if abs(derivative) < EPS_ZERO:
                    logger.warning(f"Near-zero derivative for alpha={alpha:.6f}")
                    beta = 0.0
                else:
                    beta = -numerator / (alpha * derivative)
                    
                beta_n.append(beta)
                
            except Exception as e:
                logger.warning(f"Coefficient computation failed for alpha={alpha:.6f}: {e}")
                eta_n.append(0.0)
                beta_n.append(0.0)
        
        # Filter out problematic coefficients
        valid_indices = [i for i, (eta, beta) in enumerate(zip(eta_n, beta_n)) 
                        if np.isfinite(eta) and np.isfinite(beta)]
        
        eta_n = [eta_n[i] for i in valid_indices]
        beta_n = [beta_n[i] for i in valid_indices]
        
        logger.debug(f"Computed {len(eta_n)} valid coefficients")
        
        return eta_n, beta_n
        
    except Exception as e:
        logger.error(f"Coefficient computation failed: {e}")
        return [], []

def compute_survival_probability_p1(y: float, H: float, tau: float, model_params: ModelParams) -> float:
    """Compute survival probability with numerical stability"""
    try:
        eta_n, beta_n = compute_coefficients(y, H, model_params)
        
        if not eta_n:
            logger.warning("No coefficients available, using fallback")
            return 0.5
        
        p1 = sum(beta * np.exp(-eta * tau) for eta, beta in zip(eta_n, beta_n) 
                if np.isfinite(beta) and np.isfinite(eta))
        
        # Stability bounds
        p1 = np.clip(p1, EPS_ZERO, 1 - EPS_ZERO)
        
        logger.debug(f"Survival probability: {p1:.6f}")
        return p1
        
    except Exception as e:
        logger.error(f"Survival probability computation failed: {e}")
        return 0.5

def sample_hitting_time_from_v(nu_t: float, H: float, tau: float, model_params: ModelParams) -> float:
    """Sample hitting time with robust computation"""
    try:
        eta_n, beta_n = compute_coefficients(nu_t, H, model_params)
        
        if not eta_n:
            logger.warning("No coefficients for hitting time, using uniform")
            return np.random.uniform(0, tau)
        
        # Compute total probability mass
        P = 1 - compute_survival_probability_p1(nu_t, H, tau, model_params)
        P = max(P, EPS_ZERO)
        
        U = np.random.uniform(0, P)
        
        def objective(s):
            integral = sum(beta * (1 - np.exp(-eta * s)) for beta, eta in zip(beta_n, eta_n)
                          if eta > EPS_ZERO)
            return integral - U
        
        # Try to find hitting time
        try:
            result = brentq(objective, 0, tau)
            return min(result, tau)
        except:
            logger.warning("Hitting time root finding failed, using approximation")
            return np.random.uniform(0, tau * 0.8)
            
    except Exception as e:
        logger.error(f"Hitting time sampling failed: {e}")
        return np.random.uniform(0, tau * 0.5)