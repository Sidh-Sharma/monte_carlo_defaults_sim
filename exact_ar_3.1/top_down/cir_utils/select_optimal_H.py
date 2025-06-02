import numpy as np
from scipy.optimize import minimize_scalar
import logging
from typing import Optional

from config.model_params import ModelParams
from utils.numerical import laplace_transform_G

EPS_ZERO = 1e-12
EPS_SMALL = 1e-8

logger = logging.getLogger("monte_carlo_sim")

def select_dominating_intensity_H(y: float, model_params: ModelParams, 
                              t_C: float = 1.0, t_eta: float = 0.8, buffer: float = 0.01) -> float:
    """
    Find optimal threshold H for the dominating intensity. As outlined in Appendix A of the paper.
    Eqn A2 is used to compute the optimal threshold H.
    
    Parameters:
        y: float - Initial intensity
        model_params: ModelParams - CIR model parameters
        t_C: float - Cost parameter of computing hitting time (simplified meaning)
        t_eta: float - Cost parameter of computing intensity transition (simplified meaning)
        buffer: float - Safety margin
        
    Returns:
        float: Optimal threshold H
    """
    try:
        theta = model_params.theta
        
        H_max = max(y * 3, theta * 2, 1.0)
        H_min = max(y + EPS_SMALL, theta + buffer, EPS_SMALL)
        
        def objective(H):
            try:
                G = laplace_transform_G(
                    params=model_params, 
                    H=H, 
                    y=y, 
                    logger=logger
                )
                
                numerator = H * (t_C + G * t_eta)
                denominator = 1 - G
                
                if denominator < EPS_ZERO:
                    return 1e10
                    
                return numerator / denominator
                
            except Exception as e:
                logger.debug(f"Optimization objective failed: {e}")
                return 1e10
        
        result = minimize_scalar(objective, bounds=(H_min, H_max), method='bounded')
        
        if not result.success:
            logger.warning("H optimization failed, using heuristic")
            H_star = max(y * 1.5, theta * 1.2)
        else:
            H_star = result.x
        
        # Apply safety constraints
        H_final = max(H_star, y + EPS_SMALL, theta + buffer)
        
        logger.debug(f"Optimal H: {H_final:.6f} (y={y:.6f})")
        return H_final
        
    except Exception as e:
        logger.error(f"H optimization failed: {e}")
        return max(y * 1.5, theta * 1.2, 1.0)