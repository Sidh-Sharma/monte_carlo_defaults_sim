import numpy as np
import logging
from typing import Optional, Tuple, List

from config.model_params import ModelParams
from top_down.cir_utils.hittingtime import sample_hitting_time_from_v, compute_survival_probability_p1
from top_down.cir_utils.intensity_transition import sample_conditional_intensity_from_f
from top_down.cir_utils.select_optimal_H import select_dominating_intensity_H

EPS_ZERO = 1e-12
EPS_SMALL = 1e-8

logger = logging.getLogger("monte_carlo_sim")

def simulate_next_default_step(
    t_prev: float,
    lambda_prev: float,
    T_max: float,
    model_params: ModelParams
) -> Optional[Tuple[float, float]]:
    """
    Simulates a single default step in the top-down model.

    Parameters:
        t_prev: float
            Previous default time T_{n-1}
        lambda_prev: float
            Intensity after last jump, λ_{T_{n-1}} 
        T_max: float
            Final time horizon for simulation
        model_params: ModelParams
            Model parameters (kappa, theta, sigma, gamma)

    Returns:
        Optional[Tuple[float, float]]:
            If a default occurs before T_max:
                (T_n, ν_{T_n})
            Else:
                None (no more defaults)
    """
    try:
        # STEP 1: Initialize y := λ_{T_{n-1}}, t := T_{n-1}
        y = max(lambda_prev, EPS_SMALL)
        t = t_prev 
        
        max_inner_iterations = 50  # Prevent infinite loops
        
        for iteration in range(max_inner_iterations):
            # STEP 2: Select H ≥ max(y, θ + ε) using optimal approach
            H = select_dominating_intensity_H(y, model_params)
            logger.info(f"Iteration {iteration + 1}: Selected H={H:.6f} for y={y:.6f}")
            
            # STEP 3: Sample τ ∼ Exp(H), propose next interarrival time
            tau = np.random.exponential(scale=1/H)
            logger.debug(f"Sampled interarrival time τ={tau:.6f} from Exp(H) with H={H:.6f}")
            
            # STEP 4: Compute survival probability p1 = P(y stays below H up to τ)
            p1 = compute_survival_probability_p1(y, H, tau, model_params)
            logger.info(f"Computed survival probability p1={p1:.6f} for y={y:.6f}, H={H:.6f}, τ={tau:.6f}")
            
            # STEP 5: First rejection test with u1 ∼ Unif(0,1)
            u1 = np.random.uniform()
            
            if u1 > p1:
                # Process hits H → sample σ_H from hitting time density
                logger.info(f"Process hits boundary H={H:.6f} at t={t:.6f}, sampling hitting time")
                sigma_H = sample_hitting_time_from_v(y, H, tau, model_params)
                t += sigma_H
                
                # STEP 8: Check if t exceeds T_max
                if t > T_max:
                    return None
                    
                y = H  # Update intensity and repeat
                continue
            else:
                # Process survives to τ
                t += tau
                logger.info(f"Process survives to t={t:.6f}, sampling conditional intensity")
                
                # STEP 8: Check if t exceeds T_max
                if t > T_max:
                    return None
                
                # STEP 6: Sample ν_{T_n} from conditional density f(.; y, H, τ)
                nu_Tn = sample_conditional_intensity_from_f(y, H, tau, model_params)
                logger.debug(f"Sampled conditional intensity ν_{t}={nu_Tn:.6f} at t={t:.6f}")
                
                # STEP 7: Second rejection test using u2 ∼ Unif(0,1)
                u2 = np.random.uniform()
                
                if u2 <= nu_Tn / H:
                    # Accept default at T_n = t
                    logger.info(f"Default accepted at t={t:.6f}, ν={nu_Tn:.6f}")
                    return (t, nu_Tn)
                else:
                    # Reject - update ν_t and go back to step 2
                    y = nu_Tn
                    continue
        
        logger.warning(f"Maximum inner iterations reached at t={t:.6f}")
        return None
        
    except Exception as e:
        logger.error(f"Error in simulate_next_default_step: {e}")
        return None