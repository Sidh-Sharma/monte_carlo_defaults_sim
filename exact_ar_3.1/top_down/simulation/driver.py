import time
import logging
import numpy as np
from typing import List, Tuple, Optional

from config.model_params import ModelParams
from top_down.simulation.exact_ar_step import simulate_next_default_step
from utils.numerical import validate_parameters

logger = logging.getLogger("monte_carlo_sim")

def sample_loss() -> float:
    """Sample loss from a uniform distribution as per the original model"""
    return np.random.uniform(0.24, 0.96)

def run_simulation(
    lambda_0: float,
    T_max: float,
    model_params: ModelParams,
    max_defaults: int = 100
) -> List[Tuple[float, float, float]]:
    """
    Main simulation function that generates a sequence of default times and losses.
    
    Parameters:
        lambda_0: float
            Initial intensity λ_0
        T_max: float
            Time horizon
        model_params: ModelParams
            Model parameters (kappa, theta, sigma, gamma)
        max_defaults: int
            Maximum number of defaults to simulate
            
    Returns:
        List[Tuple[float, float, float]]: List of (default_time, intensity, loss) tuples
    """
    logger.info(f"Starting simulation: λ_0={lambda_0:.6f}, T_max={T_max:.2f}")
    
    # Validate parameters first
    validate_parameters(model_params, logger)
    
    defaults = []
    t = 0.0
    lam = lambda_0
    gamma = model_params.gamma if hasattr(model_params, 'gamma') else 0.99
    
    start_time = time.time()
    
    try:
        while t < T_max and len(defaults) < max_defaults:
            result = simulate_next_default_step(t, lam, T_max, model_params)
            
            if result is None:
                logger.info(f"No more defaults before T_max={T_max}")
                break
                
            Tn, nu_Tn = result
            loss = sample_loss()
            lam = nu_Tn + gamma * loss  # Update intensity with jump
            t = Tn
            defaults.append((Tn, lam, loss))
            
            if len(defaults) % 10 == 0:
                logger.info(f"Generated {len(defaults)} defaults, t={t:.4f}, λ={lam:.6f}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Simulation completed: {len(defaults)} defaults in {elapsed_time:.2f}s")
        
        return defaults
        
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
        return defaults
    except Exception as e:
        logger.error(f"Critical simulation error: {e}")
        return defaults
