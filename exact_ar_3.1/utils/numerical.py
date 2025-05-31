import numpy as np
from scipy.special import hyp1f1
from typing import Tuple, List
from config.model_params import ModelParams

# Constants for numerical stability
EPS_ZERO = 1e-12
EPS_SMALL = 1e-8

def validate_parameters(params: ModelParams, logger) -> bool:
    """Validate CIR parameters and check Feller condition"""
    try:
        kappa = params.kappa
        theta = params.theta
        sigma = params.sigma
        
        feller_condition = params.check_feller_condition()
        if not feller_condition:
            logger.warning(f"Feller condition violated: 2kappa*theta={2*kappa*theta:.6f} < sigma^2={sigma**2:.6f}")
            
        logger.info(f"Parameters: kappa={kappa:.6f}, c={theta:.6f}, sigma={sigma:.6f}")
        logger.info(f"Feller condition {'satisfied' if feller_condition else 'violated'}")
        
        return feller_condition
        
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        raise

def _hyp1f1(a: float, b: float, z: float, logger) -> float:
    """Hypergeometric function evaluation with fallbacks"""
    try:
        # Primary method - scipy's hyp1f1 with argument checks
        if abs(z) < 100 and abs(a) < 50:
            result = hyp1f1(a, b, z)
            if np.isfinite(result) and abs(result) < 1e10:
                return result

        # Clip arguments if required    
        a_clip = np.clip(a, -50, 50)
        z_clip = np.clip(z, -100, 100)
        result = hyp1f1(a_clip, b, z_clip)
        
        if not np.isfinite(result):
            logger.warning(f"hyp1f1 returned non-finite: a={a}, b={b}, z={z}")
            return 1.0  
            
        return result
        
    except Exception as e:
        logger.warning(f"hyp1f1 evaluation failed: {e}, using fallback")
        return 1.0

def hyp1f1_derivative_a(a: float, b: float, z: float, logger, delta: float = 1e-5) -> float:
    """Numerical derivative of hypergeometric function"""
    try:
        f_plus = _hyp1f1(a + delta, b, z, logger)
        f_minus = _hyp1f1(a - delta, b, z, logger)
        derivative = (f_plus - f_minus) / (2 * delta)
        
        if not np.isfinite(derivative):
            # Try smaller delta
            delta *= 0.1
            f_plus = _hyp1f1(a + delta, b, z, logger)
            f_minus = _hyp1f1(a - delta, b, z, logger)
            derivative = (f_plus - f_minus) / (2 * delta)
            
        return derivative if np.isfinite(derivative) else 0.0
        
    except Exception as e:
        logger.warning(f"Derivative computation failed: {e}")
        return 0.0

def laplace_transform_G(params: ModelParams, H: float, y: float, logger) -> float:
    """Laplace transform of hitting time at intensity = H
    G(y, H) (H) = _hyp1f1(H/kappa, 2kappa*c/sigma^2, 2kappa*y/sigma^2) / _hyp1f1(H/kappa, 2kappa*c/sigma^2, 2kappa*H/sigma^2)
    """
    try:
        kappa = params.kappa
        theta = params.theta
        sigma = params.sigma

        a = H / kappa
        b = 2 * kappa * theta / sigma**2
        z1 = 2 * kappa * y / sigma**2
        z2 = 2 * kappa * H / sigma**2
        
        logger.debug(f"Laplace transform: a={a:.6f}, b={b:.6f}, z1={z1:.6f}, z2={z2:.6f}")
        
        numerator = _hyp1f1(a, b, z1, logger)
        denominator = _hyp1f1(a, b, z2, logger)
        
        if abs(denominator) < EPS_ZERO:
            logger.warning("Near-zero denominator in Laplace transform")
            return 0.99  # Safe fallback
            
        result = numerator / denominator
        
        # Stability check
        if not (0 <= result <= 1):
            logger.warning(f"Laplace transform out of bounds: {result:.6f}")
            result = np.clip(result, EPS_ZERO, 1 - EPS_ZERO)
            
        return result
        
    except Exception as e:
        logger.error(f"Laplace transform failed: {e}")
        return 0.5  # Conservative fallback