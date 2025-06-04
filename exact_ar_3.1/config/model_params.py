# exact_ar_3.1/config/model_params.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelParams:
    """Class for CIR model parameters with validation methods"""
    kappa: float  # Mean reversion speed
    theta: float  # Long-term mean
    sigma: float  # Volatility  
    gamma: Optional[float] = None  # Jump size multiplier
    
    def __post_init__(self):
        """Validate basic parameter constraints"""
        assert self.kappa > 0, "kappa must be positive"
        assert self.theta > 0, "theta must be positive"
        assert self.sigma > 0, "sigma must be positive"
    
    def check_feller_condition(self) -> bool:
        """Check if parameters satisfy Feller condition"""
        return 2 * self.kappa * self.theta >= self.sigma**2
    
    def __getitem__(self, key):
        return getattr(self, key)