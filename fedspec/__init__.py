"""
FedSpec: Spectrally Optimal Aggregation for Federated LoRA

This package implements FedSpec, an algorithm that minimizes the Frobenius
norm between the ideal average of dense weight updates and the reconstructed
LoRA approximation using truncated SVD.

Mathematical Foundation:
    Given client LoRA matrices B_i ∈ R^{d×r} and A_i ∈ R^{r×d}:
    
    1. Reconstruct dense updates: ΔW_i = B_i @ A_i
    2. Compute ideal average: ΔW_ideal = mean_i(ΔW_i)
    3. Truncated SVD: ΔW_ideal ≈ U_r @ Σ_r @ V_r^T
    4. Reconstruct LoRA: B_new = U_r @ sqrt(Σ_r), A_new = sqrt(Σ_r) @ V_r^T
    
    By Eckart-Young-Mirsky theorem, this gives the optimal rank-r approximation
    in Frobenius norm.

Usage:
    # Quick demo
    python demo.py
    
    # Run federated experiment
    python experiments/run_federated.py --method fedspec --rounds 20
    
    # Run tests
    python -m pytest tests/

Author: FedSpec Research Team
"""

__version__ = "0.1.0"
__author__ = "FedSpec Research Team"

from .config import Config, default_config

__all__ = [
    "Config",
    "default_config",
    "__version__"
]
