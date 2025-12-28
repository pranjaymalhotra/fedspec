"""
Seed utilities for deterministic reproducibility.
Sets all random seeds: Python, NumPy, PyTorch.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # For CUDA (not used on M2, but included for completeness)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Get the best available device.
    Prefers MPS (Apple Silicon), falls back to CPU.
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cpu_device() -> torch.device:
    """
    Get CPU device explicitly.
    Used for SVD operations that require CPU.
    
    Returns:
        torch.device: CPU device
    """
    return torch.device("cpu")
