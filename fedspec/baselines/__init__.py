"""
Baselines package initialization.
"""
from .centralized import train_centralized, evaluate_model

__all__ = [
    "train_centralized",
    "evaluate_model"
]
