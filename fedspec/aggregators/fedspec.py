"""
FedSpec: Spectrally Optimal Aggregation for Federated LoRA.

This module implements the FedSpec algorithm which minimizes the Frobenius
norm between the ideal average of dense weight updates and the reconstructed
LoRA approximation.
"""
import torch
from typing import Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lora_utils import compute_delta_w, reconstruct_lora_from_delta_w
from utils.metrics import compute_tail_energy


class FedSpecState:
    """
    Maintains state for adaptive rank selection across rounds.
    Rank is monotonically increasing.
    """
    
    def __init__(
        self,
        initial_rank: int = 8,
        max_rank: int = 16,
        rank_increase_step: int = 2,
        tail_energy_threshold: float = 0.05
    ):
        """
        Initialize FedSpec state.
        
        Args:
            initial_rank: Starting rank for LoRA decomposition
            max_rank: Maximum allowed rank (cap)
            rank_increase_step: Amount to increase rank by when threshold exceeded
            tail_energy_threshold: If E_tail / E_total > threshold, increase rank
        """
        self.current_rank = initial_rank
        self.max_rank = max_rank
        self.rank_increase_step = rank_increase_step
        self.tail_energy_threshold = tail_energy_threshold
        self.rank_history = [initial_rank]
    
    def update_rank(self, tail_energy_ratio: float) -> int:
        """
        Update rank based on tail energy ratio.
        
        Adaptive rank rule:
        If E_tail / E_total > threshold:
            rank = min(rank + step, max_rank)
        
        Rank is monotonically non-decreasing.
        
        Args:
            tail_energy_ratio: E_tail / E_total from SVD
        
        Returns:
            New rank value
        """
        if tail_energy_ratio > self.tail_energy_threshold:
            new_rank = min(
                self.current_rank + self.rank_increase_step,
                self.max_rank
            )
            # Monotonicity: never decrease rank
            self.current_rank = max(self.current_rank, new_rank)
        
        self.rank_history.append(self.current_rank)
        return self.current_rank


def fedspec_aggregate(
    client_lora_matrices: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    rank: int,
    weights: List[float] = None
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    FedSpec aggregation: spectrally optimal averaging via SVD.
    
    Algorithm:
    1. For each client i, reconstruct dense update: ΔW_i = B_i @ A_i
    2. Compute ideal average: ΔW_ideal = mean_i(ΔW_i)
    3. Compute rank-r truncated SVD: ΔW_ideal ≈ U_r @ Σ_r @ V_r^T
    4. Reconstruct LoRA: B_new = U_r @ sqrt(Σ_r), A_new = sqrt(Σ_r) @ V_r^T
    
    This minimizes ||ΔW_ideal - ΔW_fedspec||_F among all rank-r factorizations
    (Eckart-Young-Mirsky theorem).
    
    Args:
        client_lora_matrices: List of dicts, each mapping layer names to (B, A) tuples
            - B_i: shape (d_out, r)
            - A_i: shape (r, d_in)
        rank: Target rank for truncated SVD
        weights: Optional per-client weights. If None, uniform weights are used.
    
    Returns:
        Tuple of:
        - aggregated_matrices: Dict mapping layer names to (B_new, A_new) tuples
        - delta_w_ideal: Dict mapping layer names to ideal dense updates
        - singular_values_dict: Dict mapping layer names to full singular values
    """
    num_clients = len(client_lora_matrices)
    
    if num_clients == 0:
        raise ValueError("No client matrices to aggregate")
    
    # Set uniform weights if not provided
    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Get layer names from first client
    layer_names = list(client_lora_matrices[0].keys())
    
    aggregated_matrices = {}
    delta_w_ideal_dict = {}
    singular_values_dict = {}
    
    for layer_name in layer_names:
        # Step 1: Reconstruct dense updates ΔW_i = B_i @ A_i for each client
        # Step 2: Compute weighted average ΔW_ideal
        delta_w_ideal = None
        
        for i, client_matrices in enumerate(client_lora_matrices):
            B_i, A_i = client_matrices[layer_name]
            # B_i: (d_out, r), A_i: (r, d_in)
            
            # ΔW_i = B_i @ A_i, shape: (d_out, d_in)
            delta_w_i = compute_delta_w(B_i, A_i)
            
            if delta_w_ideal is None:
                delta_w_ideal = weights[i] * delta_w_i
            else:
                delta_w_ideal = delta_w_ideal + weights[i] * delta_w_i
        
        # delta_w_ideal: (d_out, d_in)
        delta_w_ideal_dict[layer_name] = delta_w_ideal
        
        # Step 3 & 4: Truncated SVD and LoRA reconstruction
        # B_new: (d_out, r), A_new: (r, d_in), singular_values: (min(d_out, d_in),)
        B_new, A_new, singular_values = reconstruct_lora_from_delta_w(
            delta_w_ideal, rank
        )
        
        # Move back to same device as input
        original_device = client_lora_matrices[0][layer_name][0].device
        B_new = B_new.to(original_device)
        A_new = A_new.to(original_device)
        
        aggregated_matrices[layer_name] = (B_new, A_new)
        singular_values_dict[layer_name] = singular_values
    
    return aggregated_matrices, delta_w_ideal_dict, singular_values_dict


def compute_fedspec_metrics(
    aggregated_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    delta_w_ideal_dict: Dict[str, torch.Tensor],
    singular_values_dict: Dict[str, torch.Tensor],
    rank: int
) -> Dict[str, float]:
    """
    Compute FedSpec-specific metrics for analysis.
    
    Args:
        aggregated_matrices: FedSpec aggregated LoRA matrices
        delta_w_ideal_dict: Ideal dense weight updates
        singular_values_dict: Singular values from SVD
        rank: Current rank
    
    Returns:
        Dict with metrics:
        - total_frobenius_gap: Sum of Frobenius gaps across layers
        - avg_tail_energy_ratio: Average tail energy ratio across layers
        - max_tail_energy_ratio: Maximum tail energy ratio across layers
    """
    from utils.metrics import frobenius_gap
    
    total_gap = 0.0
    tail_ratios = []
    
    for layer_name in aggregated_matrices.keys():
        B_new, A_new = aggregated_matrices[layer_name]
        delta_w_ideal = delta_w_ideal_dict[layer_name]
        singular_values = singular_values_dict[layer_name]
        
        # Compute reconstructed dense update
        # ΔW_fedspec = B_new @ A_new
        delta_w_fedspec = compute_delta_w(B_new.cpu(), A_new.cpu())
        
        # Frobenius gap
        gap = frobenius_gap(delta_w_ideal.cpu(), delta_w_fedspec)
        total_gap += gap
        
        # Tail energy ratio
        tail_ratio, _ = compute_tail_energy(singular_values, rank)
        tail_ratios.append(tail_ratio)
    
    metrics = {
        "total_frobenius_gap": total_gap,
        "avg_tail_energy_ratio": sum(tail_ratios) / len(tail_ratios) if tail_ratios else 0.0,
        "max_tail_energy_ratio": max(tail_ratios) if tail_ratios else 0.0
    }
    
    return metrics


def fedspec_aggregate_with_adaptive_rank(
    client_lora_matrices: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    fedspec_state: FedSpecState,
    weights: List[float] = None
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], int, Dict[str, float]]:
    """
    FedSpec aggregation with adaptive rank selection.
    
    After aggregation, checks tail energy and potentially increases rank
    for next round.
    
    Args:
        client_lora_matrices: List of client LoRA matrices
        fedspec_state: FedSpecState object tracking rank across rounds
        weights: Optional per-client weights
    
    Returns:
        Tuple of:
        - aggregated_matrices: Dict mapping layer names to (B_new, A_new)
        - new_rank: Updated rank for next round
        - metrics: Dict with FedSpec metrics
    """
    # Aggregate with current rank
    aggregated_matrices, delta_w_ideal_dict, singular_values_dict = fedspec_aggregate(
        client_lora_matrices,
        rank=fedspec_state.current_rank,
        weights=weights
    )
    
    # Compute metrics
    metrics = compute_fedspec_metrics(
        aggregated_matrices,
        delta_w_ideal_dict,
        singular_values_dict,
        fedspec_state.current_rank
    )
    
    # Update rank based on max tail energy ratio across layers
    new_rank = fedspec_state.update_rank(metrics["max_tail_energy_ratio"])
    
    return aggregated_matrices, new_rank, metrics
