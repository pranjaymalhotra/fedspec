"""
FedAvg aggregation for LoRA matrices.
Averages B and A matrices separately across clients.
"""
import torch
from typing import Dict, List, Tuple


def fedavg_aggregate(
    client_lora_matrices: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    weights: List[float] = None
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    FedAvg aggregation: average B and A matrices separately.
    
    For each LoRA layer:
    B_global = sum(w_i * B_i) / sum(w_i)
    A_global = sum(w_i * A_i) / sum(w_i)
    
    Note: This does NOT minimize ||ΔW_ideal - ΔW_agg||_F
    because (avg(B_i)) @ (avg(A_i)) != avg(B_i @ A_i) in general.
    
    Args:
        client_lora_matrices: List of dicts, each mapping layer names to (B, A) tuples
            - B_i: shape (d_out, r)
            - A_i: shape (r, d_in)
        weights: Optional per-client weights (e.g., proportional to dataset size).
                 If None, uniform weights are used.
    
    Returns:
        Dict mapping layer names to aggregated (B_avg, A_avg) tuples
    """
    num_clients = len(client_lora_matrices)
    
    if num_clients == 0:
        raise ValueError("No client matrices to aggregate")
    
    # Set uniform weights if not provided
    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Get layer names from first client
    layer_names = list(client_lora_matrices[0].keys())
    
    aggregated_matrices = {}
    
    for layer_name in layer_names:
        # Collect all client B and A matrices for this layer
        # B_i: shape (d_out, r)
        # A_i: shape (r, d_in)
        
        # Weighted sum of B matrices
        B_sum = None
        for i, client_matrices in enumerate(client_lora_matrices):
            B_i, _ = client_matrices[layer_name]
            if B_sum is None:
                B_sum = weights[i] * B_i.clone()
            else:
                B_sum = B_sum + weights[i] * B_i
        
        # Weighted sum of A matrices
        A_sum = None
        for i, client_matrices in enumerate(client_lora_matrices):
            _, A_i = client_matrices[layer_name]
            if A_sum is None:
                A_sum = weights[i] * A_i.clone()
            else:
                A_sum = A_sum + weights[i] * A_i
        
        # Store aggregated matrices
        # B_avg: shape (d_out, r)
        # A_avg: shape (r, d_in)
        aggregated_matrices[layer_name] = (B_sum, A_sum)
    
    return aggregated_matrices


def compute_fedavg_delta_w(
    aggregated_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Compute dense weight updates from FedAvg aggregated LoRA matrices.
    
    For each layer:
    ΔW_fedavg = B_avg @ A_avg
    
    Args:
        aggregated_matrices: Dict mapping layer names to (B_avg, A_avg) tuples
    
    Returns:
        Dict mapping layer names to dense weight updates
    """
    delta_w_dict = {}
    
    for layer_name, (B, A) in aggregated_matrices.items():
        # B: (d_out, r), A: (r, d_in)
        # ΔW: (d_out, d_in)
        delta_w = torch.mm(B, A)
        delta_w_dict[layer_name] = delta_w
    
    return delta_w_dict
