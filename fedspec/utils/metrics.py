"""
Metrics utilities for FedSpec experiments.
Includes Frobenius norm computation, accuracy, and communication cost tracking.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple


def frobenius_norm(matrix: torch.Tensor) -> float:
    """
    Compute Frobenius norm of a matrix.
    ||A||_F = sqrt(sum(a_ij^2))
    
    Args:
        matrix: Input tensor of shape (m, n)
    
    Returns:
        Frobenius norm as float
    """
    return torch.norm(matrix, p='fro').item()


def frobenius_gap(
    delta_w_ideal: torch.Tensor,
    delta_w_approx: torch.Tensor
) -> float:
    """
    Compute Frobenius gap between ideal and approximated weight update.
    Gap = ||ΔW_ideal - ΔW_approx||_F
    
    Args:
        delta_w_ideal: Ideal weight update, shape (d_out, d_in)
        delta_w_approx: Approximated weight update, shape (d_out, d_in)
    
    Returns:
        Frobenius gap as float
    """
    diff = delta_w_ideal - delta_w_approx
    return frobenius_norm(diff)


def relative_frobenius_error(
    delta_w_ideal: torch.Tensor,
    delta_w_approx: torch.Tensor
) -> float:
    """
    Compute relative Frobenius error.
    Error = ||ΔW_ideal - ΔW_approx||_F / ||ΔW_ideal||_F
    
    Args:
        delta_w_ideal: Ideal weight update, shape (d_out, d_in)
        delta_w_approx: Approximated weight update, shape (d_out, d_in)
    
    Returns:
        Relative error as float
    """
    gap = frobenius_gap(delta_w_ideal, delta_w_approx)
    ideal_norm = frobenius_norm(delta_w_ideal)
    if ideal_norm < 1e-10:
        return 0.0
    return gap / ideal_norm


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions, shape (batch_size,) or (batch_size, num_classes)
        labels: Ground truth labels, shape (batch_size,)
    
    Returns:
        Accuracy as float in [0, 1]
    """
    if predictions.dim() == 2:
        # predictions are logits, shape (batch_size, num_classes)
        predictions = predictions.argmax(dim=1)
    
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def compute_communication_bytes(
    lora_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> int:
    """
    Compute communication cost in bytes for transmitting LoRA matrices.
    
    Each LoRA layer has:
    - B matrix: shape (d_out, r)
    - A matrix: shape (r, d_in)
    
    Assuming float32 (4 bytes per element).
    
    Args:
        lora_matrices: Dict mapping layer names to (B, A) tuples
    
    Returns:
        Total bytes for transmission
    """
    bytes_per_float = 4  # float32
    total_elements = 0
    
    for layer_name, (B, A) in lora_matrices.items():
        # B: shape (d_out, r)
        # A: shape (r, d_in)
        total_elements += B.numel() + A.numel()
    
    return total_elements * bytes_per_float


def compute_tail_energy(singular_values: torch.Tensor, rank: int) -> Tuple[float, float]:
    """
    Compute tail energy ratio for adaptive rank selection.
    
    E_tail = sum(σ_{r+1:})
    E_total = sum(σ)
    ratio = E_tail / E_total
    
    Args:
        singular_values: Singular values from SVD, shape (min(m, n),)
        rank: Current truncation rank r
    
    Returns:
        Tuple of (tail_energy_ratio, total_energy)
    """
    total_energy = singular_values.sum().item()
    if total_energy < 1e-10:
        return 0.0, 0.0
    
    if rank >= len(singular_values):
        return 0.0, total_energy
    
    tail_energy = singular_values[rank:].sum().item()
    return tail_energy / total_energy, total_energy


class MetricsLogger:
    """
    Logger for tracking metrics across federated rounds.
    """
    
    def __init__(self):
        self.rounds: List[int] = []
        self.frobenius_gaps: List[float] = []
        self.ranks: List[int] = []
        self.accuracies: List[float] = []
        self.communication_bytes: List[int] = []
    
    def log(
        self,
        round_num: int,
        frobenius_gap: float,
        rank: int,
        accuracy: float,
        comm_bytes: int
    ) -> None:
        """Log metrics for a single round."""
        self.rounds.append(round_num)
        self.frobenius_gaps.append(frobenius_gap)
        self.ranks.append(rank)
        self.accuracies.append(accuracy)
        self.communication_bytes.append(comm_bytes)
    
    def to_dict(self) -> Dict[str, List]:
        """Convert logged metrics to dictionary."""
        return {
            "round": self.rounds,
            "frobenius_gap": self.frobenius_gaps,
            "rank": self.ranks,
            "accuracy": self.accuracies,
            "communication_bytes": self.communication_bytes
        }
    
    def save_csv(self, filepath: str) -> None:
        """Save metrics to CSV file."""
        import csv
        data = self.to_dict()
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data.keys())
            rows = zip(*data.values())
            writer.writerows(rows)
