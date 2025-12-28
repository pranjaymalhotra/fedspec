"""
FedSpec Demo: Minimal sanity check for FedSpec algorithm.

This demo:
1. Creates 3 fake clients with synthetic LoRA matrices
2. Runs 1 round of federated aggregation (both FedAvg and FedSpec)
3. Prints shapes, rank, Frobenius gap, and compares methods

Runtime target: <2 minutes on Apple M2
"""
import sys
import os
import time

import torch
import numpy as np

# Add the fedspec directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.seed import set_seed, get_device
from utils.metrics import frobenius_gap, frobenius_norm, compute_tail_energy
from utils.lora_utils import compute_delta_w, reconstruct_lora_from_delta_w
from aggregators.fedavg import fedavg_aggregate
from aggregators.fedspec import fedspec_aggregate, FedSpecState


def create_synthetic_client_matrices(
    num_clients: int,
    d_out: int,
    d_in: int,
    rank: int,
    heterogeneity: float = 0.5,
    seed: int = 42
):
    """
    Create synthetic LoRA matrices simulating different clients.
    
    This simulates what happens when clients train locally and produce
    different LoRA updates due to non-IID data.
    
    Args:
        num_clients: Number of clients
        d_out: Output dimension (768 for BERT hidden size)
        d_in: Input dimension (768 for BERT hidden size)
        rank: LoRA rank
        heterogeneity: Controls client divergence (higher = more different)
        seed: Random seed
    
    Returns:
        List of dicts, each containing LoRA matrices for two layers
        (simulating query and value projections)
    """
    set_seed(seed)
    
    client_matrices = []
    
    # Generate base matrices (shared starting point)
    # In real FL, this would be the global model at start of round
    B_base_q = torch.randn(d_out, rank) * 0.01
    A_base_q = torch.randn(rank, d_in) * 0.01
    B_base_v = torch.randn(d_out, rank) * 0.01
    A_base_v = torch.randn(rank, d_in) * 0.01
    
    for i in range(num_clients):
        # Each client's update is base + heterogeneous perturbation
        # This models local training on non-IID data
        
        # Query projection LoRA
        # B: (d_out, rank), A: (rank, d_in)
        B_q = B_base_q + heterogeneity * torch.randn(d_out, rank) * (0.5 + 0.5 * i / num_clients)
        A_q = A_base_q + heterogeneity * torch.randn(rank, d_in) * (0.5 + 0.5 * i / num_clients)
        
        # Value projection LoRA
        B_v = B_base_v + heterogeneity * torch.randn(d_out, rank) * (0.5 + 0.5 * i / num_clients)
        A_v = A_base_v + heterogeneity * torch.randn(rank, d_in) * (0.5 + 0.5 * i / num_clients)
        
        client_matrices.append({
            'layer_0_query': (B_q, A_q),
            'layer_0_value': (B_v, A_v)
        })
    
    return client_matrices


def compute_ideal_delta_w(
    client_matrices,
    layer_name: str,
    weights = None
):
    """
    Compute ideal average of dense weight updates.
    
    ΔW_ideal = (1/n) * sum_i(B_i @ A_i)
    
    This is what we ideally want to approximate.
    """
    num_clients = len(client_matrices)
    
    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    
    delta_w_ideal = None
    
    for i, matrices in enumerate(client_matrices):
        B_i, A_i = matrices[layer_name]
        # B_i: (d_out, rank), A_i: (rank, d_in)
        # delta_w_i: (d_out, d_in)
        delta_w_i = compute_delta_w(B_i, A_i)
        
        if delta_w_ideal is None:
            delta_w_ideal = weights[i] * delta_w_i
        else:
            delta_w_ideal = delta_w_ideal + weights[i] * delta_w_i
    
    return delta_w_ideal


def run_demo():
    """
    Run the FedSpec demo.
    """
    print("=" * 70)
    print("FedSpec Demo: Spectrally Optimal Aggregation for Federated LoRA")
    print("=" * 70)
    
    start_time = time.time()
    
    # Configuration
    num_clients = 3
    d_out = 768  # BERT hidden size
    d_in = 768
    rank = 8
    heterogeneity = 0.5
    seed = 42
    
    print(f"\nConfiguration:")
    print(f"  Clients: {num_clients}")
    print(f"  Dimensions: d_out={d_out}, d_in={d_in}")
    print(f"  LoRA rank: {rank}")
    print(f"  Heterogeneity: {heterogeneity}")
    print(f"  Seed: {seed}")
    
    # Get device
    device = get_device()
    print(f"  Device: {device}")
    
    # Create synthetic client matrices
    print("\n" + "-" * 70)
    print("Step 1: Creating synthetic client LoRA matrices")
    print("-" * 70)
    
    client_matrices = create_synthetic_client_matrices(
        num_clients=num_clients,
        d_out=d_out,
        d_in=d_in,
        rank=rank,
        heterogeneity=heterogeneity,
        seed=seed
    )
    
    # Print shapes
    print("\nClient matrices shapes:")
    for i, matrices in enumerate(client_matrices):
        for layer_name, (B, A) in matrices.items():
            print(f"  Client {i}, {layer_name}:")
            print(f"    B: {tuple(B.shape)} (d_out, rank)")
            print(f"    A: {tuple(A.shape)} (rank, d_in)")
    
    # Compute ideal delta W for one layer
    layer_name = 'layer_0_query'
    delta_w_ideal = compute_ideal_delta_w(client_matrices, layer_name)
    print(f"\nIdeal delta W shape: {tuple(delta_w_ideal.shape)} (d_out, d_in)")
    print(f"Ideal delta W Frobenius norm: {frobenius_norm(delta_w_ideal):.6f}")
    
    # Step 2: FedAvg aggregation
    print("\n" + "-" * 70)
    print("Step 2: FedAvg Aggregation")
    print("-" * 70)
    
    fedavg_matrices = fedavg_aggregate(client_matrices)
    
    B_fedavg, A_fedavg = fedavg_matrices[layer_name]
    delta_w_fedavg = compute_delta_w(B_fedavg, A_fedavg)
    
    gap_fedavg = frobenius_gap(delta_w_ideal, delta_w_fedavg)
    
    print(f"\nFedAvg results for {layer_name}:")
    print(f"  B_avg shape: {tuple(B_fedavg.shape)}")
    print(f"  A_avg shape: {tuple(A_fedavg.shape)}")
    print(f"  ΔW_fedavg = B_avg @ A_avg")
    print(f"  Frobenius gap ||ΔW_ideal - ΔW_fedavg||_F: {gap_fedavg:.6f}")
    
    # Step 3: FedSpec aggregation
    print("\n" + "-" * 70)
    print("Step 3: FedSpec Aggregation (SVD-based)")
    print("-" * 70)
    
    fedspec_state = FedSpecState(
        initial_rank=rank,
        max_rank=16,
        rank_increase_step=2,
        tail_energy_threshold=0.05
    )
    
    fedspec_matrices, delta_w_ideal_dict, singular_values_dict = fedspec_aggregate(
        client_matrices,
        rank=fedspec_state.current_rank
    )
    
    B_fedspec, A_fedspec = fedspec_matrices[layer_name]
    delta_w_fedspec = compute_delta_w(B_fedspec, A_fedspec)
    
    gap_fedspec = frobenius_gap(delta_w_ideal, delta_w_fedspec)
    
    # Get singular values for analysis
    singular_values = singular_values_dict[layer_name]
    tail_ratio, total_energy = compute_tail_energy(singular_values, rank)
    
    print(f"\nFedSpec results for {layer_name}:")
    print(f"  B_new shape: {tuple(B_fedspec.shape)}")
    print(f"  A_new shape: {tuple(A_fedspec.shape)}")
    print(f"  Current rank: {fedspec_state.current_rank}")
    print(f"  Top-{rank} singular values: {singular_values[:rank].numpy().round(4)}")
    print(f"  Tail energy ratio (E_tail/E_total): {tail_ratio:.4f}")
    print(f"  Frobenius gap ||ΔW_ideal - ΔW_fedspec||_F: {gap_fedspec:.6f}")
    
    # Step 4: Comparison
    print("\n" + "-" * 70)
    print("Step 4: Comparison")
    print("-" * 70)
    
    improvement = (gap_fedavg - gap_fedspec) / gap_fedavg * 100 if gap_fedavg > 0 else 0
    
    print(f"\nFrobenius Gap Comparison:")
    print(f"  FedAvg:  {gap_fedavg:.6f}")
    print(f"  FedSpec: {gap_fedspec:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    # Verify SVD optimality
    print(f"\nSVD Optimality Check:")
    print(f"  FedSpec gap <= FedAvg gap: {gap_fedspec <= gap_fedavg + 1e-6}")
    
    if gap_fedspec <= gap_fedavg + 1e-6:
        print("  PASS: FedSpec achieves spectrally optimal approximation")
    else:
        print("  WARNING: Unexpected result")
    
    # Adaptive rank check
    print(f"\nAdaptive Rank:")
    print(f"  Tail energy threshold: {fedspec_state.tail_energy_threshold}")
    print(f"  Current tail ratio: {tail_ratio:.4f}")
    
    if tail_ratio > fedspec_state.tail_energy_threshold:
        print(f"  Would increase rank from {fedspec_state.current_rank} to "
              f"{min(fedspec_state.current_rank + 2, fedspec_state.max_rank)}")
    else:
        print(f"  Rank stable at {fedspec_state.current_rank}")
    
    # Step 5: Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Demo Summary")
    print("=" * 70)
    print(f"\nDimensions:")
    print(f"  Input: B ∈ R^{{{d_out}×{rank}}}, A ∈ R^{{{rank}×{d_in}}}")
    print(f"  Dense: ΔW ∈ R^{{{d_out}×{d_in}}}")
    
    print(f"\nKey Results:")
    print(f"  FedAvg Frobenius gap: {gap_fedavg:.6f}")
    print(f"  FedSpec Frobenius gap: {gap_fedspec:.6f}")
    print(f"  FedSpec improvement: {improvement:.2f}%")
    print(f"  Current rank: {fedspec_state.current_rank}")
    
    print(f"\nRuntime: {elapsed:.2f} seconds")
    
    # Verify runtime target
    if elapsed < 120:
        print("PASS: Runtime under 2 minutes")
    else:
        print("WARNING: Runtime exceeded 2 minutes")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    return {
        'gap_fedavg': gap_fedavg,
        'gap_fedspec': gap_fedspec,
        'improvement': improvement,
        'rank': fedspec_state.current_rank,
        'runtime': elapsed
    }


if __name__ == '__main__':
    results = run_demo()
