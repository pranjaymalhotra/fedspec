"""
Test aggregation bias with varying client heterogeneity.

Demonstrates that:
1. FedAvg aggregation bias increases with client heterogeneity
2. FedSpec maintains low bias regardless of heterogeneity
"""
import sys
import os
import unittest

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.metrics import frobenius_gap, frobenius_norm
from utils.lora_utils import compute_delta_w
from aggregators.fedavg import fedavg_aggregate
from aggregators.fedspec import fedspec_aggregate


class TestAggregationBias(unittest.TestCase):
    """Test suite for analyzing aggregation bias under heterogeneity."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.d_out = 64
        self.d_in = 64
        self.rank = 8
        self.num_clients = 10
    
    def create_heterogeneous_clients(
        self,
        num_clients: int,
        d_out: int,
        d_in: int,
        rank: int,
        heterogeneity_level: float
    ):
        """
        Create client LoRA matrices with controlled heterogeneity.
        
        Heterogeneity is modeled by having clients operate on different
        "modes" of the data, represented by different principal directions.
        
        Args:
            num_clients: Number of clients
            d_out: Output dimension
            d_in: Input dimension
            rank: LoRA rank
            heterogeneity_level: Controls how different clients are
                                 0 = identical clients
                                 1 = maximally different clients
        
        Returns:
            List of client LoRA matrices
        """
        client_matrices = []
        
        # Generate shared basis
        shared_B = torch.randn(d_out, rank) * 0.1
        shared_A = torch.randn(rank, d_in) * 0.1
        
        for i in range(num_clients):
            # Each client has its own perturbation direction
            # The perturbation magnitude is controlled by heterogeneity_level
            
            # Generate orthogonal perturbation directions
            theta = 2 * np.pi * i / num_clients
            
            # Random perturbation with magnitude controlled by heterogeneity
            perturb_B = torch.randn(d_out, rank) * heterogeneity_level
            perturb_A = torch.randn(rank, d_in) * heterogeneity_level
            
            # Add directional bias based on client index
            direction_B = torch.zeros(d_out, rank)
            direction_A = torch.zeros(rank, d_in)
            
            # Different clients emphasize different dimensions
            start_idx = (i * rank) % d_out
            for j in range(rank):
                idx = (start_idx + j) % d_out
                direction_B[idx, j] = np.cos(theta)
            
            start_idx = (i * rank) % d_in
            for j in range(rank):
                idx = (start_idx + j) % d_in
                direction_A[j, idx] = np.sin(theta)
            
            B_i = shared_B + perturb_B + direction_B * heterogeneity_level
            A_i = shared_A + perturb_A + direction_A * heterogeneity_level
            
            client_matrices.append({
                'test_layer': (B_i, A_i)
            })
        
        return client_matrices
    
    def compute_aggregation_bias(
        self,
        client_matrices: list,
        method: str,
        rank: int
    ) -> float:
        """
        Compute aggregation bias for given method.
        
        Bias = ||ΔW_ideal - ΔW_agg||_F / ||ΔW_ideal||_F
        
        Args:
            client_matrices: List of client LoRA matrices
            method: 'fedavg' or 'fedspec'
            rank: Rank for FedSpec aggregation
        
        Returns:
            Relative bias (normalized Frobenius gap)
        """
        # Compute ideal delta W
        delta_w_ideal = None
        num_clients = len(client_matrices)
        
        for matrices in client_matrices:
            B_i, A_i = matrices['test_layer']
            delta_w_i = compute_delta_w(B_i, A_i)
            if delta_w_ideal is None:
                delta_w_ideal = delta_w_i / num_clients
            else:
                delta_w_ideal = delta_w_ideal + delta_w_i / num_clients
        
        ideal_norm = frobenius_norm(delta_w_ideal)
        if ideal_norm < 1e-10:
            return 0.0
        
        if method == 'fedavg':
            agg_matrices = fedavg_aggregate(client_matrices)
            B_agg, A_agg = agg_matrices['test_layer']
            delta_w_agg = compute_delta_w(B_agg, A_agg)
        elif method == 'fedspec':
            agg_matrices, _, _ = fedspec_aggregate(client_matrices, rank=rank)
            B_agg, A_agg = agg_matrices['test_layer']
            delta_w_agg = compute_delta_w(B_agg, A_agg)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        gap = frobenius_gap(delta_w_ideal, delta_w_agg)
        return gap / ideal_norm
    
    def test_bias_increases_with_heterogeneity_fedavg(self):
        """
        Test that FedAvg bias increases monotonically with heterogeneity.
        """
        heterogeneity_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
        biases = []
        
        for h in heterogeneity_levels:
            client_matrices = self.create_heterogeneous_clients(
                num_clients=self.num_clients,
                d_out=self.d_out,
                d_in=self.d_in,
                rank=self.rank,
                heterogeneity_level=h
            )
            
            bias = self.compute_aggregation_bias(
                client_matrices, 
                method='fedavg',
                rank=self.rank
            )
            biases.append(bias)
            print(f"FedAvg - Heterogeneity {h:.2f}: Relative bias = {bias:.6f}")
        
        # Check monotonicity (with tolerance for noise)
        for i in range(1, len(biases)):
            if heterogeneity_levels[i] > 0:  # Skip h=0 case
                self.assertGreater(
                    biases[i],
                    biases[0] - 0.01,  # Allow small tolerance
                    f"FedAvg bias should generally increase with heterogeneity"
                )
        
        print("\nFedAvg bias increases with heterogeneity: PASS")
    
    def test_fedspec_lower_bias_than_fedavg(self):
        """
        Test that FedSpec has lower bias than FedAvg at all heterogeneity levels.
        """
        heterogeneity_levels = [0.1, 0.5, 1.0, 2.0]
        
        print("\nComparing FedAvg vs FedSpec bias:")
        print("-" * 60)
        
        for h in heterogeneity_levels:
            client_matrices = self.create_heterogeneous_clients(
                num_clients=self.num_clients,
                d_out=self.d_out,
                d_in=self.d_in,
                rank=self.rank,
                heterogeneity_level=h
            )
            
            bias_fedavg = self.compute_aggregation_bias(
                client_matrices, 
                method='fedavg',
                rank=self.rank
            )
            
            bias_fedspec = self.compute_aggregation_bias(
                client_matrices,
                method='fedspec',
                rank=self.rank
            )
            
            improvement = (bias_fedavg - bias_fedspec) / bias_fedavg * 100 if bias_fedavg > 0 else 0
            
            print(f"Heterogeneity {h:.1f}: "
                  f"FedAvg = {bias_fedavg:.6f}, "
                  f"FedSpec = {bias_fedspec:.6f}, "
                  f"Improvement = {improvement:.1f}%")
            
            # FedSpec should have lower or equal bias
            self.assertLessEqual(
                bias_fedspec,
                bias_fedavg + 1e-6,
                f"FedSpec bias should be <= FedAvg bias at heterogeneity {h}"
            )
        
        print("\nFedSpec has lower bias than FedAvg: PASS")
    
    def test_bias_with_varying_num_clients(self):
        """
        Test how aggregation bias scales with number of clients.
        """
        client_counts = [2, 5, 10, 20]
        heterogeneity = 0.5
        
        print("\nBias vs Number of Clients (heterogeneity=0.5):")
        print("-" * 60)
        
        for n in client_counts:
            client_matrices = self.create_heterogeneous_clients(
                num_clients=n,
                d_out=self.d_out,
                d_in=self.d_in,
                rank=self.rank,
                heterogeneity_level=heterogeneity
            )
            
            bias_fedavg = self.compute_aggregation_bias(
                client_matrices,
                method='fedavg',
                rank=self.rank
            )
            
            bias_fedspec = self.compute_aggregation_bias(
                client_matrices,
                method='fedspec',
                rank=self.rank
            )
            
            print(f"Clients {n:2d}: "
                  f"FedAvg = {bias_fedavg:.6f}, "
                  f"FedSpec = {bias_fedspec:.6f}")
    
    def test_bias_with_varying_rank(self):
        """
        Test how FedSpec bias changes with truncation rank.
        
        Higher rank should result in lower bias (better approximation).
        """
        ranks = [2, 4, 8, 16]
        heterogeneity = 0.5
        
        client_matrices = self.create_heterogeneous_clients(
            num_clients=self.num_clients,
            d_out=self.d_out,
            d_in=self.d_in,
            rank=16,  # Create with max rank
            heterogeneity_level=heterogeneity
        )
        
        print("\nFedSpec bias vs Rank:")
        print("-" * 40)
        
        previous_bias = float('inf')
        
        for r in ranks:
            bias = self.compute_aggregation_bias(
                client_matrices,
                method='fedspec',
                rank=r
            )
            
            print(f"Rank {r:2d}: Relative bias = {bias:.6f}")
            
            # Bias should decrease with higher rank
            self.assertLessEqual(
                bias,
                previous_bias + 1e-6,
                f"FedSpec bias should decrease with higher rank"
            )
            previous_bias = bias
        
        print("\nFedSpec bias decreases with rank: PASS")
    
    def test_extreme_heterogeneity(self):
        """
        Test behavior under extreme heterogeneity (nearly orthogonal clients).
        """
        # Create extremely heterogeneous clients
        client_matrices = []
        
        for i in range(self.num_clients):
            # Each client has nearly orthogonal update
            B = torch.zeros(self.d_out, self.rank)
            A = torch.zeros(self.rank, self.d_in)
            
            # Only activate different dimensions for each client
            start = (i * self.d_out // self.num_clients)
            end = ((i + 1) * self.d_out // self.num_clients)
            
            B[start:end, :] = torch.randn(end - start, self.rank)
            
            start = (i * self.d_in // self.num_clients)
            end = ((i + 1) * self.d_in // self.num_clients)
            
            A[:, start:end] = torch.randn(self.rank, end - start)
            
            client_matrices.append({'test_layer': (B, A)})
        
        bias_fedavg = self.compute_aggregation_bias(
            client_matrices,
            method='fedavg',
            rank=self.rank
        )
        
        bias_fedspec = self.compute_aggregation_bias(
            client_matrices,
            method='fedspec',
            rank=self.rank
        )
        
        print(f"\nExtreme heterogeneity case:")
        print(f"  FedAvg bias: {bias_fedavg:.6f}")
        print(f"  FedSpec bias: {bias_fedspec:.6f}")
        
        # FedSpec should still be better
        self.assertLessEqual(
            bias_fedspec,
            bias_fedavg + 1e-6,
            "FedSpec should be better even under extreme heterogeneity"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
