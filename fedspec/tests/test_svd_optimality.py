"""
Test SVD optimality of FedSpec aggregation.

Verifies that FedSpec produces spectrally optimal rank-r approximation:
||ΔW_ideal - ΔW_fedspec||_F <= ||ΔW_ideal - ΔW_fedavg||_F

This follows from the Eckart-Young-Mirsky theorem: the truncated SVD
gives the best rank-r approximation in Frobenius norm.
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
from utils.lora_utils import compute_delta_w, reconstruct_lora_from_delta_w
from aggregators.fedavg import fedavg_aggregate
from aggregators.fedspec import fedspec_aggregate


class TestSVDOptimality(unittest.TestCase):
    """Test suite for verifying FedSpec's spectral optimality."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.d_out = 64  # Output dimension
        self.d_in = 64   # Input dimension
        self.rank = 8    # LoRA rank
        self.num_clients = 5
    
    def create_synthetic_lora_matrices(
        self, 
        num_clients: int,
        d_out: int,
        d_in: int,
        rank: int,
        heterogeneity: float = 0.5
    ):
        """
        Create synthetic LoRA matrices for testing.
        
        Higher heterogeneity = more different client updates.
        
        Args:
            num_clients: Number of clients
            d_out: Output dimension
            d_in: Input dimension
            rank: LoRA rank
            heterogeneity: Scale of per-client variation
        
        Returns:
            List of dicts mapping 'test_layer' to (B, A) tuples
        """
        client_matrices = []
        
        # Generate a base update
        B_base = torch.randn(d_out, rank) * 0.1
        A_base = torch.randn(rank, d_in) * 0.1
        
        for i in range(num_clients):
            # Add heterogeneous perturbation
            B_i = B_base + heterogeneity * torch.randn(d_out, rank)
            A_i = A_base + heterogeneity * torch.randn(rank, d_in)
            
            client_matrices.append({
                'test_layer': (B_i, A_i)
            })
        
        return client_matrices
    
    def test_fedspec_vs_fedavg_optimality(self):
        """
        Test that FedSpec achieves lower Frobenius gap than FedAvg.
        
        FedSpec minimizes ||ΔW_ideal - ΔW_agg||_F among all rank-r factorizations.
        FedAvg does NOT minimize this, so FedSpec should be at least as good.
        """
        for heterogeneity in [0.1, 0.5, 1.0, 2.0]:
            client_matrices = self.create_synthetic_lora_matrices(
                num_clients=self.num_clients,
                d_out=self.d_out,
                d_in=self.d_in,
                rank=self.rank,
                heterogeneity=heterogeneity
            )
            
            # Compute ideal delta W (true average of dense updates)
            delta_w_ideal = None
            for matrices in client_matrices:
                B_i, A_i = matrices['test_layer']
                delta_w_i = compute_delta_w(B_i, A_i)
                if delta_w_ideal is None:
                    delta_w_ideal = delta_w_i / self.num_clients
                else:
                    delta_w_ideal = delta_w_ideal + delta_w_i / self.num_clients
            
            # FedAvg aggregation
            fedavg_matrices = fedavg_aggregate(client_matrices)
            B_fedavg, A_fedavg = fedavg_matrices['test_layer']
            delta_w_fedavg = compute_delta_w(B_fedavg, A_fedavg)
            gap_fedavg = frobenius_gap(delta_w_ideal, delta_w_fedavg)
            
            # FedSpec aggregation
            fedspec_matrices, _, _ = fedspec_aggregate(
                client_matrices, rank=self.rank
            )
            B_fedspec, A_fedspec = fedspec_matrices['test_layer']
            delta_w_fedspec = compute_delta_w(B_fedspec, A_fedspec)
            gap_fedspec = frobenius_gap(delta_w_ideal, delta_w_fedspec)
            
            # FedSpec should have lower or equal Frobenius gap
            self.assertLessEqual(
                gap_fedspec, 
                gap_fedavg + 1e-6,  # Small tolerance for numerical errors
                f"FedSpec gap ({gap_fedspec:.6f}) should be <= FedAvg gap ({gap_fedavg:.6f}) "
                f"at heterogeneity={heterogeneity}"
            )
            
            print(f"Heterogeneity {heterogeneity}: "
                  f"FedAvg gap = {gap_fedavg:.6f}, "
                  f"FedSpec gap = {gap_fedspec:.6f}, "
                  f"Improvement = {(gap_fedavg - gap_fedspec) / gap_fedavg * 100:.2f}%")
    
    def test_svd_reconstruction_accuracy(self):
        """
        Test that SVD reconstruction is accurate.
        
        For a rank-r matrix, truncated SVD at rank >= r should be exact.
        """
        # Create exact rank-r matrix
        U = torch.randn(self.d_out, self.rank)
        V = torch.randn(self.rank, self.d_in)
        delta_w = torch.mm(U, V)  # Exact rank-r matrix
        
        # Reconstruct using SVD
        B_new, A_new, singular_values = reconstruct_lora_from_delta_w(
            delta_w, rank=self.rank
        )
        
        delta_w_reconstructed = compute_delta_w(B_new, A_new)
        
        # Should be very close to original
        gap = frobenius_gap(delta_w, delta_w_reconstructed)
        relative_error = gap / frobenius_norm(delta_w)
        
        self.assertLess(
            relative_error, 
            1e-5,
            f"SVD reconstruction error ({relative_error:.8f}) should be < 1e-5"
        )
        
        print(f"SVD reconstruction relative error: {relative_error:.2e}")
    
    def test_rank_truncation_energy_preservation(self):
        """
        Test that higher rank preserves more energy (lower approximation error).
        """
        # Create full-rank matrix
        delta_w = torch.randn(self.d_out, self.d_in)
        delta_w_norm = frobenius_norm(delta_w)
        
        previous_gap = float('inf')
        
        for r in [2, 4, 8, 16]:
            B_new, A_new, _ = reconstruct_lora_from_delta_w(delta_w, rank=r)
            delta_w_reconstructed = compute_delta_w(B_new, A_new)
            gap = frobenius_gap(delta_w, delta_w_reconstructed)
            
            # Gap should decrease with higher rank
            self.assertLess(
                gap, 
                previous_gap + 1e-6,
                f"Gap at rank {r} ({gap:.6f}) should be <= gap at lower rank ({previous_gap:.6f})"
            )
            previous_gap = gap
            
            print(f"Rank {r}: Frobenius gap = {gap:.6f}, "
                  f"Relative error = {gap/delta_w_norm:.4f}")
    
    def test_deterministic_svd(self):
        """
        Test that SVD produces deterministic results with same seed.
        """
        set_seed(42)
        delta_w = torch.randn(self.d_out, self.d_in)
        
        B1, A1, S1 = reconstruct_lora_from_delta_w(delta_w, rank=self.rank)
        B2, A2, S2 = reconstruct_lora_from_delta_w(delta_w, rank=self.rank)
        
        # Results should be identical
        self.assertTrue(
            torch.allclose(B1, B2, atol=1e-6),
            "B matrices should be identical for same input"
        )
        self.assertTrue(
            torch.allclose(A1, A2, atol=1e-6),
            "A matrices should be identical for same input"
        )
        self.assertTrue(
            torch.allclose(S1, S2, atol=1e-6),
            "Singular values should be identical for same input"
        )
        
        print("SVD is deterministic: PASS")
    
    def test_multiple_layers(self):
        """
        Test FedSpec with multiple LoRA layers.
        """
        num_layers = 4
        
        # Create multi-layer client matrices
        client_matrices = []
        for _ in range(self.num_clients):
            layer_dict = {}
            for l in range(num_layers):
                B = torch.randn(self.d_out, self.rank) * 0.1
                A = torch.randn(self.rank, self.d_in) * 0.1
                layer_dict[f'layer_{l}'] = (B, A)
            client_matrices.append(layer_dict)
        
        # Aggregate with FedSpec
        fedspec_matrices, delta_w_ideal_dict, _ = fedspec_aggregate(
            client_matrices, rank=self.rank
        )
        
        # Verify all layers are present
        self.assertEqual(
            len(fedspec_matrices), 
            num_layers,
            f"Should have {num_layers} layers"
        )
        
        # Verify shapes
        for layer_name, (B, A) in fedspec_matrices.items():
            self.assertEqual(
                B.shape, 
                (self.d_out, self.rank),
                f"B shape mismatch for {layer_name}"
            )
            self.assertEqual(
                A.shape, 
                (self.rank, self.d_in),
                f"A shape mismatch for {layer_name}"
            )
        
        print(f"Multi-layer FedSpec with {num_layers} layers: PASS")


if __name__ == '__main__':
    unittest.main(verbosity=2)
