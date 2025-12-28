"""
Quick Sanity Test Script
========================
Runs minimal experiments to verify everything works.
Target runtime: 5-10 minutes on GTX 1660 Ti

This script tests:
1. Data loading and splitting
2. Model creation
3. FedAvg aggregation
4. FedSpec aggregation
5. Centralized baseline
6. Plotting functions

Use this when:
- Testing after code changes
- Verifying new system setup
- Quick debugging
"""
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.seed import set_seed
from utils.device_manager import get_best_device, print_system_info
from experiments.run_federated import run_federated_experiment
from experiments.run_centralized import run_centralized_experiment
from plots.plot_accuracy import plot_all_metrics


def run_sanity_tests():
    """
    Run quick sanity tests.
    """
    print("=" * 70)
    print("FedSpec Quick Sanity Test")
    print("=" * 70)
    print("\nThis will run minimal experiments to verify everything works.")
    print("Estimated time: 5-10 minutes\n")
    
    start_time = time.time()
    
    # Print system info
    print_system_info()
    
    # Get device
    device = get_best_device()
    
    # Create test configuration
    print("\n" + "-" * 70)
    print("Configuration")
    print("-" * 70)
    config = Config(
        seed=42,
        num_clients=3,           # Few clients
        num_rounds=3,            # Few rounds
        local_epochs=1,
        batch_size=32,           # Larger batch for speed
        lora_rank=4,             # Lower rank for speed
        split_type='iid',        # IID is simpler
        device=str(device)
    )
    
    print(f"Clients: {config.num_clients}")
    print(f"Rounds: {config.num_rounds}")
    print(f"Batch size: {config.batch_size}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Device: {config.device}")
    
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: FedAvg
    print("\n" + "=" * 70)
    print("Test 1: FedAvg Aggregation")
    print("=" * 70)
    
    try:
        _, fedavg_acc = run_federated_experiment(
            config=config,
            aggregation_method='fedavg',
            output_dir=output_dir
        )
        print(f"\n✓ FedAvg test PASSED. Final accuracy: {fedavg_acc:.4f}")
    except Exception as e:
        print(f"\n✗ FedAvg test FAILED: {str(e)}")
        raise
    
    # Test 2: FedSpec
    print("\n" + "=" * 70)
    print("Test 2: FedSpec Aggregation")
    print("=" * 70)
    
    try:
        _, fedspec_acc = run_federated_experiment(
            config=config,
            aggregation_method='fedspec',
            output_dir=output_dir
        )
        print(f"\n✓ FedSpec test PASSED. Final accuracy: {fedspec_acc:.4f}")
    except Exception as e:
        print(f"\n✗ FedSpec test FAILED: {str(e)}")
        raise
    
    # Test 3: Centralized baseline (1 epoch only)
    print("\n" + "=" * 70)
    print("Test 3: Centralized Baseline")
    print("=" * 70)
    
    try:
        _, _, centralized_accs = run_centralized_experiment(
            config=config,
            num_epochs=1,  # Just 1 epoch for quick test
            output_dir=output_dir
        )
        print(f"\n✓ Centralized test PASSED. Final accuracy: {centralized_accs[-1]:.4f}")
    except Exception as e:
        print(f"\n✗ Centralized test FAILED: {str(e)}")
        raise
    
    # Test 4: Plotting
    print("\n" + "=" * 70)
    print("Test 4: Plotting Functions")
    print("=" * 70)
    
    try:
        plot_all_metrics(
            fedspec_csv=f"{output_dir}/fedspec_iid_alpha0.5.csv",
            fedavg_csv=f"{output_dir}/fedavg_iid_alpha0.5.csv",
            centralized_csv=f"{output_dir}/centralized.csv",
            output_path=f"{output_dir}/sanity_test_metrics.pdf"
        )
        print(f"\n✓ Plotting test PASSED. Output: {output_dir}/sanity_test_metrics.pdf")
    except Exception as e:
        print(f"\n✗ Plotting test FAILED: {str(e)}")
        raise
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Sanity Test Summary")
    print("=" * 70)
    print(f"\n✓ All tests PASSED!")
    print(f"\nResults:")
    print(f"  FedAvg accuracy:       {fedavg_acc:.4f}")
    print(f"  FedSpec accuracy:      {fedspec_acc:.4f}")
    print(f"  Centralized accuracy:  {centralized_accs[-1]:.4f}")
    print(f"\nFedSpec improvement: {(fedspec_acc - fedavg_acc) / fedavg_acc * 100:+.2f}%")
    print(f"\nTotal runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Output directory: {output_dir}/")
    
    print("\n" + "=" * 70)
    print("System is ready for full experiments!")
    print("Run 'python run_paper_experiments.py' for comprehensive results.")
    print("=" * 70)


if __name__ == '__main__':
    run_sanity_tests()
