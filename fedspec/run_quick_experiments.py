"""
Quick experiments with reduced configuration for faster results.
Target runtime: 30-60 minutes total

Reduces:
- Clients: 10 → 5
- Rounds: 20 → 10
- Batch size: Optimized for speed
- Single alpha: 0.5 (most representative)
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.device_manager import get_best_device, print_system_info, get_optimal_batch_size
from utils.checkpoint import CheckpointManager, get_experiment_id, should_resume
from experiments.run_federated import run_federated_experiment
from experiments.run_centralized import run_centralized_experiment


def main():
    """Run quick experiments with checkpointing."""
    print("=" * 70)
    print("FedSpec: Quick Experiments (30-60 min)")
    print("=" * 70)
    print("\nReduced configuration for faster validation:")
    print("- 5 clients (vs 10)")
    print("- 10 rounds (vs 20)")
    print("- Alpha = 0.5 only")
    print("- Checkpointing enabled")
    print()
    
    start_time = time.time()
    
    # Setup
    print_system_info()
    device = get_best_device()
    batch_size = get_optimal_batch_size(device)
    
    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_dir="checkpoints_quick")
    
    # Output directory
    output_dir = "quick_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    num_clients = 5
    num_rounds = 10
    alpha = 0.5
    
    print(f"\nConfiguration:")
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Alpha: {alpha}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Estimated time: 30-60 minutes")
    
    config = Config(
        seed=42,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=1,
        batch_size=batch_size,
        lora_rank=8,
        max_rank=16,
        split_type='dirichlet',
        dirichlet_alpha=alpha,
        device=str(device)
    )
    
    results = {}
    
    # Run experiments
    for method in ['fedavg', 'fedspec']:
        print("\n" + "=" * 70)
        print(f"Running: {method.upper()}")
        print("=" * 70)
        
        exp_id = get_experiment_id(method, alpha, 'dirichlet')
        
        # Check for checkpoint
        if should_resume(checkpoint_mgr, exp_id):
            print(f"→ Resuming {method.upper()} from checkpoint")
        
        try:
            _, final_acc = run_federated_experiment(
                config=config,
                aggregation_method=method,
                output_dir=output_dir,
                checkpoint_manager=checkpoint_mgr,
                experiment_id=exp_id
            )
            results[method] = final_acc
            print(f"\n✓ {method.upper()} completed: {final_acc:.4f}")
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted! Progress saved to checkpoint: {exp_id}")
            print("Run again to resume from checkpoint.")
            return
        except Exception as e:
            print(f"\n✗ {method.upper()} failed: {str(e)}")
            results[method] = None
    
    # Run centralized baseline
    print("\n" + "=" * 70)
    print("Running: Centralized Baseline")
    print("=" * 70)
    
    try:
        _, _, cent_accs = run_centralized_experiment(
            config=config,
            num_epochs=2,  # Reduced from 3
            output_dir=output_dir
        )
        results['centralized'] = cent_accs[-1]
        print(f"\n✓ Centralized: {cent_accs[-1]:.4f}")
    except Exception as e:
        print(f"\n✗ Centralized failed: {str(e)}")
        results['centralized'] = None
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Quick Experiments Complete!")
    print("=" * 70)
    print(f"\nRuntime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nResults:")
    for method, acc in results.items():
        if acc is not None:
            print(f"  {method.upper():<15}: {acc:.4f}")
    
    if results.get('fedavg') and results.get('fedspec'):
        improvement = (results['fedspec'] - results['fedavg']) / results['fedavg'] * 100
        print(f"\nFedSpec improvement: {improvement:+.2f}%")
    
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
