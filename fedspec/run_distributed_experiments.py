"""
Distributed Paper Experiments Runner
=====================================
Coordinates experiments across Mac + Windows machines.
Each machine claims work items and Mac acts as primary coordinator.

SETUP:
1. Share the Fedspec folder via network (e.g., iCloud, Dropbox, or SMB share)
2. Run this script on both Mac and Windows simultaneously
3. Mac is primary - if Windows fails, Mac takes over its work

TARGET: Complete all experiments in 1.5-2 hours (vs 3-4 hours on Mac alone)
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.device_manager import get_best_device, print_system_info, get_optimal_batch_size
from utils.checkpoint import CheckpointManager, get_experiment_id
from utils.distributed import WorkDistributor
from experiments.run_federated import run_federated_experiment
from experiments.run_centralized import run_centralized_experiment


def run_work_item(work_item, device, output_dir, checkpoint_mgr):
    """
    Execute a single work item (experiment).
    
    Args:
        work_item: Work item dict with experiment parameters
        device: Compute device
        output_dir: Output directory
        checkpoint_mgr: Checkpoint manager instance
    
    Returns:
        Final accuracy or None if failed
    """
    alpha = work_item['alpha']
    method = work_item['method']
    split_type = work_item['split_type']
    
    print(f"\n{'='*70}")
    print(f"Experiment: {method.upper()}, alpha={alpha}, split={split_type}")
    print(f"{'='*70}")
    
    batch_size = get_optimal_batch_size(device)
    
    config = Config(
        seed=42,
        num_clients=10,
        num_rounds=20,
        local_epochs=1,
        batch_size=batch_size,
        lora_rank=8,
        max_rank=16,
        split_type=split_type,
        dirichlet_alpha=alpha,
        device=str(device)
    )
    
    exp_id = get_experiment_id(method, alpha, split_type)
    
    try:
        _, final_acc = run_federated_experiment(
            config=config,
            aggregation_method=method,
            output_dir=output_dir,
            checkpoint_manager=checkpoint_mgr,
            experiment_id=exp_id
        )
        return final_acc
    except Exception as e:
        print(f"\n✗ Experiment failed: {str(e)}")
        return None


def main():
    """Run distributed experiments."""
    print("=" * 70)
    print("FedSpec: Distributed Experiments (Mac + Windows)")
    print("=" * 70)
    print("\nThis machine will claim and execute experiments from shared queue.")
    print("Multiple machines can run this simultaneously.\n")
    
    start_time = time.time()
    
    # Setup
    print_system_info()
    device = get_best_device()
    
    # Initialize managers
    work_dir = "distributed_work"  # Should be on shared folder
    checkpoint_dir = "checkpoints_distributed"  # Should be on shared folder
    output_dir = "paper_results"  # Should be on shared folder
    
    for dir in [work_dir, checkpoint_dir, output_dir]:
        os.makedirs(dir, exist_ok=True)
    
    distributor = WorkDistributor(work_dir=work_dir)
    checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # Check if this is the first machine (initialize work queue)
    work_items = distributor.load_work_queue()
    if not work_items:
        print("\n→ Initializing work queue (first machine)")
        
        # Create work items
        alphas = [0.1, 0.5, 1.0]
        methods = ['fedavg', 'fedspec']
        split_types = ['dirichlet']
        
        work_items = distributor.create_work_items(alphas, methods, split_types)
        distributor.save_work_queue(work_items)
        
        print(f"✓ Created {len(work_items)} work items")
        print(f"  - 3 alphas: {alphas}")
        print(f"  - 2 methods: {methods}")
        print(f"  - Total experiments: {len(work_items)}")
    else:
        print(f"\n→ Joining existing work queue ({len(work_items)} items)")
    
    distributor.print_progress()
    
    # Main work loop
    experiments_completed = 0
    
    try:
        while True:
            # Claim next work item
            work_item = distributor.claim_work_item()
            
            if work_item is None:
                print("\n✓ No more work items available")
                distributor.print_progress()
                break
            
            print(f"\n→ Claimed work item {work_item['item_id']}")
            
            # Execute experiment
            final_acc = run_work_item(work_item, device, output_dir, checkpoint_mgr)
            
            # Mark completion
            if final_acc is not None:
                distributor.mark_completed(work_item['item_id'], success=True)
                experiments_completed += 1
                print(f"\n✓ Completed: {final_acc:.4f}")
            else:
                distributor.mark_failed(work_item['item_id'])
                print(f"\n✗ Failed - marked for retry")
            
            # Show progress
            distributor.print_progress()
            
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user!")
        print("Progress has been saved. Run again to resume.")
        distributor.print_progress()
        return
    
    # Check if Mac should pick up failed Windows work
    if distributor.machine_id == 'mac':
        print("\n" + "="*70)
        print("Mac: Checking for failed work items to retry...")
        print("="*70)
        
        work_items = distributor.load_work_queue()
        failed_items = [item for item in work_items if item['status'] == 'pending' and item.get('assigned_to') == 'windows']
        
        if failed_items:
            print(f"→ Found {len(failed_items)} items previously attempted by Windows")
            print("→ Mac will retry these items...")
            
            for work_item in failed_items:
                print(f"\n→ Retrying work item {work_item['item_id']}")
                final_acc = run_work_item(work_item, device, output_dir, checkpoint_mgr)
                
                if final_acc is not None:
                    distributor.mark_completed(work_item['item_id'], success=True)
                    experiments_completed += 1
                
                distributor.print_progress()
    
    # Run centralized baseline (Mac only)
    if distributor.machine_id == 'mac':
        print("\n" + "="*70)
        print("Mac: Running Centralized Baseline")
        print("="*70)
        
        batch_size = get_optimal_batch_size(device)
        config = Config(
            seed=42,
            batch_size=batch_size,
            lora_rank=8,
            device=str(device)
        )
        
        try:
            _, _, cent_accs = run_centralized_experiment(
                config=config,
                num_epochs=3,
                output_dir=output_dir
            )
            print(f"\n✓ Centralized baseline: {cent_accs[-1]:.4f}")
        except Exception as e:
            print(f"\n✗ Centralized baseline failed: {str(e)}")
    
    # Final summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("Distributed Experiments Complete!")
    print("="*70)
    print(f"\nMachine: {distributor.machine_id.upper()}")
    print(f"Experiments completed: {experiments_completed}")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
    
    distributor.print_progress()
    
    summary = distributor.get_progress_summary()
    if summary['completed'] == summary['total']:
        print("\n🎉 ALL EXPERIMENTS COMPLETE!")
        print(f"Results saved to: {output_dir}/")
    else:
        print(f"\n⚠ {summary['pending']} items still pending")
        print("Run again to continue or check for errors")
    
    print("="*70)


if __name__ == '__main__':
    main()
