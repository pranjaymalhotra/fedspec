"""
Run federated learning experiments with FedSpec and FedAvg.
"""
import os
import sys
import argparse
import csv
from typing import Dict, List, Tuple

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.seed import set_seed, get_device
from utils.metrics import (
    MetricsLogger,
    frobenius_gap,
    compute_communication_bytes
)
from utils.lora_utils import (
    extract_lora_matrices,
    set_lora_matrices,
    compute_delta_w
)
from data.load_sst2 import load_sst2
from data.federated_split import iid_split, dirichlet_split
from models.lora_bert import create_lora_bert
from clients.client import create_clients
from aggregators.fedavg import fedavg_aggregate, compute_fedavg_delta_w
from aggregators.fedspec import (
    FedSpecState,
    fedspec_aggregate,
    compute_fedspec_metrics
)
from baselines.centralized import evaluate_model


def run_federated_round(
    clients: list,
    global_lora_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    num_local_epochs: int,
    aggregation_method: str,
    fedspec_state: FedSpecState = None
) -> Tuple[Dict, Dict]:
    """
    Run one round of federated learning.
    
    Args:
        clients: List of FederatedClient instances
        global_lora_matrices: Current global LoRA matrices
        num_local_epochs: Number of local training epochs per client
        aggregation_method: 'fedavg' or 'fedspec'
        fedspec_state: FedSpecState for adaptive rank (required for fedspec)
    
    Returns:
        Tuple of (aggregated_matrices, round_metrics)
    """
    # Collect client updates
    client_matrices = []
    client_weights = []  # Proportional to dataset size
    
    print(f"  Training {len(clients)} clients...")
    for client_idx, client in enumerate(clients):
        print(f"  Client {client_idx}/{len(clients)} starting local training...")
        # Local training
        lora_matrices, avg_loss = client.train_local(
            num_epochs=num_local_epochs,
            global_lora_matrices=global_lora_matrices
        )
        print(f"  Client {client_idx}/{len(clients)} finished. Avg loss: {avg_loss:.4f}")
        client_matrices.append(lora_matrices)
        client_weights.append(client.num_samples)
    
    # Normalize weights
    total_samples = sum(client_weights)
    client_weights = [w / total_samples for w in client_weights]
    
    # Compute ideal delta W (for metrics)
    layer_names = list(client_matrices[0].keys())
    delta_w_ideal_dict = {}
    
    for layer_name in layer_names:
        delta_w_ideal = None
        for i, matrices in enumerate(client_matrices):
            B_i, A_i = matrices[layer_name]
            delta_w_i = compute_delta_w(B_i, A_i)
            if delta_w_ideal is None:
                delta_w_ideal = client_weights[i] * delta_w_i
            else:
                delta_w_ideal = delta_w_ideal + client_weights[i] * delta_w_i
        delta_w_ideal_dict[layer_name] = delta_w_ideal
    
    round_metrics = {}
    
    print(f"  Aggregating {len(client_matrices)} client updates...")
    if aggregation_method == 'fedavg':
        aggregated_matrices = fedavg_aggregate(client_matrices, client_weights)
        
        # Compute Frobenius gap for FedAvg
        total_gap = 0.0
        for layer_name in layer_names:
            B_avg, A_avg = aggregated_matrices[layer_name]
            delta_w_fedavg = compute_delta_w(B_avg, A_avg)
            gap = frobenius_gap(delta_w_ideal_dict[layer_name], delta_w_fedavg)
            total_gap += gap
        
        round_metrics['frobenius_gap'] = total_gap
        round_metrics['rank'] = client_matrices[0][layer_names[0]][0].shape[1]  # Original rank
        
        print(f"  FedAvg aggregation complete. Frobenius gap: {total_gap:.6f}")
        
    elif aggregation_method == 'fedspec':
        if fedspec_state is None:
            raise ValueError("fedspec_state required for FedSpec aggregation")
        
        aggregated_matrices, delta_w_ideal_agg, singular_values_dict = fedspec_aggregate(
            client_matrices,
            rank=fedspec_state.current_rank,
            weights=client_weights
        )
        
        metrics = compute_fedspec_metrics(
            aggregated_matrices,
            delta_w_ideal_agg,
            singular_values_dict,
            fedspec_state.current_rank
        )
        
        # Update rank based on tail energy
        fedspec_state.update_rank(metrics['max_tail_energy_ratio'])
        
        round_metrics['frobenius_gap'] = metrics['total_frobenius_gap']
        round_metrics['rank'] = fedspec_state.current_rank
        round_metrics['tail_energy_ratio'] = metrics['max_tail_energy_ratio']
        
        print(f"  FedSpec aggregation complete. Frobenius gap: {metrics['total_frobenius_gap']:.6f}, "
              f"Rank: {fedspec_state.current_rank}, Tail energy: {metrics['max_tail_energy_ratio']:.4f}")
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Communication cost
    round_metrics['communication_bytes'] = compute_communication_bytes(client_matrices[0]) * len(clients)
    
    return aggregated_matrices, round_metrics


def run_federated_experiment(
    config: Config,
    aggregation_method: str = 'fedspec',
    output_dir: str = 'logs',
    checkpoint_manager = None,
    experiment_id: str = None
) -> Tuple[MetricsLogger, float]:
    """
    Run complete federated learning experiment with checkpointing support.
    
    Args:
        config: Configuration object
        aggregation_method: 'fedavg' or 'fedspec'
        output_dir: Directory for output logs
        checkpoint_manager: CheckpointManager instance (optional)
        experiment_id: Unique experiment ID for checkpointing (optional)
    
    Returns:
        Tuple of (metrics_logger, final_accuracy)
    """
    print(f"Running federated experiment with {aggregation_method}")
    print(f"Config: {config.num_clients} clients, {config.num_rounds} rounds, "
          f"split={config.split_type}, alpha={config.dirichlet_alpha}")
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading SST-2 dataset...")
    train_dataset, val_dataset, _ = load_sst2(
        model_name=config.model_name,
        max_length=config.max_length,
        seed=config.seed
    )
    
    # Split data for clients
    print(f"Splitting data using {config.split_type} method...")
    if config.split_type == 'iid':
        client_datasets = iid_split(
            train_dataset,
            num_clients=config.num_clients,
            seed=config.seed
        )
    else:
        client_datasets = dirichlet_split(
            train_dataset,
            num_clients=config.num_clients,
            alpha=config.dirichlet_alpha,
            seed=config.seed
        )
    
    # Create model
    print("Creating LoRA-BERT model...")
    model = create_lora_bert(
        model_name=config.model_name,
        num_labels=config.num_labels,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        device=device
    )
    
    # Create clients
    print("Creating clients...")
    clients = create_clients(
        num_clients=config.num_clients,
        model=model,
        client_datasets=client_datasets,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        seed=config.seed
    )
    
    # Initialize global LoRA matrices
    global_lora_matrices = extract_lora_matrices(model)
    
    # Initialize FedSpec state if needed
    fedspec_state = None
    if aggregation_method == 'fedspec':
        fedspec_state = FedSpecState(
            initial_rank=config.lora_rank,
            max_rank=config.max_rank,
            rank_increase_step=config.rank_increase_step,
            tail_energy_threshold=config.tail_energy_threshold
        )
    
    # Metrics logger
    metrics_logger = MetricsLogger()
    
    # Check for checkpoint to resume from
    start_round = 0
    if checkpoint_manager and experiment_id:
        checkpoint_state = checkpoint_manager.load_experiment_state(experiment_id)
        if checkpoint_state:
            start_round = checkpoint_state.get('current_round', 0)
            metrics_logger = MetricsLogger()
            metrics_logger.rounds = checkpoint_state.get('rounds', [])
            metrics_logger.frobenius_gaps = checkpoint_state.get('frobenius_gaps', [])
            metrics_logger.ranks = checkpoint_state.get('ranks', [])
            metrics_logger.accuracies = checkpoint_state.get('accuracies', [])
            metrics_logger.comm_bytes = checkpoint_state.get('comm_bytes', [])
            
            # Load model state
            model_state = checkpoint_manager.load_model_state(experiment_id)
            if model_state:
                model.load_state_dict(model_state, strict=False)
                global_lora_matrices = extract_lora_matrices(model)
            
            print(f"✓ Resumed from round {start_round}/{config.num_rounds}")
    
    # Run federated rounds
    for round_num in range(start_round, config.num_rounds):
        print(f"\nRound {round_num + 1}/{config.num_rounds}")
        
        # Run round
        global_lora_matrices, round_metrics = run_federated_round(
            clients=clients,
            global_lora_matrices=global_lora_matrices,
            num_local_epochs=config.local_epochs,
            aggregation_method=aggregation_method,
            fedspec_state=fedspec_state
        )
        
        # Update global model
        set_lora_matrices(model, global_lora_matrices)
        
        # Evaluate on validation set
        print(f"  Evaluating on validation set...")
        accuracy, val_loss = evaluate_model(
            model=model,
            dataset=val_dataset,
            batch_size=config.batch_size,
            device=device,
            seed=config.seed
        )
        
        # Log metrics
        metrics_logger.log(
            round_num=round_num + 1,
            frobenius_gap=round_metrics['frobenius_gap'],
            rank=round_metrics['rank'],
            accuracy=accuracy,
            comm_bytes=round_metrics['communication_bytes']
        )
        
        print(f"  Frobenius gap: {round_metrics['frobenius_gap']:.6f}")
        print(f"  Rank: {round_metrics['rank']}")
        print(f"  Val accuracy: {accuracy:.4f}")
        
        # Save checkpoint after each round
        if checkpoint_manager and experiment_id:
            checkpoint_state = {
                'current_round': round_num + 1,
                'total_rounds': config.num_rounds,
                'progress': f"{round_num + 1}/{config.num_rounds}",
                'rounds': metrics_logger.rounds,
                'frobenius_gaps': metrics_logger.frobenius_gaps,
                'ranks': metrics_logger.ranks,
                'accuracies': metrics_logger.accuracies,
                'comm_bytes': metrics_logger.comm_bytes
            }
            checkpoint_manager.save_experiment_state(
                experiment_id=experiment_id,
                state=checkpoint_state,
                model_state=model.state_dict()
            )
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir, 
        f"{aggregation_method}_{config.split_type}_alpha{config.dirichlet_alpha}.csv"
    )
    metrics_logger.save_csv(csv_path)
    print(f"\nMetrics saved to {csv_path}")
    
    final_accuracy = metrics_logger.accuracies[-1] if metrics_logger.accuracies else 0.0
    return metrics_logger, final_accuracy


def main():
    parser = argparse.ArgumentParser(description='Run federated learning experiment')
    parser.add_argument('--method', type=str, default='fedspec', 
                       choices=['fedavg', 'fedspec'],
                       help='Aggregation method')
    parser.add_argument('--split', type=str, default='dirichlet',
                       choices=['iid', 'dirichlet'],
                       help='Data split type')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha (for non-IID)')
    parser.add_argument('--clients', type=int, default=10,
                       help='Number of clients')
    parser.add_argument('--rounds', type=int, default=20,
                       help='Number of federated rounds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='logs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        seed=args.seed,
        num_clients=args.clients,
        num_rounds=args.rounds,
        split_type=args.split,
        dirichlet_alpha=args.alpha
    )
    
    # Run experiment
    run_federated_experiment(
        config=config,
        aggregation_method=args.method,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
