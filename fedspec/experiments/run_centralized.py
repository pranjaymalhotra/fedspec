"""
Run centralized training experiment (baseline).
"""
import os
import sys
import argparse
import csv
from typing import List

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.seed import set_seed, get_device
from data.load_sst2 import load_sst2
from models.lora_bert import create_lora_bert, get_trainable_parameters
from baselines.centralized import train_centralized


def run_centralized_experiment(
    config: Config,
    num_epochs: int = 3,
    output_dir: str = 'logs'
) -> tuple:
    """
    Run centralized training experiment.
    
    This is the upper bound baseline - training on all data without
    federated constraints.
    
    Args:
        config: Configuration object
        num_epochs: Number of training epochs
        output_dir: Directory for output logs
    
    Returns:
        Tuple of (train_losses, val_losses, val_accuracies)
    """
    print("Running centralized training experiment")
    
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
    
    print(f"Train samples: {train_dataset['input_ids'].shape[0]}")
    print(f"Val samples: {val_dataset['input_ids'].shape[0]}")
    
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
    
    trainable_params = get_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print(f"\nTraining for {num_epochs} epochs...")
    train_losses, val_losses, val_accuracies = train_centralized(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        seed=config.seed,
        verbose=True
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'centralized.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
        for i in range(num_epochs):
            writer.writerow([
                i + 1,
                train_losses[i],
                val_losses[i],
                val_accuracies[i]
            ])
    
    print(f"\nResults saved to {csv_path}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    
    return train_losses, val_losses, val_accuracies


def main():
    parser = argparse.ArgumentParser(description='Run centralized training experiment')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='logs',
                       help='Output directory')
    parser.add_argument('--rank', type=int, default=8,
                       help='LoRA rank')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        seed=args.seed,
        lora_rank=args.rank
    )
    
    # Run experiment
    run_centralized_experiment(
        config=config,
        num_epochs=args.epochs,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
