"""
Configuration for FedSpec experiments.
All hyperparameters in one place for reproducibility.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # Random seed for reproducibility
    seed: int = 42
    
    # Model configuration
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    num_labels: int = 2  # SST-2 is binary classification
    
    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("query", "value")
    
    # FedSpec adaptive rank configuration
    max_rank: int = 16
    rank_increase_step: int = 2
    tail_energy_threshold: float = 0.05  # If E_tail / E_total > 0.05, increase rank
    
    # Federated learning configuration
    num_clients: int = 10
    num_rounds: int = 20
    local_epochs: int = 1
    participation_rate: float = 1.0  # Fraction of clients per round
    
    # Data configuration
    dirichlet_alpha: float = 0.5  # For non-IID splits (0.1, 0.5, 1.0)
    split_type: Literal["iid", "dirichlet"] = "dirichlet"
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    
    # Device configuration
    # MPS preferred for M2, CPU fallback for SVD operations
    device: str = "mps"  # Will be set dynamically based on availability
    
    # Logging configuration
    log_dir: str = "logs"
    save_csv: bool = True


# Default configuration instance
default_config = Config()
