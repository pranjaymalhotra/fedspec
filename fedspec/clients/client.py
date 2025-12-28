"""
Federated learning client implementation.
Handles local training and LoRA matrix extraction.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Tuple, Optional
from peft import PeftModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lora_utils import extract_lora_matrices, set_lora_matrices
from data.load_sst2 import get_dataloader


class FederatedClient:
    """
    A federated learning client that performs local training
    and extracts LoRA matrices for aggregation.
    """
    
    def __init__(
        self,
        client_id: int,
        model: PeftModel,
        dataset: Dict,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        device: torch.device = torch.device("cpu"),
        seed: int = 42
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: PEFT model with LoRA adapters (will be cloned)
            dataset: Dict with 'input_ids', 'attention_mask', 'labels'
            batch_size: Local training batch size
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: Weight decay for regularization
            device: Target device (MPS or CPU)
            seed: Random seed for reproducibility
        """
        self.client_id = client_id
        self.device = device
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Store dataset
        self.dataset = dataset
        self.num_samples = dataset["input_ids"].shape[0]
        
        # Create dataloader
        self.dataloader = get_dataloader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            seed=seed + client_id  # Different seed per client for shuffling
        )
        
        # Store reference to model (will receive global model each round)
        self.model = model
    
    def train_local(
        self,
        num_epochs: int = 1,
        global_lora_matrices: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], float]:
        """
        Perform local training for specified number of epochs.
        
        Steps:
        1. If provided, set global LoRA matrices as starting point
        2. Train locally for num_epochs
        3. Extract and return updated LoRA matrices
        
        Args:
            num_epochs: Number of local training epochs
            global_lora_matrices: Optional global LoRA matrices to start from
        
        Returns:
            Tuple of (lora_matrices, average_loss)
            - lora_matrices: Dict mapping layer names to (B, A) tuples
            - average_loss: Average training loss over all batches
        """
        # Set global model if provided
        if global_lora_matrices is not None:
            set_lora_matrices(self.model, global_lora_matrices)
        
        self.model.train()
        self.model.to(self.device)
        
        # Setup optimizer (only for LoRA parameters)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids, attention_mask, labels = batch
                
                # Move to device
                # input_ids: (batch_size, max_length)
                # attention_mask: (batch_size, max_length)
                # labels: (batch_size,)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # outputs.logits: (batch_size, num_labels)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    avg_loss_so_far = total_loss / num_batches
                    print(f"    Client {self.client_id}, Epoch {epoch+1}/{num_epochs}, "
                          f"Batch {batch_idx+1}/{len(self.dataloader)}, "
                          f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss_so_far:.4f}")
        
        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Extract updated LoRA matrices
        lora_matrices = extract_lora_matrices(self.model)
        
        return lora_matrices, average_loss
    
    def evaluate(self, eval_dataset: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Evaluate model on dataset.
        
        Args:
            eval_dataset: Optional dataset to evaluate on. 
                         Uses client's local data if not provided.
        
        Returns:
            Tuple of (accuracy, loss)
        """
        if eval_dataset is None:
            eval_dataset = self.dataset
        
        eval_loader = get_dataloader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed
        )
        
        self.model.eval()
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids, attention_mask, labels = batch
                
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * labels.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        average_loss = total_loss / total if total > 0 else 0.0
        
        return accuracy, average_loss


def create_clients(
    num_clients: int,
    model: PeftModel,
    client_datasets: list,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> list:
    """
    Create multiple federated clients.
    
    Args:
        num_clients: Number of clients to create
        model: Base PEFT model (shared reference)
        client_datasets: List of datasets, one per client
        batch_size: Local training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Target device
        seed: Base random seed
    
    Returns:
        List of FederatedClient instances
    """
    clients = []
    for i in range(num_clients):
        client = FederatedClient(
            client_id=i,
            model=model,
            dataset=client_datasets[i],
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=seed
        )
        clients.append(client)
    
    return clients
