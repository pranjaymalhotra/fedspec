"""
Centralized LoRA training baseline.
Trains on pooled SST-2 data (non-federated).
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Tuple, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_sst2 import get_dataloader
from utils.metrics import compute_accuracy


def train_centralized(
    model,
    train_dataset: Dict,
    val_dataset: Dict,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train model in centralized fashion on full dataset.
    
    This serves as an upper bound baseline - the best we can do
    without federated constraints.
    
    Args:
        model: PEFT model with LoRA adapters
        train_dataset: Training data dict with 'input_ids', 'attention_mask', 'labels'
        val_dataset: Validation data dict
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for regularization
        device: Target device (MPS or CPU)
        seed: Random seed for reproducibility
        verbose: Whether to print progress
    
    Returns:
        Tuple of (train_losses, val_losses, val_accuracies)
        - Each is a list with one value per epoch
    """
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        seed=seed
    )
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        seed=seed
    )
    
    model = model.to(device)
    
    # Setup optimizer (only LoRA parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            
            # Move to device
            # input_ids: (batch_size, max_length)
            # attention_mask: (batch_size, max_length)
            # labels: (batch_size,)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # outputs.logits: (batch_size, num_labels)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                total_val_loss += loss.item() * labels.size(0)
                
                predictions = logits.argmax(dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        avg_val_loss = total_val_loss / len(all_labels)
        val_accuracy = compute_accuracy(all_predictions, all_labels)
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, "
                  f"Val Acc = {val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies


def evaluate_model(
    model,
    dataset: Dict,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PEFT model to evaluate
        dataset: Dataset dict with 'input_ids', 'attention_mask', 'labels'
        batch_size: Evaluation batch size
        device: Target device
        seed: Random seed
    
    Returns:
        Tuple of (accuracy, loss)
    """
    dataloader = get_dataloader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        seed=seed
    )
    
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            
            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    accuracy = compute_accuracy(all_predictions, all_labels)
    avg_loss = total_loss / len(all_labels)
    
    return accuracy, avg_loss
