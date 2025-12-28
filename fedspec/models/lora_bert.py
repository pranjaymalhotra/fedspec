"""
BERT model with LoRA adapters using PEFT.
Provides functions to create and manage LoRA-augmented BERT for sequence classification.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional


def create_lora_bert(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: tuple = ("query", "value"),
    device: Optional[torch.device] = None
):
    """
    Create BERT model with LoRA adapters for sequence classification.
    
    LoRA (Low-Rank Adaptation) adds trainable rank decomposition matrices
    to transformer attention layers while keeping original weights frozen.
    
    For each target module (e.g., query projection):
    - Original weight W: shape (d_out, d_in) = (768, 768) for BERT-base
    - LoRA adds: ΔW = B @ A
    - B: shape (d_out, r) = (768, r)
    - A: shape (r, d_in) = (r, 768)
    - Forward: h = W @ x + (B @ A) @ x
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classification labels (2 for SST-2)
        lora_rank: Rank r of LoRA decomposition
        lora_alpha: LoRA scaling factor (effective learning rate scaling)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Which attention modules to apply LoRA to
        device: Target device (MPS or CPU)
    
    Returns:
        PEFT model with LoRA adapters
    """
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Configure LoRA
    # target_modules for BERT: "query", "key", "value", "dense"
    # We target query and value as specified
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",  # Do not train biases
        inference_mode=False
    )
    
    # Create PEFT model
    model = get_peft_model(base_model, lora_config)
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model


def get_trainable_parameters(model) -> int:
    """
    Count trainable parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_parameters(model) -> int:
    """
    Count total parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def freeze_base_model(model) -> None:
    """
    Freeze all parameters except LoRA adapters.
    
    This is usually done automatically by PEFT, but provided
    for explicit control.
    
    Args:
        model: PEFT model with LoRA adapters
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def copy_lora_state(source_model, target_model) -> None:
    """
    Copy LoRA adapter weights from source to target model.
    
    Args:
        source_model: Source PEFT model
        target_model: Target PEFT model
    """
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    
    # Copy only LoRA parameters
    for key in target_state.keys():
        if "lora_" in key:
            target_state[key] = source_state[key].clone()
    
    target_model.load_state_dict(target_state)


def reset_lora_weights(model) -> None:
    """
    Reset LoRA weights to initial state.
    
    A matrices: initialized with Kaiming uniform
    B matrices: initialized with zeros
    
    This follows the original LoRA initialization scheme.
    
    Args:
        model: PEFT model with LoRA adapters
    """
    import math
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Reset A with Kaiming uniform
            if hasattr(module.lora_A, 'default'):
                weight_A = module.lora_A['default'].weight
                weight_B = module.lora_B['default'].weight
            else:
                weight_A = module.lora_A.weight
                weight_B = module.lora_B.weight
            
            # Kaiming uniform initialization for A
            # A: shape (r, d_in)
            fan_in = weight_A.shape[1]
            bound = math.sqrt(6.0 / fan_in)
            torch.nn.init.uniform_(weight_A, -bound, bound)
            
            # Zero initialization for B
            # B: shape (d_out, r)
            torch.nn.init.zeros_(weight_B)
