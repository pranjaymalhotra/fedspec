"""
LoRA utilities for extracting and manipulating LoRA matrices.
Provides functions to extract B and A matrices from PEFT models.
"""
import torch
from typing import Dict, Tuple, List
from peft import PeftModel


def extract_lora_matrices(
    model: PeftModel,
    target_modules: Tuple[str, ...] = ("query", "value")
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract LoRA B and A matrices from a PEFT model.
    
    For each LoRA layer, extracts:
    - lora_B: shape (d_out, r) - the "up" projection
    - lora_A: shape (r, d_in) - the "down" projection
    
    The effective weight update is: ΔW = B @ A, shape (d_out, d_in)
    
    Args:
        model: PEFT model with LoRA adapters
        target_modules: Tuple of module name patterns to extract
    
    Returns:
        Dict mapping full layer names to (B, A) tuples
    """
    lora_matrices = {}
    
    for name, module in model.named_modules():
        # Check if this is a LoRA layer
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Check if module name contains any target pattern
            if any(target in name for target in target_modules):
                # Extract matrices from the default adapter
                # lora_A is stored as nn.Linear or nn.Parameter
                # In PEFT, lora_A has shape (r, d_in) and lora_B has shape (d_out, r)
                
                # Get the default adapter matrices
                if hasattr(module.lora_A, 'default'):
                    # PEFT stores adapters in a ModuleDict
                    A = module.lora_A['default'].weight.detach().clone()  # shape: (r, d_in)
                    B = module.lora_B['default'].weight.detach().clone()  # shape: (d_out, r)
                else:
                    A = module.lora_A.weight.detach().clone()
                    B = module.lora_B.weight.detach().clone()
                
                lora_matrices[name] = (B, A)
    
    return lora_matrices


def set_lora_matrices(
    model: PeftModel,
    lora_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    target_modules: Tuple[str, ...] = ("query", "value")
) -> None:
    """
    Set LoRA B and A matrices in a PEFT model.
    
    Args:
        model: PEFT model with LoRA adapters
        lora_matrices: Dict mapping layer names to (B, A) tuples
        target_modules: Tuple of module name patterns (for consistency)
    """
    for name, module in model.named_modules():
        if name in lora_matrices:
            B, A = lora_matrices[name]
            
            if hasattr(module.lora_A, 'default'):
                module.lora_A['default'].weight.data.copy_(A)
                module.lora_B['default'].weight.data.copy_(B)
            else:
                module.lora_A.weight.data.copy_(A)
                module.lora_B.weight.data.copy_(B)


def compute_delta_w(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute dense weight update from LoRA matrices.
    
    ΔW = B @ A
    
    Args:
        B: LoRA B matrix, shape (d_out, r)
        A: LoRA A matrix, shape (r, d_in)
    
    Returns:
        Dense weight update, shape (d_out, d_in)
    """
    # B: (d_out, r), A: (r, d_in)
    # Result: (d_out, d_in)
    return torch.mm(B, A)


def reconstruct_lora_from_delta_w(
    delta_w: torch.Tensor,
    rank: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruct LoRA matrices from dense weight update using truncated SVD.
    
    Given ΔW, compute rank-r truncated SVD:
    ΔW ≈ U_r @ Σ_r @ V_r^T
    
    Then reconstruct:
    B_new = U_r @ sqrt(Σ_r), shape (d_out, r)
    A_new = sqrt(Σ_r) @ V_r^T, shape (r, d_in)
    
    Args:
        delta_w: Dense weight update, shape (d_out, d_in)
        rank: Target rank r for truncation
    
    Returns:
        Tuple of (B_new, A_new, singular_values)
        - B_new: shape (d_out, r)
        - A_new: shape (r, d_in)
        - singular_values: all singular values, shape (min(d_out, d_in),)
    """
    # Move to CPU for SVD (more stable on CPU)
    delta_w_cpu = delta_w.cpu().float()
    
    # Full SVD: delta_w = U @ diag(S) @ V^T
    # U: (d_out, min(d_out, d_in))
    # S: (min(d_out, d_in),)
    # V^T: (min(d_out, d_in), d_in)
    U, S, Vh = torch.linalg.svd(delta_w_cpu, full_matrices=False)
    
    # Truncate to rank r
    # U_r: (d_out, r)
    # S_r: (r,)
    # Vh_r: (r, d_in)
    effective_rank = min(rank, len(S))
    U_r = U[:, :effective_rank]
    S_r = S[:effective_rank]
    Vh_r = Vh[:effective_rank, :]
    
    # Compute sqrt(Σ_r)
    sqrt_S_r = torch.sqrt(S_r)  # shape: (r,)
    
    # B_new = U_r @ diag(sqrt(Σ_r)), shape: (d_out, r)
    B_new = U_r * sqrt_S_r.unsqueeze(0)  # broadcasting: (d_out, r) * (1, r) = (d_out, r)
    
    # A_new = diag(sqrt(Σ_r)) @ Vh_r, shape: (r, d_in)
    A_new = sqrt_S_r.unsqueeze(1) * Vh_r  # broadcasting: (r, 1) * (r, d_in) = (r, d_in)
    
    return B_new, A_new, S


def get_lora_layer_names(
    model: PeftModel,
    target_modules: Tuple[str, ...] = ("query", "value")
) -> List[str]:
    """
    Get names of all LoRA layers in a model.
    
    Args:
        model: PEFT model with LoRA adapters
        target_modules: Tuple of module name patterns
    
    Returns:
        List of layer names containing LoRA adapters
    """
    layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if any(target in name for target in target_modules):
                layer_names.append(name)
    return layer_names


def count_lora_parameters(
    lora_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> int:
    """
    Count total number of parameters in LoRA matrices.
    
    Args:
        lora_matrices: Dict mapping layer names to (B, A) tuples
    
    Returns:
        Total parameter count
    """
    total = 0
    for name, (B, A) in lora_matrices.items():
        total += B.numel() + A.numel()
    return total
