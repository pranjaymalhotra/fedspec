"""
Device manager for cross-platform support.
Automatically detects and selects best available device.
"""
import torch
import platform
from typing import Optional


def get_best_device(prefer_device: Optional[str] = None) -> torch.device:
    """
    Get the best available compute device.
    
    Priority:
    1. CUDA (NVIDIA GPU) - fastest for most operations
    2. MPS (Apple Silicon) - good for Mac M1/M2
    3. CPU - fallback, always available
    
    Args:
        prefer_device: Preferred device type ('cuda', 'mps', 'cpu')
    
    Returns:
        torch.device: Best available device
    """
    if prefer_device:
        if prefer_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return device
        elif prefer_device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using Apple MPS (Metal Performance Shaders)")
            return device
        elif prefer_device == "cpu":
            device = torch.device("cpu")
            print(f"Using CPU")
            return device
    
    # Auto-detect best device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Auto-detected: CUDA")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Auto-detected: Apple MPS")
        return device
    else:
        device = torch.device("cpu")
        print(f"Auto-detected: CPU (no GPU available)")
        return device


def print_system_info():
    """Print system information for debugging."""
    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("="*60 + "\n")


def get_optimal_batch_size(device: torch.device) -> int:
    """
    Get optimal batch size based on device.
    
    Args:
        device: Compute device
    
    Returns:
        Recommended batch size
    """
    if device.type == "cuda":
        # Get GPU memory
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / 1e9
        
        # GTX 1660 Ti has 6GB VRAM
        if memory_gb < 8:
            return 32  # 1660 Ti can handle 32 with BERT-base
        elif memory_gb < 16:
            return 64
        else:
            return 128
    elif device.type == "mps":
        # M2 typically has 8-16GB unified memory
        return 16  # Conservative for shared memory
    else:
        # CPU
        return 8  # Smaller batch for CPU
