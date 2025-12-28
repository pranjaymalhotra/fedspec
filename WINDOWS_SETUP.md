# Windows Setup Instructions (GTX 1660 Ti)

## Prerequisites

1. **NVIDIA Driver**: Download latest driver from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)
2. **Python 3.10**: Download from [python.org](https://www.python.org/downloads/)
3. **CUDA Toolkit 11.8+**: Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

## Installation Steps

### 1. Install Python and CUDA

```cmd
# Verify Python installation
python --version

# Verify CUDA installation
nvcc --version
```

### 2. Clone Repository

```cmd
cd C:\Users\YourUsername\Documents
git clone <repo-url>
cd Fedspec
```

### 3. Create Virtual Environment

```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 4. Install PyTorch with CUDA Support

```cmd
# For CUDA 11.8 (GTX 1660 Ti compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Install Other Dependencies

```cmd
pip install transformers peft datasets numpy scipy matplotlib pytest
```

### 6. Verify CUDA Installation

```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce GTX 1660 Ti
```

## Running Experiments

### Quick Sanity Test (5-10 minutes)

```cmd
cd fedspec
python run_sanity_test.py
```

This will:
- Test all components
- Verify CUDA works
- Generate test plots
- Output to `test_outputs/`

### Full Paper Experiments (2-4 hours)

```cmd
cd fedspec
python run_paper_experiments.py
```

This will:
- Run comprehensive experiments
- Test multiple heterogeneity levels
- Generate publication-quality plots
- Output to `paper_results/`

### Run Individual Experiments

```cmd
# FedSpec with specific configuration
python experiments/run_federated.py --method fedspec --rounds 20 --clients 10 --alpha 0.5

# FedAvg baseline
python experiments/run_federated.py --method fedavg --rounds 20 --clients 10 --alpha 0.5

# Centralized baseline
python experiments/run_centralized.py --epochs 3
```

## Performance Tips for GTX 1660 Ti

1. **Batch Size**: The scripts automatically detect optimal batch size (32 for 1660 Ti)
2. **Memory Management**: Close other GPU applications during training
3. **Cooling**: Ensure good airflow, training is GPU-intensive
4. **Power Settings**: Set Windows to "High Performance" power mode

## Monitoring GPU Usage

### Using nvidia-smi

```cmd
# Check GPU status
nvidia-smi

# Monitor in real-time (updates every 2 seconds)
nvidia-smi -l 2
```

### Expected GPU Usage During Training

- GPU Memory: 4-5 GB / 6 GB
- GPU Utilization: 80-95%
- Temperature: 70-80°C (normal)

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in config
```python
# Edit fedspec/config.py
batch_size: int = 16  # Reduce from 32 to 16
```

### Issue: Slow Training

**Solution**: 
1. Check GPU is being used: `nvidia-smi`
2. Verify CUDA version matches: `torch.version.cuda`
3. Update NVIDIA drivers

### Issue: Import Errors

**Solution**: Reinstall packages
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets
```

## File Locations

After running experiments, results are in:

- **Sanity tests**: `Fedspec/fedspec/test_outputs/`
- **Paper results**: `Fedspec/fedspec/paper_results/`
- **Plots**: `.pdf` files in results directories
- **Metrics**: `.csv` files in results directories

## Next Steps

1. Run `run_sanity_test.py` first to verify setup
2. If sanity test passes, run `run_paper_experiments.py`
3. Check `paper_results/PAPER_RESULTS_SUMMARY.txt` for results
4. Use generated PDF plots in your paper

## Support

If you encounter issues:
1. Check `device_config.yaml` for device settings
2. Run demo.py to test core functionality
3. Check CUDA installation with `nvidia-smi`
