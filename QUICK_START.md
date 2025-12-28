# FedSpec: Quick Start Guide

## Choose Your Platform

### macOS (Apple M2) Setup
```bash
cd /path/to/Fedspec
bash setup_mac.sh
```

### Windows (GTX 1660 Ti) Setup
```cmd
cd C:\path\to\Fedspec
setup_windows.bat
```

## Running Experiments

### 1. Quick Sanity Test (5-10 minutes)
Tests that everything works correctly with minimal compute.

**Run:**
```bash
cd fedspec
python run_sanity_test.py
```

**What it does:**
- Tests FedAvg, FedSpec, and centralized baseline
- 3 clients, 3 rounds, 1 epoch per round
- IID data split (easiest case)
- Generates plots in `test_outputs/`

**Expected output:**
```
=== FedSpec Sanity Test ===
Device: cuda (NVIDIA GeForce GTX 1660 Ti, 6.00 GB)
Optimal batch size: 32

Testing FedAvg...
Round 1/3
Training 3 clients...
Client 0/3 starting...
Client 0, Epoch 1/1, Batch 10/420, Loss: 0.6931, Avg Loss: 0.6935
...
Client 0/3 finished. Avg loss: 0.6850
Aggregating 3 client updates...
Evaluating...
FedAvg - Round 1 - Accuracy: 52.41%

Testing FedSpec...
Round 1/3
...
FedSpec - Round 1 - Accuracy: 54.82%
Frobenius gap: 15.234, Rank: 6, Tail energy: 0.0450

Sanity test complete! Results in test_outputs/
```

### 2. Full Paper Experiments (2-4 hours on GTX 1660 Ti)
Comprehensive experiments for publication-quality results.

**Run:**
```bash
cd fedspec
python run_paper_experiments.py
```

**What it does:**
- Tests multiple heterogeneity levels (α = 0.1, 0.5, 1.0)
- Compares IID vs non-IID data splits
- 10 clients, 20 rounds per experiment
- Generates 7 publication-quality PDF plots
- Creates summary report with LaTeX tables

**Expected runtime:**
- GTX 1660 Ti: ~2-3 hours
- Apple M2: ~3-4 hours
- CPU: ~8-12 hours (not recommended)

**Output files in `paper_results/`:**
```
figure_1_frobenius_gap_alpha_0.1.pdf
figure_2_frobenius_gap_alpha_0.5.pdf
figure_3_frobenius_gap_alpha_1.0.pdf
figure_4_accuracy_comparison_alpha_0.5.pdf
figure_5_iid_vs_noniid_gap.pdf
figure_6_iid_vs_noniid_accuracy.pdf
figure_7_centralized_baseline.pdf
PAPER_RESULTS_SUMMARY.txt  # LaTeX tables
*.csv                       # Raw data
```

## Understanding the Output

### Training Progress
```
Client 0, Epoch 1/1, Batch 10/420, Loss: 0.6931, Avg Loss: 0.6935
                     │      │       │           └─ Moving average
                     │      │       └─ Current batch loss
                     │      └─ Batch number / total batches
                     └─ Current epoch / total epochs
```

### Aggregation Output
```
FedSpec aggregation complete.
Frobenius gap: 15.234    # Lower is better (FedSpec ≤ FedAvg)
Rank: 6                   # Adaptive rank (less than original)
Tail energy: 0.0450      # Energy in discarded components (<5%)
```

### Final Results
```
=== Experiment: FedAvg (alpha=0.5) ===
Final Accuracy: 85.32%
Avg Frobenius Gap: 234.567

=== Experiment: FedSpec (alpha=0.5) ===
Final Accuracy: 87.15%        # Higher accuracy
Avg Frobenius Gap: 156.789    # Lower gap (33% improvement)
```

## Monitoring GPU Usage

### Windows (GTX 1660 Ti)
```cmd
# In a separate terminal
nvidia-smi -l 2
```

### macOS (Apple M2)
```bash
# In a separate terminal
sudo powermetrics --samplers gpu_power -i1000
```

**Expected GPU usage:**
- Memory: 4-5 GB / 6 GB (GTX 1660 Ti)
- Utilization: 80-95%
- Temperature: 70-80°C

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in `fedspec/config.py`:
```python
batch_size: int = 16  # Reduce from 32
```

### Training appears stuck
It's not stuck! Each round takes time:
- Client training: ~30-60 seconds per client
- Aggregation: ~5-10 seconds
- Evaluation: ~10-20 seconds

Check the logs for progress messages every 10 batches.

### Slow performance
1. Verify GPU is being used (check nvidia-smi or system info in output)
2. Close other GPU applications
3. Use the sanity test first to verify setup

## Next Steps

1. **Run sanity test** to verify everything works
2. **Run paper experiments** for full results
3. **Check plots** in `paper_results/` directory
4. **Use summary report** `PAPER_RESULTS_SUMMARY.txt` for your paper

## File Structure After Running

```
Fedspec/
├── fedspec/
│   ├── test_outputs/          # Sanity test results
│   │   ├── sanity_test_metrics.pdf
│   │   └── *.csv
│   └── paper_results/         # Full experiment results
│       ├── figure_*.pdf       # 7 publication plots
│       ├── PAPER_RESULTS_SUMMARY.txt
│       └── *.csv              # Raw metrics
├── setup_mac.sh               # macOS setup
├── setup_windows.bat          # Windows setup
├── WINDOWS_SETUP.md           # Detailed Windows instructions
└── QUICK_START.md             # This file
```

## Key Configuration Files

- `fedspec/config.py` - Hyperparameters (batch size, learning rate, etc.)
- `fedspec/device_config.yaml` - Device-specific settings
- `fedspec/run_sanity_test.py` - Quick test configuration
- `fedspec/run_paper_experiments.py` - Full experiment suite

## Questions?

- Check logs for detailed error messages
- Verify GPU with `nvidia-smi` (Windows) or system info in output (Mac)
- Start with sanity test before running full experiments
- Ensure virtual environment is activated (`.venv\Scripts\activate` or `source .venv/bin/activate`)
