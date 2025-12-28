# FedSpec: Spectrally Optimal Aggregation for Federated LoRA

A PyTorch implementation of FedSpec, an algorithm for federated fine-tuning of large language models using LoRA (Low-Rank Adaptation) with spectrally optimal aggregation.

## Mathematical Foundation

Given client LoRA matrices:
- **B_i** ∈ ℝ^{d × r} (up projection)
- **A_i** ∈ ℝ^{r × d} (down projection)

FedSpec performs the following steps:

1. **Reconstruct dense updates**: ΔW_i = B_i @ A_i
2. **Compute ideal average**: ΔW_ideal = (1/n) Σ_i ΔW_i
3. **Truncated SVD**: ΔW_ideal ≈ U_r Σ_r V_r^T
4. **Reconstruct LoRA**: B_new = U_r @ √Σ_r, A_new = √Σ_r @ V_r^T

By the **Eckart-Young-Mirsky theorem**, this gives the optimal rank-r approximation in Frobenius norm.

### Adaptive Rank Rule

FedSpec includes adaptive rank selection:

```
E_tail = Σ_{i>r} σ_i    (tail energy)
E_total = Σ_i σ_i        (total energy)

if E_tail / E_total > 0.05:
    rank = min(rank + 2, max_rank)
```

The rank is monotonically non-decreasing.

## Project Structure

```
fedspec/
├── demo.py                     # Minimal sanity check (<2 min on M2)
├── config.py                   # All hyperparameters
├── utils/
│   ├── seed.py                 # Reproducibility utilities
│   ├── metrics.py              # Frobenius gap, accuracy, etc.
│   └── lora_utils.py           # LoRA matrix extraction/manipulation
├── data/
│   ├── load_sst2.py            # SST-2 dataset loading
│   └── federated_split.py      # IID and Dirichlet splits
├── models/
│   └── lora_bert.py            # BERT + LoRA using PEFT
├── clients/
│   └── client.py               # Federated client implementation
├── aggregators/
│   ├── fedavg.py               # FedAvg baseline
│   └── fedspec.py              # FedSpec (SVD-based)
├── baselines/
│   └── centralized.py          # Centralized training baseline
├── experiments/
│   ├── run_federated.py        # Full federated experiments
│   └── run_centralized.py      # Centralized baseline
├── tests/
│   ├── test_svd_optimality.py  # Verify spectral optimality
│   └── test_aggregation_bias.py # Analyze aggregation bias
└── plots/
    ├── plot_frobenius_gap.py   # Frobenius gap visualization
    └── plot_accuracy.py        # Accuracy comparison plots
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd fedspec

# Create virtual environment (Python 3.10 recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Demo

```bash
cd fedspec
python demo.py
```

This creates 3 synthetic clients, runs 1 round of aggregation, and compares FedSpec vs FedAvg.

Expected output:
- FedSpec achieves ~26% lower Frobenius gap than FedAvg
- Runtime: <1 second on M2

### Run Tests

```bash
cd fedspec
python -m pytest tests/ -v
```

### Run Full Experiment

```bash
# FedSpec with non-IID data (Dirichlet α=0.5)
python experiments/run_federated.py --method fedspec --split dirichlet --alpha 0.5

# FedAvg baseline
python experiments/run_federated.py --method fedavg --split dirichlet --alpha 0.5

# Centralized baseline
python experiments/run_centralized.py --epochs 3
```

### Generate Plots

```bash
python plots/plot_frobenius_gap.py \
    --fedspec logs/fedspec_dirichlet_alpha0.5.csv \
    --fedavg logs/fedavg_dirichlet_alpha0.5.csv \
    --output frobenius_gap.pdf

python plots/plot_accuracy.py \
    --fedspec logs/fedspec_dirichlet_alpha0.5.csv \
    --fedavg logs/fedavg_dirichlet_alpha0.5.csv \
    --centralized logs/centralized.csv \
    --output accuracy.pdf \
    --all
```

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 8 | Initial LoRA rank |
| `max_rank` | 16 | Maximum rank (cap) |
| `tail_energy_threshold` | 0.05 | Threshold for rank increase |
| `num_clients` | 10 | Number of federated clients |
| `num_rounds` | 20 | Number of federated rounds |
| `local_epochs` | 1 | Local training epochs per round |
| `dirichlet_alpha` | 0.5 | Dirichlet concentration (lower = more non-IID) |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 2e-4 | AdamW learning rate |

## Device Support

- **MPS (Apple Silicon)**: Preferred for training
- **CPU**: Fallback, used for SVD operations

The code automatically detects and uses MPS on M-series Macs.

## Metrics Logged

Per round:
- Frobenius gap: ||ΔW_ideal - ΔW_agg||_F
- Rank: Current LoRA rank
- Validation accuracy
- Communication bytes

All metrics are saved to CSV files.

## Baselines

1. **FedAvg-LoRA**: Averages B and A matrices separately (no SVD)
2. **Centralized**: Trains on pooled data (upper bound)

## Key Results

From synthetic experiments:
- FedSpec consistently achieves lower Frobenius gap than FedAvg
- Improvement increases with client heterogeneity
- SVD optimality is mathematically guaranteed (Eckart-Young-Mirsky theorem)

## Citation

```bibtex
@article{fedspec2024,
  title={FedSpec: Spectrally Optimal Aggregation for Federated LoRA},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License
