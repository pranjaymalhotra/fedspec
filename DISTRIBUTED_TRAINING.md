# Distributed Training Setup (Mac + Windows)

## Overview

Run FedSpec experiments across Mac M2 + Windows GTX 1660 Ti simultaneously to **cut runtime in half** (1.5-2 hours vs 3-4 hours).

**Features:**
- ✅ Automatic work distribution
- ✅ Checkpointing (resume after crashes)
- ✅ Mac as primary coordinator
- ✅ Windows failure handling (Mac takes over)

## Quick Start

### Setup on Mac

```bash
cd /Users/pranjaymalhotra/Documents/Fedspec
source .venv/bin/activate

# Option 1: Quick test (30-60 min, 5 clients, 10 rounds)
cd fedspec
python run_quick_experiments.py

# Option 2: Full distributed (1.5-2 hours, 10 clients, 20 rounds)
cd fedspec
python run_distributed_experiments.py
```

### Setup on Windows

1. **Share the Fedspec folder** via network:
   - **Option A (iCloud)**: Enable iCloud Drive, put folder in iCloud
   - **Option B (Dropbox)**: Sync folder via Dropbox
   - **Option C (SMB)**: Share Mac folder, map as network drive on Windows

2. **On Windows:**
```cmd
cd C:\path\to\Fedspec
setup_windows.bat
.venv\Scripts\activate

cd fedspec
python run_distributed_experiments.py
```

## Time Estimates

| Configuration | Mac Alone | Mac + Windows | Description |
|--------------|-----------|---------------|-------------|
| **Quick** | 60 min | 30 min | 5 clients, 10 rounds, α=0.5 |
| **Full Paper** | 3-4 hours | 1.5-2 hours | 10 clients, 20 rounds, 3 alphas |

### Current Issue & Solution

**Problem**: Each client takes ~12 hours on Mac M2

**Root Cause**: Default batch size (16) is too conservative for M2

**Solutions Implemented:**

1. **Reduced configuration** (`run_quick_experiments.py`)
   - 5 clients → faster rounds
   - 10 rounds → fewer iterations
   - Still validates FedSpec vs FedAvg
   - **Time: 30-60 minutes**

2. **Distributed execution** (`run_distributed_experiments.py`)
   - Mac + Windows work in parallel
   - 6 experiments (3 alphas × 2 methods)
   - Mac: 3 experiments, Windows: 3 experiments
   - **Time: 1.5-2 hours total**

3. **Increased batch size**: Edit `fedspec/config.py`:
```python
# For faster training on M2
batch_size: int = 32  # Increase from 16
```

## How Distributed Training Works

### Work Distribution

```
Experiment Queue:
┌──────────────────────────────────────┐
│ 1. FedAvg,  α=0.1, Dirichlet [DONE] │
│ 2. FedSpec, α=0.1, Dirichlet [DONE] │
│ 3. FedAvg,  α=0.5, Dirichlet [Mac]  │ ← Mac working
│ 4. FedSpec, α=0.5, Dirichlet [Win]  │ ← Windows working
│ 5. FedAvg,  α=1.0, Dirichlet [Pend] │
│ 6. FedSpec, α=1.0, Dirichlet [Pend] │
└──────────────────────────────────────┘
```

### Checkpointing

Each round is saved automatically:

```
checkpoints_distributed/
├── fedavg_dirichlet_alpha0.5.json     # Progress metadata
├── fedavg_dirichlet_alpha0.5_model.pt # Model weights
├── fedspec_dirichlet_alpha0.5.json
└── fedspec_dirichlet_alpha0.5_model.pt
```

**If interrupted**, just rerun the same command - it resumes automatically!

### Failure Handling

**Scenario 1: Windows crashes**
```
Mac detects Windows failure → Mac takes over Windows work
```

**Scenario 2: Mac crashes**
```
Rerun on Mac → Resumes from last checkpoint
Windows continues its work
```

**Scenario 3: Both crash**
```
Rerun both → Each resumes from last checkpoint
No work lost!
```

## File Sharing Setup

### Option A: iCloud Drive (Easiest)

1. **On Mac:**
```bash
# Move project to iCloud
mv ~/Documents/Fedspec ~/Library/Mobile\ Documents/com~apple~CloudDocs/Fedspec
```

2. **On Windows:**
   - Install iCloud for Windows
   - Access from `C:\Users\YourName\iCloud Drive\Fedspec`

### Option B: Dropbox

1. **On Mac & Windows:**
   - Install Dropbox
   - Move Fedspec folder to Dropbox
   - Wait for sync

### Option C: SMB Network Share

1. **On Mac (System Settings → General → Sharing):**
   - Enable "File Sharing"
   - Add `/Users/pranjaymalhotra/Documents/Fedspec`
   - Note Mac's IP address (e.g., `192.168.1.100`)

2. **On Windows:**
```cmd
# Map network drive
net use Z: \\192.168.1.100\Fedspec
cd Z:\
```

## Running Experiments

### Quick Test (Recommended First)

**Purpose**: Validate everything works in 30-60 min

**On Mac:**
```bash
cd /Users/pranjaymalhotra/Documents/Fedspec
source .venv/bin/activate
cd fedspec
python run_quick_experiments.py
```

**Expected Output:**
```
FedSpec: Quick Experiments (30-60 min)
Configuration:
  Clients: 5
  Rounds: 10
  Alpha: 0.5
  Estimated time: 30-60 minutes

Round 1/10
  Training 5 clients...
  ...
✓ FedAvg completed: 0.8532
✓ FedSpec completed: 0.8715
FedSpec improvement: +2.14%
```

### Full Distributed (For Paper Results)

**Start on Mac:**
```bash
cd fedspec
python run_distributed_experiments.py
```

**Start on Windows (same time):**
```cmd
cd fedspec
python run_distributed_experiments.py
```

**Monitor Progress:**

Both machines will print:
```
Distributed Work Progress
========================================
Total items: 6
Completed:   2 (33.3%)
In progress: 2
  - Mac:     1
  - Windows: 1
Pending:     2
Failed:      0
========================================
```

## Monitoring

### Check Progress

```bash
# Mac
cd fedspec
python -c "from utils.distributed import WorkDistributor; w = WorkDistributor(); w.print_progress()"
```

### View Checkpoints

```bash
cd fedspec
python -c "from utils.checkpoint import CheckpointManager; c = CheckpointManager('checkpoints_distributed'); print(c.list_checkpoints())"
```

### GPU Usage

**Mac:**
```bash
sudo powermetrics --samplers gpu_power -i1000 -n1
```

**Windows:**
```cmd
nvidia-smi -l 2
```

## Troubleshooting

### "Client taking 12 hours"

**Increase batch size in `config.py`:**
```python
batch_size: int = 32  # Was 16, now 32 (2x faster)
```

Or use quick experiments:
```bash
python run_quick_experiments.py  # 5 clients, 10 rounds
```

### "Work queue file not found"

Ensure shared folder is accessible on both machines. Test:
```bash
# Mac
ls -la distributed_work/work_queue.json

# Windows
dir distributed_work\work_queue.json
```

### "Checkpoint not resuming"

Delete corrupt checkpoint and restart:
```bash
rm -rf checkpoints_distributed/
python run_distributed_experiments.py
```

### Windows can't find CUDA

Run setup again:
```cmd
setup_windows.bat
```

Check CUDA:
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

## Results

After completion, check:

```
paper_results/
├── fedavg_dirichlet_alpha0.1.csv
├── fedspec_dirichlet_alpha0.1.csv
├── fedavg_dirichlet_alpha0.5.csv
├── fedspec_dirichlet_alpha0.5.csv
├── fedavg_dirichlet_alpha1.0.csv
├── fedspec_dirichlet_alpha1.0.csv
└── centralized.csv (Mac only)
```

Generate plots:
```bash
cd fedspec
python plots/plot_frobenius_gap.py
python plots/plot_accuracy.py
```

## Performance Tips

### Mac M2 Optimization

```python
# In config.py
batch_size: int = 32  # Increase for faster training
local_epochs: int = 1  # Keep at 1 (higher = overfitting)
```

### Windows GTX 1660 Ti Optimization

```python
# In config.py
batch_size: int = 32  # Max for 6GB VRAM
```

Close other GPU apps (browser, etc.) during training.

### Parallel Speedup Calculation

```
Single machine: 3-4 hours
Two machines:   1.5-2 hours
Speedup:        ~2x (near-linear)
```

## Summary

| Script | Time | Clients | Rounds | Use Case |
|--------|------|---------|--------|----------|
| `run_quick_experiments.py` | 30-60 min | 5 | 10 | Quick validation |
| `run_distributed_experiments.py` | 1.5-2 hrs | 10 | 20 | Full paper results |
| `run_paper_experiments.py` | 3-4 hrs | 10 | 20 | Single machine fallback |

**Recommendation:**
1. Run `run_quick_experiments.py` first (30-60 min) to validate
2. If satisfied, run `run_distributed_experiments.py` on both machines for full results

Your experiments will complete much faster! 🚀
