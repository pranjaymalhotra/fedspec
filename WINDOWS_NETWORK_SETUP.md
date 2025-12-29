# Windows Setup for Distributed Training

This guide shows how to connect your Windows machine (with GTX 1660 Ti) to your Mac on the same network for distributed FedSpec experiments.

---

## 🌐 How to Connect Windows to Mac

### Step 1: Find Mac's IP Address

**On Mac:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Example output: `inet 192.168.1.100`

Note this IP address - you'll need it for Windows.

---

### Step 2: Enable File Sharing on Mac

1. Open **System Settings** → **General** → **Sharing**
2. Turn on **File Sharing**
3. Click the **(i)** icon next to "File Sharing"
4. Add the Fedspec folder:
   - Click **"+"** button
   - Navigate to: `/Users/pranjaymalhotra/Documents/Fedspec`
   - Set permissions: **"Read & Write"** for your user
5. Note your **Mac username** (you'll need this on Windows)

**Keep Mac awake during experiments:**
- System Settings → Lock Screen
- Set "Turn display off after" to **Never** (temporary)
- Enable "Prevent automatic sleeping when display is off"

---

### Step 3: Connect from Windows

**On Windows:**

1. **Map Network Drive:**
   - Open **File Explorer**
   - Right-click **"This PC"** → **"Map network drive"**
   - Choose drive letter: `Z:`
   - Folder: `\\192.168.1.100\Fedspec` (use your Mac's IP)
   - ✅ Check **"Reconnect at sign-in"**
   - Click **"Finish"**
   - Enter **Mac username and password** when prompted

2. **Verify connection:**
   ```cmd
   Z:
   dir
   ```
   You should see the Fedspec folder contents.

---

## 🔧 Windows Software Setup

### Install Python and CUDA

1. **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/)
2. **CUDA Toolkit 11.8**: Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive)
3. **NVIDIA Driver**: Latest from [NVIDIA](https://www.nvidia.com/Download/index.aspx)

Verify installation:
```cmd
python --version
nvcc --version
nvidia-smi
```

### Install Dependencies

Navigate to mapped drive:
```cmd
Z:
cd Fedspec
```

Create virtual environment:
```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch with CUDA:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install other dependencies:
```cmd
pip install -r requirements.txt
```

Verify CUDA works:
```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Should print: `CUDA available: True`

---

## 🚀 Running Distributed Experiments

### On Mac (Terminal 1)

```bash
cd /Users/pranjaymalhotra/Documents/Fedspec
source .venv/bin/activate
cd fedspec
python run_distributed_experiments.py
```

Mac will:
- Create work queue (6 experiments)
- Start running experiments
- Monitor Windows progress
- Take over if Windows fails

### On Mac (Terminal 2) - Monitor Progress

```bash
cd /Users/pranjaymalhotra/Documents/Fedspec/fedspec
source .venv/bin/activate
python view_progress.py --watch 30
```

This shows:
- ✅ Completed experiments
- 🔄 Currently running (round progress, accuracy)
- 💻 Mac vs Windows contribution
- 💾 Checkpoint verification
- **No performance impact** - just reads JSON files!

### On Windows

```cmd
Z:
cd Fedspec
.venv\Scripts\activate
cd fedspec
python run_distributed_experiments.py
```

Windows will:
- Detect existing work queue
- Claim available experiments
- Save checkpoints to shared folder
- Work in parallel with Mac

---

## 📊 Expected Timeline

| Stage | Time | Mac | Windows |
|-------|------|-----|---------|
| Setup | 10-15 min | Enable sharing | Map drive, install |
| Experiment 1 | 15-20 min | fedspec α=0.1 | fedavg α=0.1 |
| Experiment 2 | 15-20 min | fedspec α=0.5 | fedavg α=0.5 |
| Experiment 3 | 15-20 min | fedspec α=1.0 | fedavg α=1.0 |
| **Total** | **1.5-2 hours** | 3 experiments | 3 experiments |

**Single machine (Mac only):** 3-4 hours  
**Distributed (Mac + Windows):** 1.5-2 hours  
**Speedup:** ~2x faster ⚡

---

## 🔍 Progress Viewer Features

Run `python view_progress.py` on Mac (in separate terminal):

### Real-time Information:
- 🎯 **Work Queue**: Total/completed/in-progress/pending/failed
- 🔄 **Currently Running**: Shows which experiments are running, round progress, latest accuracy
- 💻 **Machine Breakdown**: Mac vs Windows contributions
- ✅ **Completed Results**: Final accuracy for finished experiments
- 💾 **Checkpoint Status**: Verifies all checkpoints saved properly

### Example Output:

```
======================================================================
 📊 FedSpec Distributed Training Progress
======================================================================

🎯 Work Queue Status:
   Total Experiments: 6
   ✅ Completed:      2
   🔄 In Progress:    2
   ⏳ Pending:        2
   Progress: [███████████░░░░░░░░░░░░░░░] 33.3%

🔄 Currently Running:
   • exp_fedspec_0.1: fedspec (α=0.1) on mac
     Round 8/20 [████████░░░░░░░░░░░░] 40.0%
     Latest accuracy: 84.23%
     Last updated: 2 min ago
   
   • exp_fedavg_0.1: fedavg (α=0.1) on windows
     Round 12/20 [████████████░░░░░░░░] 60.0%
     Latest accuracy: 82.45%
     Last updated: just now

💻 Machine Progress:
   🍎 Mac: 1 experiments completed
   🪟 Windows: 1 experiments completed

✅ Completed Experiments:
   • fedavg (α=0.5): 83.12%
   • fedspec (α=0.5): 85.34%

💾 Checkpoint Status:
   Total checkpoints: 4
   Disk usage: 234.5 MB
   ✅ All checkpoints verified and saved

======================================================================
```

### Auto-Refresh Mode:
```bash
python view_progress.py --watch 30  # Refresh every 30 seconds
```

---

## 🛠️ Troubleshooting

### Cannot connect to Mac from Windows

**Check network:**
```cmd
ping 192.168.1.100  # Use your Mac's IP
```

If ping fails:
- Ensure both on same WiFi network
- Check Mac firewall: System Settings → Network → Firewall
- Temporarily disable to test
- Re-enable and allow "File Sharing" through firewall

### Work queue not found

- Mac must run first to create `distributed_work/work_queue.json`
- Check Z: drive is accessible: `dir Z:\Fedspec\fedspec\distributed_work`
- Test file sharing: Create test file on Mac, check if visible on Windows

### CUDA not available on Windows

```cmd
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check
python -c "import torch; print(torch.cuda.is_available())"
```

### Windows claims work but doesn't complete

- Mac monitors and will retry after 30 minutes
- Check Windows terminal for errors
- Verify checkpoints saving: `dir Z:\Fedspec\fedspec\checkpoints_distributed`

### Checkpoints not syncing

- SMB/Network share: Check connection (`ping` Mac)
- Verify write permissions on shared folder
- Test: Create file on Windows, check if Mac sees it

### Different results on Mac vs Windows

- Both use same code (network share)
- GPU differences (M2 MPS vs 1660 Ti CUDA) may cause slight variance
- Use same random seed (already configured in code)

---

## ⚡ Performance Tips

1. **Increase batch size on Windows** (1660 Ti has 6GB VRAM):
   - Edit `fedspec/config.py`:
   ```python
   batch_size: int = 32  # Up from 16
   ```

2. **Use Ethernet if possible** (faster than WiFi)

3. **Keep Mac plugged in** (prevent sleep)

4. **Run progress viewer only on Mac** (no need on Windows)

5. **Close other apps** on both machines during experiments

---

## ✅ Quick Verification Checklist

Before starting experiments:

- [ ] Mac IP address noted: `ifconfig | grep inet`
- [ ] Mac File Sharing enabled
- [ ] Windows can ping Mac: `ping <mac-ip>`
- [ ] Z: drive mapped and accessible on Windows
- [ ] Python 3.10+ on both machines
- [ ] CUDA available on Windows: `torch.cuda.is_available() == True`
- [ ] Dependencies installed on both: `pip list | grep torch`
- [ ] Mac set to not sleep during experiments
- [ ] Tested: Create file on Mac → visible on Windows

Once all checked, you're ready! 🚀

---

## 📝 Quick Reference Commands

### Mac
```bash
# Start experiments (Terminal 1)
python run_distributed_experiments.py

# Monitor progress (Terminal 2)
python view_progress.py --watch 30

# Verify checkpoints
python verify_checkpoints.py
```

### Windows
```cmd
# Navigate to shared folder
Z:
cd Fedspec\fedspec

# Start experiments
python run_distributed_experiments.py
```

---

## 🆘 Getting Help

If you encounter issues:

1. Check progress viewer: `python view_progress.py`
2. Verify checkpoints: `python verify_checkpoints.py`
3. Check work queue: `cat distributed_work/work_queue.json` (Mac) or `type distributed_work\work_queue.json` (Windows)
4. View logs in experiment output directories

Both machines save detailed logs, and Mac acts as backup if Windows fails. The system is designed to be fault-tolerant! ✅
