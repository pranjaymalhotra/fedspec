# Quick Start Guide: Running on Both Machines

Follow these steps to run distributed training on your Mac M2 and Windows GTX 1660 Ti simultaneously.

---

## 🍎 Part 1: Mac Setup (5 minutes)

### Step 1: Run the Mac setup script

```bash
cd /Users/pranjaymalhotra/Documents/Fedspec
bash setup_mac_sharing.sh
```

This will show you:
- ✅ Your Mac's IP address (e.g., `192.168.1.100`)
- ✅ Instructions for enabling File Sharing
- ✅ Network path for Windows (e.g., `\\192.168.1.100\Fedspec`)

### Step 2: Enable File Sharing

1. Open **System Settings**
2. Go to **General** → **Sharing**
3. Turn **ON** "File Sharing"
4. Click the **(i)** icon next to "File Sharing"
5. Click **"+"** to add a folder
6. Navigate to `/Users/pranjaymalhotra/Documents/Fedspec`
7. Set permissions to **"Read & Write"** for your user
8. Click **Done**

### Step 3: Prevent Mac from sleeping

1. **System Settings** → **Lock Screen**
2. Set "Turn display off after" to **Never** (temporary, during experiments)
3. Enable "Prevent automatic sleeping when display is off"

✅ Mac is ready!

---

## 🪟 Part 2: Windows Setup (10 minutes)

### Step 1: Get Mac's IP address

From Mac terminal output (Step 1 above), note the IP address.

Example: `192.168.1.100`

### Step 2: Map Network Drive on Windows

1. Open **File Explorer**
2. Right-click **"This PC"** → **"Map network drive"**
3. Drive letter: **Z:**
4. Folder: `\\192.168.1.100\Fedspec` (use your Mac's IP)
5. ✅ Check **"Reconnect at sign-in"**
6. Click **"Finish"**
7. When prompted, enter:
   - Username: Your **Mac username** (shown in setup script)
   - Password: Your **Mac password**

### Step 3: Verify connection

Open **Command Prompt** and test:

```cmd
ping 192.168.1.100
```

Should show replies. If not, check:
- Both on same WiFi network
- Mac firewall allows file sharing
- Mac File Sharing is enabled

```cmd
Z:
dir
```

Should show Fedspec folder contents.

### Step 4: Run Windows setup script

```cmd
Z:
cd Fedspec
setup_windows_distributed.bat
```

This will:
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA
- ✅ Install all dependencies
- ✅ Verify GPU is available

Wait for installation (5-10 minutes).

✅ Windows is ready!

---

## 🚀 Part 3: Running Experiments (1.5-2 hours)

Now both machines are set up. Time to run!

### On Mac - Terminal 1 (Experiments)

```bash
cd /Users/pranjaymalhotra/Documents/Fedspec
source .venv/bin/activate
cd fedspec
python run_distributed_experiments.py
```

You'll see:
```
======================================================================
FedSpec: Distributed Experiments (Mac + Windows)
======================================================================

Device: Apple M2 (MPS)
Creating work queue with 6 experiments...
✓ Work queue initialized

Claiming work...
✓ Claimed: exp_fedspec_0.1
Starting experiment...
```

### On Mac - Terminal 2 (Monitor Progress)

Open a **new terminal** window:

```bash
cd /Users/pranjaymalhotra/Documents/Fedspec/fedspec
source ../.venv/bin/activate
python view_progress.py --watch 30
```

You'll see real-time updates every 30 seconds:
```
🎯 Work Queue Status:
   Total Experiments: 6
   ✅ Completed:      0
   🔄 In Progress:    1
   ⏳ Pending:        5

🔄 Currently Running:
   • exp_fedspec_0.1: fedspec (α=0.1) on mac
     Round 3/20 [███████░░░░░░░░░░░░░] 15.0%
     Latest accuracy: 82.45%
     Last updated: 30 sec ago
```

### On Windows - Command Prompt

```cmd
Z:
cd Fedspec
.venv\Scripts\activate
cd fedspec
python run_distributed_experiments.py
```

You'll see:
```
======================================================================
FedSpec: Distributed Experiments (Mac + Windows)
======================================================================

Device: NVIDIA GeForce GTX 1660 Ti (CUDA)
Detected existing work queue
Claiming work...
✓ Claimed: exp_fedavg_0.1
Starting experiment...
```

**Both machines now work in parallel!** 🎉

---

## 📊 What Happens Next

### Timeline (approximate)

| Time | Mac | Windows | Progress Monitor |
|------|-----|---------|------------------|
| 0:00 | Claims fedspec α=0.1 | Claims fedavg α=0.1 | Shows 2 in progress |
| 0:20 | Round 8/20 | Round 10/20 | Updates every 30s |
| 0:30 | Completes experiment 1 | Completes experiment 1 | Shows 2 completed |
| 0:31 | Claims fedspec α=0.5 | Claims fedavg α=0.5 | Shows 2 in progress |
| 1:00 | Completes experiment 2 | Completes experiment 2 | Shows 4 completed |
| 1:01 | Claims fedspec α=1.0 | Claims fedavg α=1.0 | Shows 2 in progress |
| 1:30 | Completes experiment 3 | Completes experiment 3 | Shows 6 completed |
| 1:30 | 🎉 All done! | 🎉 All done! | 🎉 100% complete |

**Total time: ~1.5-2 hours** (vs 3-4 hours on Mac alone)

### What you'll see in progress viewer:

```
======================================================================
 📊 FedSpec Distributed Training Progress
======================================================================

🎯 Work Queue Status:
   Total Experiments: 6
   ✅ Completed:      4
   🔄 In Progress:    2
   ⏳ Pending:        0
   Progress: [████████████████████░░░░░░░░] 66.7%

🔄 Currently Running:
   • exp_fedspec_1.0: fedspec (α=1.0) on mac
     Round 15/20 [███████████████░░░░░] 75.0%
     Latest accuracy: 87.12%
     Last updated: 20 sec ago
   
   • exp_fedavg_1.0: fedavg (α=1.0) on windows
     Round 18/20 [██████████████████░░] 90.0%
     Latest accuracy: 85.34%
     Last updated: just now

💻 Machine Progress:
   🍎 Mac: 2 experiments completed
      Completed: exp_fedspec_0.1, exp_fedspec_0.5
   🪟 Windows: 2 experiments completed
      Completed: exp_fedavg_0.1, exp_fedavg_0.5

✅ Completed Experiments:
   • fedavg (α=0.1): 81.34%
   • fedavg (α=0.5): 83.12%
   • fedspec (α=0.1): 86.78%
   • fedspec (α=0.5): 88.45%

💾 Checkpoint Status:
   Total checkpoints: 6
   Disk usage: 456.3 MB
   ✅ All checkpoints verified and saved

======================================================================
```

---

## ✅ Verification Checklist

Before starting, make sure:

**Mac:**
- [ ] File Sharing enabled
- [ ] Fedspec folder shared with Read & Write permissions
- [ ] Mac won't sleep (display can turn off, but not sleep)
- [ ] Terminal 1 ready for `run_distributed_experiments.py`
- [ ] Terminal 2 ready for `view_progress.py --watch 30`

**Windows:**
- [ ] Can ping Mac's IP address
- [ ] Z: drive mapped to `\\<mac-ip>\Fedspec`
- [ ] Can see Fedspec folder contents: `dir Z:\Fedspec`
- [ ] Python installed: `python --version`
- [ ] CUDA available: `nvidia-smi` works
- [ ] Dependencies installed: `setup_windows_distributed.bat` completed
- [ ] Command Prompt ready for `run_distributed_experiments.py`

**Both:**
- [ ] Same WiFi network
- [ ] Mac's firewall allows file sharing
- [ ] Both can read/write to shared folder

---

## 🔧 Troubleshooting

### Windows can't connect to Mac

**Solution 1: Check network**
```cmd
ping 192.168.1.100  # Use your Mac's IP
```

**Solution 2: Disable Mac firewall temporarily**
- System Settings → Network → Firewall → OFF
- Test connection
- Turn back ON, allow File Sharing

**Solution 3: Verify File Sharing**
- Mac: System Settings → General → Sharing
- File Sharing should show "On"
- Fedspec folder should be in shared folders list

### CUDA not available on Windows

```cmd
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test
python -c "import torch; print(torch.cuda.is_available())"
```

### Windows completes experiments but Mac doesn't see updates

**Solution:** Check file sync
```cmd
# Windows: Create test file
echo test > Z:\Fedspec\test.txt

# Mac: Check if file exists
ls /Users/pranjaymalhotra/Documents/Fedspec/test.txt
```

If file doesn't appear, network share has issues. Try:
- Remapping Z: drive
- Restarting File Sharing on Mac
- Checking network connection

### Progress viewer shows old data

**Solution:** Checkpoints update every round (every 2-3 minutes)
- Wait for next round to complete
- Progress viewer reads JSON files updated after each round
- If stuck, check experiment terminals for errors

---

## 🎉 Success!

When all experiments complete:

**Mac Terminal 1 shows:**
```
🎉 ALL EXPERIMENTS COMPLETE!
Results saved to: paper_results/
```

**Progress viewer shows:**
```
✅ Completed Experiments: 6/6
💻 Mac: 3 experiments
💻 Windows: 3 experiments
```

**Check results:**
```bash
cd /Users/pranjaymalhotra/Documents/Fedspec/fedspec
ls paper_results/
```

You'll see:
- `comparison_plot.pdf` - Visual comparison of all methods
- `results_table.tex` - LaTeX table for paper
- Individual result files for each experiment

---

## 📝 Quick Command Reference

### Mac - Terminal 1 (Run)
```bash
cd /Users/pranjaymalhotra/Documents/Fedspec/fedspec
source ../.venv/bin/activate
python run_distributed_experiments.py
```

### Mac - Terminal 2 (Monitor)
```bash
cd /Users/pranjaymalhotra/Documents/Fedspec/fedspec
source ../.venv/bin/activate
python view_progress.py --watch 30
```

### Windows
```cmd
Z:
cd Fedspec\fedspec
.venv\Scripts\activate
python run_distributed_experiments.py
```

---

**Ready? Let's start with Mac setup!** 🚀

Run on Mac:
```bash
cd /Users/pranjaymalhotra/Documents/Fedspec
bash setup_mac_sharing.sh
```
