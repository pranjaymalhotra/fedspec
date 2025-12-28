#!/bin/bash
# macOS Setup Script for FedSpec (Apple M2)
# Run this with: bash setup_mac.sh

set -e  # Exit on error

echo "========================================"
echo "FedSpec macOS Setup (Apple M2)"
echo "========================================"
echo ""

# Check Python installation
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install from python.org or use Homebrew:"
    echo "  brew install python@3.10"
    exit 1
fi
python3 --version
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    python3 -m venv .venv
    echo "Virtual environment created."
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source .venv/bin/activate
echo ""

# Upgrade pip
echo "[4/5] Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "pip upgraded."
echo ""

# Install dependencies
echo "[5/5] Installing dependencies..."
echo "This may take 5-10 minutes..."
echo ""

echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi
echo "PyTorch installed."

echo "Installing other dependencies..."
pip install transformers peft datasets numpy scipy matplotlib pytest pyyaml --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "All dependencies installed."
echo ""

# Verify MPS
echo "========================================"
echo "Verifying MPS (Metal Performance Shaders)"
echo "========================================"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}'); print(f'PyTorch version: {torch.__version__}')"
echo ""

# Check if MPS is available
python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" &> /dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: MPS not available. Training will use CPU."
    echo "Ensure you have macOS 12.3+ and Apple Silicon (M1/M2/M3)"
    echo ""
else
    echo "SUCCESS: MPS is working! Apple M2 GPU will be used."
    echo ""
fi

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Keep this terminal open (virtual environment is activated)"
echo "2. Run quick test: cd fedspec && python run_sanity_test.py"
echo "3. Run full experiments: cd fedspec && python run_paper_experiments.py"
echo ""
echo "To activate environment later:"
echo "  source .venv/bin/activate"
echo ""
