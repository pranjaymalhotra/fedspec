@echo off
REM Windows Setup Script for FedSpec
REM Run this script to set up everything automatically

echo ========================================
echo FedSpec Windows Setup (GTX 1660 Ti)
echo ========================================
echo.

REM Check Python installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10 from python.org
    pause
    exit /b 1
)
python --version
echo.

REM Check CUDA installation
echo [2/6] Checking CUDA installation...
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: CUDA toolkit not found. PyTorch will use CPU.
    echo For GPU acceleration, install CUDA 11.8 from developer.nvidia.com/cuda-downloads
) else (
    nvcc --version | findstr "release"
)
echo.

REM Create virtual environment
echo [3/6] Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists.
) else (
    python -m venv .venv
    echo Virtual environment created.
)
echo.

REM Activate virtual environment
echo [4/6] Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [5/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded.
echo.

REM Install dependencies
echo [6/6] Installing dependencies...
echo This may take 5-10 minutes...
echo.

echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo PyTorch installed.

echo Installing other dependencies...
pip install transformers peft datasets numpy scipy matplotlib pytest pyyaml --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo All dependencies installed.
echo.

REM Verify CUDA
echo ========================================
echo Verifying CUDA Installation
echo ========================================
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
echo.

REM Check if CUDA is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: CUDA not available. Training will use CPU (slow).
    echo To enable GPU:
    echo 1. Install NVIDIA drivers from nvidia.com
    echo 2. Install CUDA Toolkit 11.8
    echo 3. Re-run this setup script
    echo.
) else (
    echo SUCCESS: CUDA is working! GTX 1660 Ti detected.
    echo.
)

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Keep this terminal open (virtual environment is activated)
echo 2. Run quick test: cd fedspec ^&^& python run_sanity_test.py
echo 3. Run full experiments: cd fedspec ^&^& python run_paper_experiments.py
echo.
echo To activate environment later:
echo   .venv\Scripts\activate
echo.
pause
