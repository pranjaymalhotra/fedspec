@echo off
REM Windows Setup Script for Distributed Training
REM Run this on Windows after mapping the network drive

echo ==========================================
echo Windows Setup for Distributed Training
echo ==========================================
echo.

REM Step 1: Verify we're on the mapped drive
echo Step 1: Verifying mapped drive...
if exist "fedspec\" (
    echo [OK] Fedspec directory found
) else (
    echo [ERROR] Fedspec directory not found
    echo Please make sure you're in the mapped Z: drive
    echo.
    echo Instructions:
    echo 1. Open File Explorer
    echo 2. Right-click "This PC" -^> "Map network drive"
    echo 3. Drive letter: Z:
    echo 4. Folder: \\YOUR-MAC-IP\Fedspec
    echo 5. Check "Reconnect at sign-in"
    echo 6. Enter Mac username and password
    echo.
    pause
    exit /b 1
)

REM Step 2: Check Python
echo.
echo Step 2: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

python --version
echo [OK] Python found

REM Step 3: Check CUDA
echo.
echo Step 3: Checking CUDA/GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] nvidia-smi not found
    echo GPU may not be available
) else (
    echo [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name --format=csv,noheader
)

REM Step 4: Check/Create virtual environment
echo.
echo Step 4: Setting up virtual environment...
if exist ".venv\" (
    echo [OK] Virtual environment exists
) else (
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Step 5: Activate and install dependencies
echo.
echo Step 5: Installing dependencies...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

echo Installing other dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

REM Step 6: Verify CUDA works
echo.
echo Step 6: Verifying CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

if %errorlevel% neq 0 (
    echo [WARNING] Could not verify CUDA
)

echo.
echo ==========================================
echo Windows Setup Complete!
echo ==========================================
echo.
echo [OK] All checks passed
echo.
echo Ready to run distributed experiments!
echo.
echo ==========================================
echo Next Steps:
echo ==========================================
echo.
echo 1. On Mac, start experiments:
echo    Terminal 1: cd fedspec ^&^& python run_distributed_experiments.py
echo    Terminal 2: cd fedspec ^&^& python view_progress.py --watch 30
echo.
echo 2. On Windows (this machine), run:
echo    cd fedspec
echo    python run_distributed_experiments.py
echo.
echo 3. Both machines will work in parallel
echo    Mac monitors progress and shows real-time updates
echo.
echo ==========================================
echo.
pause
