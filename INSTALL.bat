@echo off
echo ================================================
echo   WaterSlayer - Installation Script (GPU)
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
) else (
    echo Virtual environment already exists, skipping...
)

echo.
echo [2/5] Upgrading pip...
.\venv\Scripts\python -m pip install --upgrade pip

echo.
echo [3/5] Installing PyTorch with CUDA 12.4 support...
echo      (This may take a few minutes for the ~2.5GB download)
.\venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/5] Installing Ultralytics YOLO...
.\venv\Scripts\pip install ultralytics

echo.
echo [5/5] Installing other dependencies...
.\venv\Scripts\pip install -r requirements.txt

echo.
echo ================================================
echo   Installation Complete!
echo ================================================
echo.
echo To start WaterSlayer, run: START.bat
echo.
echo Verifying CUDA installation...
.\venv\Scripts\python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo.
pause
