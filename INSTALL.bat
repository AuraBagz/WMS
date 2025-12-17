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

echo [1/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
) else (
    echo Virtual environment already exists, skipping...
)

echo.
echo [2/6] Upgrading pip...
.\venv\Scripts\python -m pip install --upgrade pip

echo.
echo [3/6] Installing PyTorch with CUDA 12.4 support...
echo      (This may take a few minutes for the ~2.5GB download)
.\venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/6] Installing Ultralytics YOLO...
.\venv\Scripts\pip install ultralytics

echo.
echo [5/6] Installing other dependencies...
.\venv\Scripts\pip install -r requirements.txt

echo.
echo [6/6] Checking FFmpeg installation...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo.
    echo ================================================
    echo   FFmpeg Not Found - Installing via winget
    echo ================================================
    echo FFmpeg is required for audio preservation.
    echo.
    
    REM Try winget first (Windows 10/11)
    winget --version >nul 2>&1
    if not errorlevel 1 (
        echo Installing FFmpeg via winget...
        winget install FFmpeg -e --silent --accept-package-agreements --accept-source-agreements
        
        REM Check if it worked
        where ffmpeg >nul 2>&1
        if errorlevel 1 (
            echo.
            echo WARNING: FFmpeg installation may require a restart or PATH update.
            echo If audio is not preserved, please install FFmpeg manually:
            echo   1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases
            echo   2. Extract to C:\ffmpeg
            echo   3. Add C:\ffmpeg\bin to your PATH environment variable
        ) else (
            echo FFmpeg installed successfully!
        )
    ) else (
        echo.
        echo WARNING: winget not available. Please install FFmpeg manually:
        echo   1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases
        echo   2. Extract to C:\ffmpeg
        echo   3. Add C:\ffmpeg\bin to your PATH environment variable
        echo.
        echo Without FFmpeg, output videos will not have audio.
    )
) else (
    echo FFmpeg is already installed!
    ffmpeg -version | findstr "ffmpeg version"
)

echo.
echo ================================================
echo   Installation Complete!
echo ================================================
echo.
echo To start WaterSlayer, run: START.bat
echo.
echo Verifying installation...
.\venv\Scripts\python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo.

REM Check FFmpeg again
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg not in PATH - audio will not be preserved
) else (
    echo FFmpeg: OK
)
echo.
pause
