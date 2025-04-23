@echo off
echo ========== Setting up CUDA-compatible environment ==========

:: Check if FFmpeg is installed
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo FFmpeg is not installed.
    echo Opening FFmpeg download page...
    start https://ffmpeg.org/download.html
    exit /b
)

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

echo [1/3] Installing CUDA-compatible PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo [2/3] Installing Whisper and Pydub...
pip install openai-whisper pydub

echo [3/3] Setup complete!

echo.
echo ================================
echo FFmpeg is required for Whisper to work
echo Please install it from the following URL:
echo https://ffmpeg.org/download.html
echo ================================

echo ===== Activating virtual environment and starting the app =====
call venv\Scripts\activate
python app.py
