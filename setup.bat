@echo off
title Pachinko Setup (Windows + AMD RX 9070 16GB)
cd /d "%~dp0"

echo ==============================================================
echo   Pachinko Setup - Autonomous Discord Voice ^& Chat Core
echo ==============================================================
echo.

py -V:3.12 --version >nul 2>nul || (echo [!] Python 3.12 is not installed. Run: winget install Python.Python.3.12 & pause & exit /b 1)
where git >nul 2>nul || (echo [!] Git is not installed. Run: winget install Git.Git & pause & exit /b 1)
where ffmpeg >nul 2>nul || (echo [!] FFmpeg is not installed. Run: winget install Gyan.FFmpeg & pause & exit /b 1)

where ollama >nul 2>nul
if errorlevel 1 (
    echo [INFO] Ollama was not detected on PATH.
    echo        For 100%% local LLM on AMD RX 9070 16GB, install with:
    echo        winget install Ollama.Ollama
    echo        then run: ollama pull qwen3:8b
    echo.
)

if not exist .venv (
    echo [1/3] Creating Python virtual environment...
    py -V:3.12 -m venv .venv || (pause & exit /b 1)
)

echo [2/3] Installing Python packages (Core + DAVE Voice + Local STT/TTS + Web Panel)...
.venv\Scripts\python.exe -m pip install --upgrade pip -q || (echo [!] Failed to upgrade pip & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install -r requirements.txt || (echo [!] Failed to install packages & pause & exit /b 1)

if not exist .env (
    copy .env.example .env >nul
    echo [INFO] Created new .env file. Please fill in DISCORD_TOKEN and GUILD_ID in Notepad.
    notepad .env
)

echo [3/3] Downloading offline local voice model ^& checking system readiness...
.venv\Scripts\python.exe setup_models.py

echo.
echo Setup complete! Double-click run.bat to start the Bot + Web Panel.
pause
