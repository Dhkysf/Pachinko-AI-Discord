@echo off
title Pachinko Live Core (Bot + Panel :7878)
cd /d "%~dp0"
if not exist .venv (
    echo [!] Run setup.bat first!
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
REM Enable Vulkan acceleration on Ollama if AMD ROCm driver is not active
set OLLAMA_VULKAN=1
echo Starting Pachinko... Open http://127.0.0.1:7878 in your browser for the Admin Panel!
echo (Tip: for daily use, double-click Pachinko.lnk or Pachinko.pyw for the system tray control app.)
.venv\Scripts\python.exe main.py
pause
