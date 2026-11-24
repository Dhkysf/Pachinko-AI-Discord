@echo off
title Miyoria-Py Live Core (Bot + Panel :7878)
cd /d "%~dp0"
if not exist .venv (
    echo [!] Jalankan setup.bat terlebih dahulu!
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
REM Aktifkan akselerasi Vulkan pada Ollama bila ROCm driver AMD belum aktif
set OLLAMA_VULKAN=1
echo Menyalakan Miyoria-Py... Buka http://127.0.0.1:7878 di browser untuk Admin Panel!
python main.py
pause
