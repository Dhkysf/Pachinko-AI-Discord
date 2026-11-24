@echo off
title Miyoria-Py Setup (Windows + AMD RX 9070 16GB)
cd /d "%~dp0"

echo ==============================================================
echo   Miyoria-Py Setup - Autonomous Voice Character ^& VTuber Core
echo ==============================================================
echo.

py -V:3.12 --version >nul 2>nul || (echo [!] Python 3.12 belum terinstall. Jalankan: winget install Python.Python.3.12 & pause & exit /b 1)
where git >nul 2>nul || (echo [!] Git belum terinstall. Jalankan: winget install Git.Git & pause & exit /b 1)
where ffmpeg >nul 2>nul || (echo [!] FFmpeg belum terinstall. Jalankan: winget install Gyan.FFmpeg & pause & exit /b 1)

where ollama >nul 2>nul
if errorlevel 1 (
    echo [INFO] Ollama belum terdeteksi di PATH.
    echo        Untuk LLM 100%% lokal di AMD RX 9070 16GB, install dengan:
    echo        winget install Ollama.Ollama
    echo        lalu jalankan: ollama pull qwen3:8b
    echo.
)

if not exist .venv (
    echo [1/3] Membuat virtual environment Python...
    py -V:3.12 -m venv .venv || (pause & exit /b 1)
)

echo [2/3] Menginstall paket Python (Core + DAVE Voice + Local STT/TTS + Web Panel)...
.venv\Scripts\python.exe -m pip install --upgrade pip -q || (echo [!] Gagal mengupgrade pip & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install -r requirements.txt || (echo [!] Gagal menginstall paket & pause & exit /b 1)

if not exist .env (
    copy .env.example .env >nul
    echo [INFO] File .env baru dibuat. Silakan isi DISCORD_TOKEN dan GUILD_ID di Notepad.
    notepad .env
)

echo [3/3] Mengunduh model suara offline lokal ^& mengecek kesiapan sistem...
.venv\Scripts\python.exe setup_models.py

echo.
echo Setup selesai! Klik dua kali run.bat untuk menyalakan Bot + Web Panel.
pause
