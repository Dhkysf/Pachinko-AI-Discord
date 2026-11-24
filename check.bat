@echo off
title Pachinko Offline Check
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo [!] .venv\Scripts\python.exe not found. Run setup.bat first!
    pause
    exit /b 1
)

set FAILED=0
echo Running compileall...
.venv\Scripts\python.exe -m compileall -q brain core discord_bot heart memory play sing speech voice web app tests config.py main.py setup_models.py Pachinko.pyw
if errorlevel 1 set FAILED=1

echo Running offline unit tests...
.venv\Scripts\python.exe -m unittest discover tests -v
if errorlevel 1 set FAILED=1

echo.
if %FAILED%==0 (
    echo PASS ^(Tip: you can also run checks from Pachinko Control via Pachinko.lnk^)
) else (
    echo FAIL
)
pause
exit /b %FAILED%
