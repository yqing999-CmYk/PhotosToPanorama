@echo off
:: Clear the broken PYTHONHOME/PYTHONPATH env vars for this session
set PYTHONHOME=
set PYTHONPATH=

cd /d "%~dp0backend"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Is Python 3.11 installed?
        pause
        exit /b 1
    )
)

echo Installing / verifying dependencies...
venv\Scripts\pip install -r requirements.txt --quiet

echo.
echo Starting FastAPI server on http://0.0.0.0:8000 ...
echo Press Ctrl+C to stop.
echo.
venv\Scripts\uvicorn main:app --reload --host 0.0.0.0 --port 8000

if errorlevel 1 (
    echo.
    echo ERROR: Server failed to start. Port 8000 may already be in use.
    echo Run this in PowerShell to free the port:
    echo   Get-Process | Where-Object {$_.MainWindowTitle -eq ""} ^| Stop-Process
    echo   netstat -ano ^| findstr :8000
    pause
)
