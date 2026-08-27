@echo off
echo ===================================================
echo Forensic DGP Setup (Thesis Deployment)
echo ===================================================
echo.

echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ and check the box "Add Python to PATH".
    pause
    exit /b
)

echo Creating virtual environment (venv)...
python -m venv venv

echo Activating venv and installing requirements...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===================================================
echo Installation Complete! 
echo You can now double-click 'start.bat' to run the Web UI.
echo ===================================================
pause
