@echo off
SETLOCAL EnableDelayedExpansion
echo ===================================================
echo  Setting up the Environment for Book Recommender
echo ===================================================
echo.

:: Check for python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system's PATH.
    echo Please install Python 3.12.7 and check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Create virtual environment
echo [1/3] Creating virtual environment (.venv)...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate virtual environment and install packages
echo [2/3] Activating virtual environment and installing libraries...
call .venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install required libraries.
    pause
    exit /b 1
)

:: Create quick run script
echo [3/3] Creating run_app.bat helper script...
(
echo @echo off
echo echo Starting Book Recommender System...
echo call .venv\Scripts\activate
echo streamlit run app.py
echo pause
) > run_app.bat

echo.
echo ===================================================
echo  Setup Completed Successfully!
echo ===================================================
echo  1. To start the app, double-click "run_app.bat" in this directory.
echo  2. Or run: streamlit run app.py with your environment active.
echo ===================================================
echo.
pause
