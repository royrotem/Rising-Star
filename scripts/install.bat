@echo off
title UAIE - Installation
color 0B

echo ========================================
echo    UAIE - Full Installation
echo ========================================
echo.

cd /d "%~dp0.."

:: Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed!
    echo Please download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)
python --version
echo Python OK!
echo.

:: Check Node.js
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Node.js is not installed!
    echo Please download Node.js from: https://nodejs.org/
    echo.
    pause
    exit /b 1
)
node --version
echo Node.js OK!
echo.

:: Check npm
echo [3/6] Checking npm...
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm is not installed!
    pause
    exit /b 1
)
npm --version
echo npm OK!
echo.

:: Setup Backend
echo [4/6] Setting up Backend...
cd backend

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

cd ..
echo Backend setup complete!
echo.

:: Setup Frontend
echo [5/6] Setting up Frontend...
cd frontend

echo Installing Node.js dependencies...
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)

cd ..
echo Frontend setup complete!
echo.

:: Create .env file
echo [6/6] Creating environment file...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo Created .env file from template
    )
)

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo You can now run the application using:
echo   - run.bat (local development)
echo   - run-docker.bat (Docker - recommended)
echo.
pause
