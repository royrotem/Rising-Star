@echo off
title UAIE - Running Application
color 0E

echo ========================================
echo    UAIE - Starting Application
echo ========================================
echo.

cd /d "%~dp0.."

echo Starting Backend and Frontend...
echo.
echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C in each window to stop
echo.

:: Start Backend in new window
echo [1/2] Starting Backend...
start "UAIE Backend" cmd /k "cd /d "%~dp0..\backend" && call venv\Scripts\activate.bat && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait a moment for backend to start
timeout /t 3 /nobreak >nul

:: Start Frontend in new window
echo [2/2] Starting Frontend...
start "UAIE Frontend" cmd /k "cd /d "%~dp0..\frontend" && npm run dev"

echo.
echo ========================================
echo    Application Started!
echo ========================================
echo.
echo Two new windows have opened:
echo   - UAIE Backend (Python/FastAPI)
echo   - UAIE Frontend (React/Vite)
echo.
echo Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul

:: Open browser
start http://localhost:5173

echo.
echo To stop the application, close both windows
echo or press Ctrl+C in each window.
echo.
pause
