@echo off
title UAIE - Update from Git
color 0A

echo ========================================
echo    UAIE - Pulling Latest Updates
echo ========================================
echo.

cd /d "%~dp0.."

echo [1/3] Fetching from remote...
git fetch origin
if errorlevel 1 (
    echo ERROR: Failed to fetch from remote
    pause
    exit /b 1
)

echo.
echo [2/3] Switching to development branch...
git checkout claude/refine-saas-description-3Bzwc
if errorlevel 1 (
    echo ERROR: Failed to checkout branch
    pause
    exit /b 1
)

echo.
echo [3/3] Pulling latest changes...
git pull origin claude/refine-saas-description-3Bzwc
if errorlevel 1 (
    echo ERROR: Failed to pull changes
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Update Complete!
echo ========================================
echo.
pause
