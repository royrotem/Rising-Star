@echo off
title UAIE - Docker Run
color 0D

echo ========================================
echo    UAIE - Starting with Docker
echo ========================================
echo.

cd /d "%~dp0.."

:: Check Docker
echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Docker is not installed!
    echo Please download Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop
    echo.
    pause
    exit /b 1
)
docker --version
echo Docker OK!
echo.

:: Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and wait for it to initialize.
    echo.
    pause
    exit /b 1
)
echo Docker is running!
echo.

echo Starting services with Docker Compose...
echo.
echo This may take a few minutes on first run...
echo.

:: Run docker-compose
docker-compose up --build

pause
