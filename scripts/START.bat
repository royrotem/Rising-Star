@echo off
title UAIE - Quick Start
color 0F

echo.
echo  ========================================
echo       UAIE - Universal Autonomous
echo          Insight Engine
echo  ========================================
echo.
echo  Select an option:
echo.
echo  [1] Run with Docker (Recommended)
echo  [2] Run Locally (Python + Node)
echo  [3] Update from Git
echo  [4] Install Dependencies
echo  [5] Exit
echo.
echo  ========================================
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    call "%~dp0run-docker.bat"
) else if "%choice%"=="2" (
    call "%~dp0run.bat"
) else if "%choice%"=="3" (
    call "%~dp0update.bat"
) else if "%choice%"=="4" (
    call "%~dp0install.bat"
) else if "%choice%"=="5" (
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    call "%~dp0START.bat"
)
