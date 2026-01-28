@echo off
title UAIE - Stop Docker
color 0C

echo ========================================
echo    UAIE - Stopping Docker Services
echo ========================================
echo.

cd /d "%~dp0.."

echo Stopping all containers...
docker-compose down

echo.
echo ========================================
echo    All services stopped!
echo ========================================
echo.
pause
