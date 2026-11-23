@echo off
title Waste Classifier AI
cd /d "%~dp0"

set "PY=py -3.11"
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=venv\Scripts\python.exe"
)

echo =======================================
echo   Dang khoi dong he thong phan loai rac
echo =======================================
echo.

%PY% app_windows.py

pause
