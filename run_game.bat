@echo off
title Green Warrior - Game phan loai rac
cd /d "%~dp0"

set "PY=py -3.11"
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=venv\Scripts\python.exe"
)

echo =======================================
echo   CHIEN BINH XANH - AI PHAN LOAI RAC
echo =======================================
echo.
echo Dang khoi dong server va mo trinh duyet...
echo (Lan dau tai model co the mat khoang 10 giay)
echo.

%PY% launch_game.py

pause
