@echo off
rem NICS Vision Geo: show and send the report of the latest finished training run (no training).
cd /d "%~dp0\..\.."
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
git pull --ff-only
venv\Scripts\python.exe scripts\train_geo.py --resend
pause
