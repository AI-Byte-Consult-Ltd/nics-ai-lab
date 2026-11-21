@echo off
rem Measures how fast this laptop trains and runs NICS Language models (about 2 minutes, no data needed).
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
git pull --ff-only
venv\Scripts\python.exe scripts\bench_text.py %*
pause
