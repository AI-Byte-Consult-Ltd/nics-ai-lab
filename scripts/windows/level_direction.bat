@echo off
rem Level method and direction rules on public hourly history since 2008 (research, nothing is sent to trading).
rem Takes a few minutes. The summary goes to the private dashboard and to experiments\research.
cd /d "%~dp0\..\.."
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
git pull --ff-only
venv\Scripts\python.exe scripts\level_direction_research.py %*
pause
