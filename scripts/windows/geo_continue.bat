@echo off
rem NICS Vision Geo: continue training from the latest run (no download, no rebuild).
rem   scripts\windows\geo_continue.bat                 30 more epochs, full speed
rem   scripts\windows\geo_continue.bat --gentle        low priority, half of the threads (daytime)
rem   scripts\windows\geo_continue.bat --max-minutes 120
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
git pull --ff-only
echo Training continues. Progress is written to logs\geo_continue.txt, this window shows the result at the end.
venv\Scripts\python.exe scripts\train_geo.py --continue %* > logs\geo_continue.txt 2>&1
type logs\geo_continue.txt | more
pause
