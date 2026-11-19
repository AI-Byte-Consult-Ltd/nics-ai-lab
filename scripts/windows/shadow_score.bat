@echo off
rem NICS Trading shadow scoring of new scanner signals (nothing is sent anywhere).
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
venv\Scripts\python.exe scripts\score_live.py >> logs\shadow_score.txt 2>&1
