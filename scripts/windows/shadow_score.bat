@echo off
rem NICS Trading shadow scoring of new scanner signals (nothing is sent anywhere).
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python scripts\score_live.py >> logs\shadow_score.txt 2>&1
