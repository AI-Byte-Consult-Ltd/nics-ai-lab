@echo off
rem NICS Trading nightly self-learning cycle (download -> learn -> compare -> keep the better model).
rem Needs NICS_DATA_URL and NICS_DATA_KEY set once with setx (see docs/laptop-guide-ru.md).
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
git pull --ff-only >> logs\self_learn.txt 2>&1
python scripts\self_learn.py %* >> logs\self_learn.txt 2>&1
