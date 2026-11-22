@echo off
rem NICS Language by hand: downloads the texts if needed, then trains until the given time.
rem   scripts\windows\language_train.bat                  until 08:00
rem   scripts\windows\language_train.bat --max-minutes 60
rem Normally nothing has to be started: the nightly NICS Trading task does this by itself.
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
git pull --ff-only
venv\Scripts\python.exe scripts\language_auto.py %*
pause
