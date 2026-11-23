@echo off
rem NICS Vision Geo: the model trained here against the model in the phone app, same test photos.
cd /d "%~dp0\..\.."
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
git pull --ff-only
venv\Scripts\python.exe scripts\geo_compare.py %*
pause
