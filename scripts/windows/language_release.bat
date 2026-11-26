@echo off
rem NICS Language: make a release folder (weights, tokenizer, config, checksums) for other products.
rem Training is not touched. The folder appears in releases\language\.
cd /d "%~dp0\..\.."
set PYTHONIOENCODING=utf-8
git pull --ff-only
venv\Scripts\python.exe scripts\release_language.py %*
if not errorlevel 1 start "" "releases\language"
pause
