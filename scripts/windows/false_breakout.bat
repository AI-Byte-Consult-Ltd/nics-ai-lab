@echo off
rem False breakouts on public hourly history since 2008: can they be told apart at entry? (research, nothing is sent to trading)
rem First run downloads Dukascopy hourly history for all instruments (about an hour, cached), later runs take minutes.
rem The summary goes to the private dashboard and to experiments\research\false_breakout_last.txt
cd /d "%~dp0\..\.."
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
git pull --ff-only
venv\Scripts\python.exe scripts\false_breakout_research.py --fetch %*
pause
