@echo off
rem NICS Trading opinion on every signal sent to users -> Telegram, to Alexander only.
rem Shadow only: nothing changes signals, users, the bot or the EA.
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
venv\Scripts\python.exe scripts\opinion_live.py >> logs\opinion.txt 2>&1
