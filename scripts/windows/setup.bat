@echo off
rem One-time setup on the Windows laptop, from the nics-ai-lab folder:  scripts\windows\setup.bat
cd /d "%~dp0\..\.."
if not exist venv python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements\cpu.txt
pip install MetaTrader5
python tests\test_trading_pipeline.py
