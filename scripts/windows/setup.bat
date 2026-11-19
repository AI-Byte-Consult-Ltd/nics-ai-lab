@echo off
rem One-time setup on the Windows laptop. Works from cmd and PowerShell:  scripts\windows\setup.bat
rem Uses venv\Scripts\python.exe directly, so PowerShell's script policy (Activate.ps1) does not matter.
cd /d "%~dp0\..\.."
if not exist venv\Scripts\python.exe (
  python -m venv venv || goto :fail
)
set PY=venv\Scripts\python.exe
%PY% -m pip install --upgrade pip || goto :fail
%PY% -m pip install -r requirements\cpu.txt || goto :fail
%PY% -m pip install MetaTrader5 || echo MetaTrader5 package not installed: only needed for fetch_mt5_history.py
%PY% tests\test_trading_pipeline.py || goto :fail
echo.
echo Setup OK. Next: .\venv\Scripts\python.exe scripts\self_learn.py --from-scratch
exit /b 0
:fail
echo.
echo SETUP FAILED. Send the text above to Claude.
exit /b 1
