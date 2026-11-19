@echo off
rem Registers two Windows scheduled tasks for the current user:
rem   "NICS Trading self-learning"  every night at 03:30
rem   "NICS Trading shadow score"   every hour
rem Run once from the nics-ai-lab folder:  scripts\windows\install_tasks.bat
set HERE=%~dp0
schtasks /Create /F /SC DAILY /ST 03:30 /TN "NICS Trading self-learning" /TR "\"%HERE%self_learn.bat\""
schtasks /Create /F /SC HOURLY /MO 1 /TN "NICS Trading shadow score" /TR "\"%HERE%shadow_score.bat\""
echo.
echo Done. Check: schtasks /Query /TN "NICS Trading self-learning"
echo Remove:    scripts\windows\remove_tasks.bat
