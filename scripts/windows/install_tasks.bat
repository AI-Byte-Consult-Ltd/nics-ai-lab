@echo off
rem Registers the nightly self-learning, hourly shadow-scoring, 2-minute opinion and nightly DB backup tasks.
rem Run once from the nics-ai-lab folder (cmd or PowerShell):  scripts\windows\install_tasks.bat
rem A missed night (laptop off or asleep at 03:30) runs as soon as the laptop is on again.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_tasks.ps1"
