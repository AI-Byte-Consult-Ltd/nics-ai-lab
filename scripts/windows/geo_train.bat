@echo off
rem NICS Vision Geo on the laptop: fetch metadata, download photos, train from random weights.
rem   scripts\windows\geo_train.bat                      everything, full speed
rem   scripts\windows\geo_train.bat --max-minutes 60 --gentle   an hour, low priority (daytime)
rem First run downloads the museum photos (about 1-2 GB of traffic, kept small on disk).
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
git pull --ff-only
venv\Scripts\python.exe scripts\geo_fetch.py --sources smithsonian,nhm,commons >> logs\geo.txt 2>&1
venv\Scripts\python.exe scripts\geo_build.py >> logs\geo.txt 2>&1
venv\Scripts\python.exe scripts\train_geo.py %* >> logs\geo.txt 2>&1
type logs\geo.txt | more
pause
