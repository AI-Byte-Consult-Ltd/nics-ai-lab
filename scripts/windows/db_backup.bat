@echo off
rem Nightly full backup of the NICS Trading database to this laptop (read-only on the server).
rem Needs NICS_BACKUP_KEY (and NICS_DATA_URL) set once with setx (see docs/laptop-guide-ru.md, section 11).
rem Backups: C:\NICS\backups\db\<date_time>\ ; the last 14 are kept.
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
venv\Scripts\python.exe scripts\db_backup.py %* >> logs\db_backup.txt 2>&1
