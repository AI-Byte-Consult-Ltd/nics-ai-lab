# Registers three scheduled tasks for the current user (called by install_tasks.bat).
#   "NICS Trading self-learning"  every night at 03:30; if the laptop was off or asleep,
#                                 it runs as soon as the laptop is on again (StartWhenAvailable)
#   "NICS Trading shadow score"   every hour; missed hours are simply skipped
#   "NICS Trading opinion"        every 2 minutes: model opinion on new sent signals -> Telegram (Alexander)
# The first two wake the laptop from sleep (WakeToRun), wait for a network connection and also run on battery.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6)
$learn = New-ScheduledTaskAction -Execute "$here\self_learn.bat"
$daily = New-ScheduledTaskTrigger -Daily -At 03:30
Register-ScheduledTask -TaskName 'NICS Trading self-learning' -Action $learn -Trigger $daily `
    -Settings $settings -Force | Out-Null

$hourlySettings = New-ScheduledTaskSettingsSet -WakeToRun -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$score = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$here\hidden.vbs`" `"$here\shadow_score.bat`""
$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName 'NICS Trading shadow score' -Action $score -Trigger $hourly `
    -Settings $hourlySettings -Force | Out-Null

$opSettings = New-ScheduledTaskSettingsSet -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$opinion = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$here\hidden.vbs`" `"$here\opinion.bat`""
$every2 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 2) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName 'NICS Trading opinion' -Action $opinion -Trigger $every2 `
    -Settings $opSettings -Force | Out-Null

Write-Host ''
Write-Host 'Done. Three tasks registered:'
Get-ScheduledTask -TaskName 'NICS Trading*' | Format-Table TaskName, State -AutoSize
Write-Host 'Remove: scripts\windows\remove_tasks.bat'
