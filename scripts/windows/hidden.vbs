' Runs the .bat given as the first argument with no console window (used by the scheduled tasks).
CreateObject("WScript.Shell").Run """" & WScript.Arguments(0) & """", 0, True
