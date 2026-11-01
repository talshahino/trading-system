@echo off
rem Stops every bot running in the background (started with botN_hidden.vbs)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*runner.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; 'Stopped: ' + $_.CommandLine }"
echo.
echo Done.
pause
