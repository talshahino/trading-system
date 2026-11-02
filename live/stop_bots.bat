@echo off
rem Stops the manager and every bot running in the background (they start again with Windows or manager_hidden.vbs)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and ($_.CommandLine -like '*runner.py*' -or $_.CommandLine -like '*manager.py*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; 'Stopped: ' + $_.CommandLine }"
taskkill /F /IM cloudflared.exe >nul 2>&1 && echo Stopped: the real-time dashboard tunnel
echo.
echo Done.
pause
