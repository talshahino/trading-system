@echo off
rem Stops every bot running in the background (started with botN_hidden.vbs)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*runner.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; 'Stopped: ' + $_.CommandLine }"
taskkill /F /IM cloudflared.exe >nul 2>&1 && echo Stopped: the real-time dashboard tunnel
echo.
echo Done.
pause
