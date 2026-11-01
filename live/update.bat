@echo off
rem Downloads the latest robot code from GitHub into C:\trading-system (logs and state files are kept)
echo Close the bot windows before updating. Press any key to continue...
pause >nul
powershell -NoProfile -Command "Invoke-WebRequest https://github.com/talshahino/trading-system/archive/refs/heads/main.zip -OutFile $env:TEMP\ts.zip; Remove-Item -Recurse -Force $env:TEMP\ts -ErrorAction SilentlyContinue; Expand-Archive $env:TEMP\ts.zip $env:TEMP\ts"
robocopy "%TEMP%\ts\trading-system-main" "C:\trading-system" /E /NFL /NDL /NJH /NJS
echo.
echo Updated. Start the bot again (bot1.bat).
pause
