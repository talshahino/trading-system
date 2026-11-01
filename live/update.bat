@echo off
rem Downloads the latest robot code from GitHub into C:\trading-system (logs, token and state files are kept)
echo Close the bot windows before updating. Press any key to continue...
pause >nul
set "T=%TEMP%\ts_%RANDOM%%RANDOM%"
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; Invoke-WebRequest -UseBasicParsing ('https://github.com/talshahino/trading-system/archive/refs/heads/main.zip?x=' + [guid]::NewGuid()) -OutFile '%T%.zip'; Expand-Archive -Force '%T%.zip' '%T%'"
if errorlevel 1 (
  echo.
  echo UPDATE FAILED - nothing was changed. Send a screenshot of this window.
  pause
  exit /b 1
)
robocopy "%T%\trading-system-main" "C:\trading-system" /E /NFL /NDL /NJH /NJS
findstr /c:"running: checking the strategies" C:\trading-system\live\runner.py >nul && echo. && echo Updated OK. Start the bot again (bot1.bat). || echo UPDATE DID NOT APPLY - send a screenshot.
pause
