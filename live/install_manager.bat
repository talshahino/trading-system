@echo off
rem One-time setup of the bots' manager (runs hidden in the background, starts with Windows).
cd /d C:\trading-system
echo Installing the packages the manager needs...
python -m pip install -q -r live\requirements.txt
if errorlevel 1 (
  echo.
  echo PACKAGE INSTALL FAILED - send a screenshot of this window.
  pause
  exit /b 1
)
echo.
python live\manager.py --set-pin --only-if-missing
copy /y live\manager_hidden.vbs "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\ftmo_bots_manager.vbs" >nul
echo.
echo Starting the manager...
wscript live\manager_hidden.vbs
timeout /t 5 >nul
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*manager.py*' }) { 'MANAGER IS RUNNING. It also starts by itself when Windows starts.' } else { 'THE MANAGER DID NOT START - send a screenshot.' }"
echo.
pause
