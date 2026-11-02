@echo off
rem Bot 2: portfolio 2 (22 combos), risk 0.45% per trade
cd /d %~dp0..
set BOT_LOOP=1
:run
python live\runner.py --config live\bots\bot2.json --terminal C:\MT5_bot2\terminal64.exe
if %errorlevel%==3 goto run
pause
