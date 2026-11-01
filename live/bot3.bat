@echo off
rem Bot 3: portfolio 3 (26 combos), risk 0.375% per trade
cd /d %~dp0..
python live\runner.py --config live\bots\bot3.json --terminal C:\MT5_bot3\terminal64.exe
pause
