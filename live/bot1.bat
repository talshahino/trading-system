@echo off
rem Bot 1: portfolio 1 (39 combos), risk 0.375% per trade
cd /d %~dp0..
python live\runner.py --config live\bots\bot1.json --terminal C:\MT5_bot1\terminal64.exe
pause
