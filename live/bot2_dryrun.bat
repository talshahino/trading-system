@echo off
rem Bot 2: portfolio 2 (22 combos), risk 0.45% per trade - DRY RUN: logs orders, sends nothing
cd /d %~dp0..
python live\runner.py --config live\bots\bot2.json --terminal C:\MT5_bot2\terminal64.exe --dry-run
pause
