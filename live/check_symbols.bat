@echo off
rem Which of bot 2 / bot 3 instruments exist on the demo server of the standard MetaTrader 5 (read-only)
cd /d %~dp0..
python live\check_symbols.py "C:\Program Files\MetaTrader 5\terminal64.exe"
pause
