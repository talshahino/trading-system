' Starts bot 2 with no window (log: C:\trading-system\live\bot2.log; stop it with stop_bots.bat)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\trading-system"
sh.Run "pythonw live\runner.py --config live\bots\bot2.json --terminal C:\MT5_bot2\terminal64.exe", 0, False
