' Starts bot 1 with no window (log: C:\trading-system\live\bot1.log; stop it with stop_bots.bat)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\trading-system"
sh.Run "pythonw live\runner.py --config live\bots\bot1.json --terminal C:\MT5_bot1\terminal64.exe", 0, False
