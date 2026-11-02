' Starts the bots' manager with no window (log: C:\trading-system\live\manager.log). It starts the bots.
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\trading-system"
sh.Run "pythonw live\manager.py", 0, False
