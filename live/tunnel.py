"""Real-time dashboard feed: a small read-only web server inside the bot plus a free Cloudflare
"quick tunnel" (no account needed) that gives it a public https address.

- The bot keeps two snapshots in memory: LIVE (account + open positions, refreshed every half
  second) and FULL (the whole dashboard state, refreshed every 15 seconds).
- Server: http://127.0.0.1:<port>/live and /state, JSON, read-only, CORS open.
- Tunnel: cloudflared.exe (downloaded to live/ on first use) prints an address like
  https://xxxx.trycloudflare.com; the bot writes it into the GitHub state file ("live_url"),
  and the dashboard then reads /live from it every second.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "live" / "cloudflared.exe"
DOWNLOAD = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

SNAP = {"live": {}, "full": {}}
URL = {"url": None, "err": None}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        doc = SNAP["live"] if path == "/live" else SNAP["full"] if path == "/state" else None
        if doc is None:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(doc, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


def serve(port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _ensure_exe(log):
    if EXE.exists() and EXE.stat().st_size > 1_000_000:
        return True
    try:
        import requests
        log("downloading cloudflared (the free tunnel program, ~60 MB, once)...")
        with requests.get(DOWNLOAD, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = EXE.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(EXE)
        return True
    except Exception as e:  # noqa: BLE001
        URL["err"] = f"cloudflared download failed: {e}"
        log(URL["err"])
        return False


def _kill_old(pid_file):
    """A tunnel left behind by an earlier run of this bot (it has no window, so closing the bot doesn't stop it)."""
    try:
        pid = int(pid_file.read_text().strip())
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def tunnel(port, log, on_url, name="bot"):
    """Keep a quick tunnel running (restart it if it stops); call on_url(url) whenever the address changes."""
    pid_file = ROOT / "live" / f"{name}_tunnel.pid"
    _kill_old(pid_file)

    def run():
        while True:
            if not _ensure_exe(log):
                time.sleep(600)
                continue
            try:
                flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0   # no extra black window
                p = subprocess.Popen([str(EXE), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags)
                pid_file.write_text(str(p.pid))
                for line in p.stdout:
                    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                    if m and m.group(0) != URL["url"]:
                        URL["url"] = m.group(0)
                        on_url(URL["url"])
                p.wait()
                log(f"tunnel stopped (code {p.returncode}); restarting in 30 s")
            except Exception as e:  # noqa: BLE001
                log(f"tunnel error: {e}; restarting in 30 s")
            URL["url"] = None
            time.sleep(30)
    threading.Thread(target=run, daemon=True).start()
