"""Real-time dashboard feed: a small read-only web server inside the bot plus a free Cloudflare
"quick tunnel" (no account needed) that gives it a public https address.

- The bot keeps two snapshots in memory: LIVE (account + open positions, refreshed every quarter
  second) and FULL (the whole dashboard state, refreshed every 15 seconds).
- Server: http://127.0.0.1:<port>/live and /state, JSON, read-only, CORS open.
- Tunnel: cloudflared.exe (downloaded to live/ on first use) prints an address like
  https://xxxx.trycloudflare.com; the bot writes it into the GitHub state file ("live_url"),
  and the dashboard then reads /live from it 4 times a second.
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
ROUTES = {}            # extra read-only paths, e.g. "/candles": fn(query dict) -> json doc (or None for 404)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        doc = SNAP["live"] if path == "/live" else SNAP["full"] if path == "/state" else None
        if path in ROUTES:
            from urllib.parse import parse_qsl
            try:
                doc = ROUTES[path](dict(parse_qsl(self.path.partition("?")[2])))
            except Exception as e:  # noqa: BLE001
                doc = {"error": str(e)}
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


class ExclusiveServer(ThreadingHTTPServer):
    """A port only one program can hold (Windows otherwise lets a second copy bind the same port,
    which hides a duplicate bot or manager)."""
    allow_reuse_address = False

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(port):
    srv = ExclusiveServer(("127.0.0.1", port), _Handler)
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
        wait = 30
        while True:
            if not _ensure_exe(log):
                time.sleep(600)
                continue
            try:
                flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0   # no extra black window
                attempt = getattr(run, "n", 0)
                run.n = attempt + 1
                proto = ["--protocol", "http2"] if attempt % 2 else []     # some networks block QUIC (UDP): try plain TCP too
                p = subprocess.Popen([str(EXE), "tunnel", "--no-autoupdate", *proto, "--url", f"http://127.0.0.1:{port}"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags,
                                     encoding="utf-8", errors="replace")   # cloudflared prints UTF-8; Windows' default (cp1252) crashed on it
                pid_file.write_text(str(p.pid))
                tail = []
                for line in p.stdout:
                    tail = (tail + [line.strip()])[-12:]
                    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                    if m and m.group(0) != URL["url"]:
                        URL["url"], URL["err"] = m.group(0), None
                        wait = 30
                        on_url(URL["url"])
                p.wait()
                why = next((t for t in reversed(tail) if " ERR " in t or "error" in t.lower() or "failed" in t.lower()), tail[-1] if tail else "")
                URL["err"] = why[-200:]
                log(f"tunnel stopped (code {p.returncode}): {why[-200:]}; restarting in {wait} s")
            except Exception as e:  # noqa: BLE001
                URL["err"] = str(e)
                log(f"tunnel error: {e}; restarting in {wait} s")
                try:
                    p.kill()                    # never leave a running tunnel behind before starting another one
                except Exception:  # noqa: BLE001
                    pass
            URL["url"] = None
            time.sleep(wait)
            wait = min(wait * 2, 600)        # Cloudflare limits how often a new quick tunnel may be asked for
    threading.Thread(target=run, daemon=True).start()
