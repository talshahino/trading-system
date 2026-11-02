"""The bots' manager: one always-on background program on the trading computer (your PC now, a VPS
later). The app / dashboard talks to it through ONE free Cloudflare tunnel.

What it does
- Starts every enabled bot (hidden, no black window) and restarts a bot that stops.
- Adds a trading account sent from the app: number, password, server, which bot. The password is
  encrypted with Windows' own protection (DPAPI: only this Windows user on this computer can read it)
  and is never sent anywhere else. A copy of MetaTrader is made for the account and the bot logs in
  through it.
- Start / stop a bot from the app.
- Phone notifications straight to the app (Web Push), with an "approve update" button.
- Checks GitHub for new bot code and installs it when you approve in the app, then restarts all.

Security: every action needs your secret code (PIN), set once at install. Its hash only is stored.
After 5 wrong codes, actions are locked for 15 minutes. Reading the live data stays open, as before.

  python live/manager.py            (live/manager.bat, or hidden at Windows start: manager_hidden.vbs)
  python live/manager.py --set-pin  (change the secret code)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import requests  # noqa: E402

from live import reporter as REP  # noqa: E402
from live import tunnel as TUN  # noqa: E402
from live import updater as UPD  # noqa: E402

LIVE = ROOT / "live"
CFG_FILE = LIVE / "manager.json"
VAPID_FILE = LIVE / "vapid_private.pem"
PORT = 8760
BOT_TEMPLATES = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3"}
JOURNAL = deque(maxlen=300)
PROCS = {}
LOCK = threading.RLock()
FAILS = {"n": 0, "until": 0}
CFG = {}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    JOURNAL.append(line)
    try:
        with open(LIVE / "manager.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass
    if sys.stdout:
        print(line, flush=True)


# ---------------- settings and secrets ----------------
def load():
    global CFG
    CFG = json.loads(CFG_FILE.read_text(encoding="utf-8")) if CFG_FILE.exists() else {}
    CFG.setdefault("accounts", [])
    CFG.setdefault("subs", [])
    CFG.setdefault("local_key", secrets.token_hex(16))   # bots use it to ask for a push (localhost only)


def save():
    with LOCK:
        tmp = CFG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(CFG, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, CFG_FILE)


def set_pin(pin):
    salt = secrets.token_hex(16)
    CFG["pin_salt"] = salt
    CFG["pin_hash"] = hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), 200_000).hex()
    save()


def pin_ok(pin):
    if time.time() < FAILS["until"] or not CFG.get("pin_hash") or not pin:
        return False
    h = hashlib.pbkdf2_hmac("sha256", str(pin).encode(), bytes.fromhex(CFG["pin_salt"]), 200_000).hex()
    if hmac.compare_digest(h, CFG["pin_hash"]):
        FAILS["n"] = 0
        return True
    FAILS["n"] += 1
    if FAILS["n"] >= 5:
        FAILS["until"], FAILS["n"] = time.time() + 900, 0
        log("5 wrong secret codes: actions locked for 15 minutes")
        push("קוד שגוי 5 פעמים", "מישהו הקליד קוד שגוי 5 פעמים. הפעולות נעולות ל־15 דקות.", urgent=True)
    return False


def _dpapi(data: bytes, protect: bool) -> bytes:
    """Windows DPAPI: data readable only by this Windows user on this computer."""
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    buf = ctypes.create_string_buffer(data, len(data))
    src, dst = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), BLOB()
    fn = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    if not fn(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        raise OSError("DPAPI failed")
    out = ctypes.string_at(dst.pbData, dst.cbData)
    ctypes.windll.kernel32.LocalFree(dst.pbData)
    return out


def encrypt(text):
    if os.name != "nt":
        raise OSError("passwords can be stored only on Windows")
    return base64.b64encode(_dpapi(text.encode(), True)).decode()


def decrypt(blob):
    return _dpapi(base64.b64decode(blob), False).decode()


# ---------------- web push ----------------
def vapid_public():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    if not VAPID_FILE.exists():
        key = ec.generate_private_key(ec.SECP256R1())
        VAPID_FILE.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                 serialization.NoEncryption()))
    key = serialization.load_pem_private_key(VAPID_FILE.read_bytes(), None)
    raw = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def push(title, body, urgent=False, action=None, tag=None):
    """Notification to every phone/computer that turned notifications on in the app."""
    subs = list(CFG.get("subs", []))
    if not subs:
        return
    try:
        from pywebpush import WebPushException, webpush
    except ImportError:
        log("pywebpush is not installed; run: python -m pip install -r requirements.txt")
        return
    quiet = REP.quiet_now()                  # quiet hours: everything silent, no exceptions (the user's rule)
    data = json.dumps({"title": title, "body": body, "silent": quiet, "action": action, "tag": tag}, ensure_ascii=False)
    dead = []
    for s in subs:
        try:
            webpush(subscription_info=s, data=data, vapid_private_key=str(VAPID_FILE),
                    vapid_claims={"sub": "mailto:bots@trading-system.local"}, ttl=24 * 3600)
        except WebPushException as e:
            code = getattr(e.response, "status_code", None)
            if code in (404, 410):
                dead.append(s)
            else:
                log(f"push failed: {e}")
        except Exception as e:  # noqa: BLE001
            log(f"push failed: {e}")
    if dead:
        with LOCK:
            CFG["subs"] = [s for s in CFG["subs"] if s not in dead]
        save()


# ---------------- bots ----------------
def accounts():
    return CFG["accounts"]


def acc_by(name):
    return next((a for a in accounts() if a["name"] == name), None)


def running(name):
    p = PROCS.get(name)
    return p is not None and p.poll() is None


def start_bot(a):
    if running(a["name"]):
        return
    env = os.environ.copy()
    env.update(MANAGED="1", BOT_PORT=str(a["port"]), MANAGER_KEY=CFG["local_key"], BOT_NICKNAME=a.get("nickname") or "")
    if a.get("password"):
        env.update(MT5_LOGIN=str(a["login"]), MT5_SERVER=a["server"], MT5_PASSWORD=decrypt(a["password"]))
    py = sys.executable.replace("pythonw.exe", "python.exe")
    flags = 0x08000000 if os.name == "nt" else 0          # no window: only MetaTrader shows
    PROCS[a["name"]] = subprocess.Popen([py, str(LIVE / "runner.py"), "--config", str(LIVE / "bots" / f"{a['name']}.json"),
                                         "--terminal", a["terminal"]], cwd=str(ROOT), env=env, creationflags=flags,
                                        stdout=subprocess.DEVNULL, stderr=open(LIVE / f"{a['name']}_stderr.log", "a"))
    a["started"] = time.time()
    log(f"started {a['name']}")


def port_owner(port):
    """The process listening on a local port (Windows netstat), or None."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) >= 5 and f[1].endswith(f":{port}") and f[3].upper() == "LISTENING":
                return int(f[4])
    except Exception:  # noqa: BLE001
        pass
    return None


def free_port(a):
    """A bot left running by an earlier manager (or by botN.bat) still holds the account's port and runs
    old code: stop it so this manager can start (and later update) the bot itself."""
    pid = port_owner(a["port"])
    if pid and pid != os.getpid() and not running(a["name"]):
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=20)
        log(f"{a['name']}: stopped an older copy (pid {pid}) that was holding port {a['port']}; starting the managed bot")
        time.sleep(3)


def stop_bot(name):
    p = PROCS.pop(name, None)
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(15)
        except Exception:  # noqa: BLE001
            p.kill()
        log(f"stopped {name}")


def watchdog():
    while True:
        try:
            for a in list(accounts()):
                if a.get("enabled") and not running(a["name"]):
                    if a.get("started") and time.time() - a["started"] < 120:
                        continue                              # it just stopped right after starting: wait
                    p = PROCS.get(a["name"])
                    if p is not None and p.returncode == 4:   # an older copy holds the port: take over
                        free_port(a)
                    elif a.get("started"):
                        log(f"{a['name']} stopped by itself: starting it again")
                        push(f"{a.get('label', a['name'])} נעצר", "המנהל מפעיל אותו מחדש.", urgent=True)
                    start_bot(a)
        except Exception:  # noqa: BLE001
            log(f"watchdog error:\n{traceback.format_exc()}")
        time.sleep(30)


def add_account(d):
    bot, login, server = d.get("bot"), str(d.get("login", "")).strip(), str(d.get("server", "")).strip()
    password, nickname = d.get("password") or "", str(d.get("nickname", "")).strip()[:40]
    if bot not in BOT_TEMPLATES or not login.isdigit() or not server or not password or not nickname:
        raise ValueError("חסר מידע: שם לחשבון, בוט, מספר חשבון, שרת וסיסמה")
    if "demo" not in server.lower() and "trial" not in server.lower():
        raise ValueError("כרגע מותר רק חשבון דמו (השרת חייב להכיל Demo)")
    name = f"{bot}_{login}"
    if acc_by(name):
        raise ValueError("החשבון הזה כבר מחובר לבוט הזה")
    if any(str(a.get("login")) == login for a in accounts()):
        raise ValueError("החשבון הזה כבר מחובר לבוט אחר. אסור שני בוטים על אותו חשבון.")
    template = Path(d.get("template") or CFG.get("mt5_template") or r"C:\MT5_bot1")
    if not (template / "terminal64.exe").exists():
        raise ValueError(f"לא מצאתי MetaTrader להעתקה ב־{template}")
    dest = Path(rf"C:\MT5_{login}")
    if not dest.exists():
        log(f"copying MetaTrader to {dest} ...")
        shutil.copytree(template, dest, ignore=shutil.ignore_patterns("logs", "Logs", "Bases", "Tester", "*.log"))
    cfg = json.loads((LIVE / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
    cfg.update(name=name, label=f"{nickname} · {BOT_TEMPLATES[bot]}", bot_label=BOT_TEMPLATES[bot], account=int(login), live_tunnel=False)
    (LIVE / "bots" / f"{name}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    port = max([PORT] + [a["port"] for a in accounts()]) + 1
    a = {"name": name, "bot": bot, "nickname": nickname, "label": f"{nickname} · {BOT_TEMPLATES[bot]}", "login": int(login), "server": server,
         "terminal": str(dest / "terminal64.exe"), "port": port, "enabled": True, "password": encrypt(password),
         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with LOCK:
        accounts().append(a)
    save()
    log(f"account added: {a['label']} ({server})")
    start_bot(a)
    return a


def rename_account(name, nickname):
    nickname = str(nickname or "").strip()[:40]
    if not nickname:
        raise ValueError("צריך לכתוב שם")
    a = acc_by(name)
    a["nickname"] = nickname
    a["label"] = f"{nickname} · {BOT_TEMPLATES.get(a.get('bot'), a['name'])}"
    save()
    log(f"renamed {name}: {a['label']}")
    if running(name):                          # the bot shows the new name from its next start
        stop_bot(name)
        start_bot(a)
    push_manager_state()


def remove_account(name):
    stop_bot(name)
    with LOCK:
        CFG["accounts"] = [a for a in accounts() if a["name"] != name]
    save()
    log(f"account removed: {name}")


def public_accounts(with_run=True):
    out = []
    for a in accounts():
        out.append({"name": a["name"], "label": a.get("label", a["name"]), "nickname": a.get("nickname"), "bot": a.get("bot"), "login": a.get("login"),
                    "server": a.get("server"), "enabled": a.get("enabled", False), "running": running(a["name"]) if with_run else None,
                    "archived": bool(a.get("archived")), "archived_at": a.get("archived_at"), "hidden": bool(a.get("hidden")),
                    "created": a.get("created"), "broker": "FTMO", "platform": "MetaTrader 5", "target_pct": a.get("target_pct", 10),
                    "withdrawals": a.get("withdrawals", [])})
    return out


def archive_account(name):
    """'Close account' in the app: the bot stops for good and the account moves to the history list."""
    a = acc_by(name)
    a["enabled"] = False
    stop_bot(name)
    a["archived"] = True
    a["archived_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    log(f"account closed (moved to history): {name}")
    push_manager_state()


def hide_account(name, hidden):
    a = acc_by(name)
    a["hidden"] = bool(hidden)
    save()
    push_manager_state()


# ---------------- updates ----------------
UPDATE = {"sha": None, "title": None, "checked": 0}


def update_check():
    while True:
        try:
            sha, title = UPD.remote()
            if UPD.local() is None:
                UPD.VERSION.write_text(sha)
            elif sha != UPD.local() and UPDATE["sha"] != sha:
                if not UPD.touches_bot(UPD.local(), sha):
                    UPD.VERSION.write_text(sha)
                else:
                    UPDATE.update(sha=sha, title=title)
                    log(f"update available: {sha[:7]} {title}")
                    push("יש עדכון לבוטים", title, action="approve_update", tag="update")
            UPDATE["checked"] = time.time()
        except Exception as e:  # noqa: BLE001
            log(f"update check failed: {e}")
        time.sleep(300)


def install_update():
    sha = UPDATE["sha"]
    if not sha:
        raise ValueError("אין עדכון שמחכה")
    log(f"installing update {sha[:7]}: stopping the bots")
    push("מתעדכן", "עוצר את הבוטים, מתקין ומפעיל הכל מחדש. העסקאות נשארות עם הסטופ והיעד.", tag="update")
    names = [a["name"] for a in accounts() if running(a["name"])]
    for n in names:
        stop_bot(n)
    for a in accounts():
        if a.get("enabled"):
            free_port(a)
    try:
        UPD.apply(sha)
    except Exception as e:  # noqa: BLE001
        log(f"update failed: {e}")
        push("העדכון נכשל", f"{e}. הבוטים חוזרים לעבוד בגרסה הקודמת.", urgent=True, tag="update")
        for a in accounts():
            if a.get("enabled"):
                start_bot(a)
        return
    CFG["updated_to"] = sha
    save()
    log("update installed: restarting the manager")
    subprocess.Popen([sys.executable] + sys.argv, cwd=str(ROOT), creationflags=(0x00000008 | 0x00000200) if os.name == "nt" else 0)
    os._exit(0)


# ---------------- API ----------------
def bot_port(name):
    a = acc_by(name)
    return a["port"] if a else None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, doc):
        body = json.dumps(doc, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Pin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_OPTIONS(self):  # noqa: N802
        self._send(204, {})

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        parts = path.strip("/").split("/")
        if path in ("", "/status"):
            return self._send(200, {"ok": True, "accounts": public_accounts(), "version": (UPD.local() or "")[:7],
                                    "update": {"sha": (UPDATE["sha"] or "")[:7], "title": UPDATE["title"]} if UPDATE["sha"] else None,
                                    "pin_set": bool(CFG.get("pin_hash")), "push_devices": len(CFG.get("subs", [])),
                                    "watchdog_push": WATCH["ok"]})
        if path == "/vapid":
            return self._send(200, {"key": vapid_public()})
        if len(parts) == 3 and parts[0] == "bot" and parts[2] in ("live", "state", "candles"):
            port = bot_port(parts[1])
            if not port:
                return self._send(404, {"error": "no such bot"})
            try:
                q = self.path.partition("?")[2] if parts[2] == "candles" else ""
                r = requests.get(f"http://127.0.0.1:{port}/{parts[2]}" + (f"?{q}" if q else ""), timeout=15 if q else 5)
                return self._send(r.status_code, r.json())
            except Exception:  # noqa: BLE001
                return self._send(503, {"error": "bot not answering"})
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            d = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:  # noqa: BLE001
            return self._send(400, {"error": "bad request"})
        if path == "/push":                                   # from a bot on this computer
            if self.headers.get("X-Local") != CFG["local_key"]:
                return self._send(403, {"error": "forbidden"})
            push(d.get("title", ""), d.get("body", ""), urgent=bool(d.get("urgent")), tag=d.get("tag"))
            return self._send(200, {"ok": True})
        if time.time() < FAILS["until"]:
            return self._send(429, {"error": "נעול ל־15 דקות אחרי 5 קודים שגויים"})
        if not pin_ok(self.headers.get("X-Pin") or d.get("pin")):
            return self._send(401, {"error": "קוד שגוי"})
        try:
            if path == "/auth":
                return self._send(200, {"ok": True})
            if path == "/subscribe":
                sub = d.get("subscription")
                if not sub or "endpoint" not in sub:
                    raise ValueError("no subscription")
                with LOCK:
                    if sub not in CFG["subs"]:
                        CFG["subs"].append(sub)
                save()
                push("התראות פעילות", "מעכשיו ההתראות של הבוטים יגיעו לאפליקציה הזו.")
                threading.Thread(target=push_manager_state, daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/accounts/add":
                a = add_account(d)
                return self._send(200, {"ok": True, "name": a["name"]})
            if path.startswith("/accounts/") and path.endswith("/rename"):
                name = path.split("/")[2]
                if not acc_by(name):
                    raise ValueError("אין חשבון כזה")
                rename_account(name, d.get("nickname"))
                return self._send(200, {"ok": True})
            if path.startswith("/accounts/") and path.endswith(("/withdraw", "/withdraw_del")):
                a = acc_by(path.split("/")[2])
                if not a:
                    raise ValueError("אין חשבון כזה")
                w = a.setdefault("withdrawals", [])
                if path.endswith("/withdraw_del"):
                    a["withdrawals"] = [x for x in w if x.get("id") != d.get("id")]
                else:
                    amt = round(float(d.get("amount")), 2)
                    if not 0 < amt < 10_000_000:
                        raise ValueError("סכום לא תקין")
                    date = str(d.get("date") or time.strftime("%Y-%m-%d"))[:10]
                    time.strptime(date, "%Y-%m-%d")
                    w.append({"id": f"{int(time.time() * 1000)}", "amount": amt, "date": date, "note": str(d.get("note") or "")[:80]})
                    w.sort(key=lambda x: x["date"])
                save()
                push_manager_state()
                return self._send(200, {"ok": True})
            if path.startswith("/accounts/") and path.endswith("/target"):
                a = acc_by(path.split("/")[2])
                if not a:
                    raise ValueError("אין חשבון כזה")
                v = float(d.get("pct"))
                if not 0.5 <= v <= 50:
                    raise ValueError("יעד בין 0.5% ל־50%")
                a["target_pct"] = round(v, 2)
                save()
                push_manager_state()
                return self._send(200, {"ok": True})
            if path.startswith("/accounts/") and path.endswith(("/archive", "/hide")):
                name, act = path.split("/")[2], path.split("/")[3]
                if not acc_by(name):
                    raise ValueError("אין חשבון כזה")
                if act == "archive":
                    archive_account(name)
                else:
                    hide_account(name, d.get("hidden", True))
                return self._send(200, {"ok": True})
            if path.startswith("/accounts/") and path.endswith(("/start", "/stop", "/remove")):
                name, act = path.split("/")[2], path.split("/")[3]
                a = acc_by(name)
                if not a:
                    raise ValueError("אין חשבון כזה")
                if act == "start":
                    if a.get("archived"):
                        raise ValueError("החשבון סגור ונמצא בהיסטוריה")
                    a["enabled"] = True
                    save()
                    start_bot(a)
                elif act == "stop":
                    a["enabled"] = False
                    save()
                    stop_bot(name)
                else:
                    remove_account(name)
                return self._send(200, {"ok": True})
            if path == "/logs":                          # behind the secret code only; never published
                which = str(d.get("which") or "manager")
                if which != "manager" and not acc_by(which):
                    raise ValueError("אין לוג כזה")
                f = LIVE / f"{which}.log"
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()[-400:] if f.exists() else []
                lines = [l for l in lines if "password" not in l.lower()]
                return self._send(200, {"ok": True, "lines": lines})
            if path == "/update/approve":
                threading.Thread(target=install_update, daemon=True).start()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            log(f"action error:\n{traceback.format_exc()}")
            return self._send(500, {"error": str(e)})


WATCH = {"ok": None, "msg": ""}


def watchdog_secret():
    """Give GitHub's watchdog (it notices when this computer is off) the key it needs to send the
    alert straight to the app: the push signing key is stored as an encrypted GitHub Actions secret.
    Needs the GitHub token to have "Secrets: Read and write" on this repository."""
    tok = REP.token()
    if not tok:
        WATCH.update(ok=False, msg="no GitHub token")
        return
    try:
        from nacl import encoding, public
        h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
        base = f"https://api.github.com/repos/{REP.REPO}/actions/secrets"
        r = requests.get(base + "/public-key", headers=h, timeout=20)
        if r.status_code in (401, 403, 404):
            WATCH.update(ok=False, msg="the GitHub token needs the Secrets permission")
            log("computer-off alerts to the app: the GitHub token needs 'Secrets: Read and write' (see the app)")
            return
        r.raise_for_status()
        k = r.json()
        box = public.SealedBox(public.PublicKey(k["key"].encode(), encoding.Base64Encoder()))
        val = base64.b64encode(box.encrypt(VAPID_FILE.read_bytes())).decode()
        r = requests.put(base + "/VAPID_PRIVATE_PEM", headers=h, timeout=20, json={"encrypted_value": val, "key_id": k["key_id"]})
        r.raise_for_status()
        WATCH.update(ok=True, msg="")
        log("computer-off alerts: GitHub's watchdog can now notify the app directly")
    except Exception as e:  # noqa: BLE001
        WATCH.update(ok=False, msg=str(e)[:120])
        log(f"computer-off alerts setup failed: {e}")


def push_manager_state():
    """Tell the app where the manager is (public GitHub file). It also lists the app's notification
    subscriptions for GitHub's watchdog: without the private signing key (kept on this computer and,
    encrypted, as a GitHub secret) nobody can send anything to them."""
    state = {"url": TUN.URL["url"], "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "accounts": public_accounts(with_run=False),
             "subs": CFG.get("subs", []), "tunnel_error": None if TUN.URL["url"] else (TUN.URL.get("err") or "connecting")}
    err = REP.push_state("manager", state)
    if err:
        log(err)


def heartbeat():
    """Every 10 minutes: the app's address + a sign of life for GitHub's watchdog."""
    while True:
        time.sleep(600)
        push_manager_state()
        if not WATCH["ok"]:
            watchdog_secret()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-pin", action="store_true")
    ap.add_argument("--only-if-missing", action="store_true")
    args = ap.parse_args()
    load()
    if args.set_pin and args.only_if_missing and CFG.get("pin_hash"):
        print("A secret code is already set (to change it: python live\\manager.py --set-pin).")
        return
    if (args.set_pin or not CFG.get("pin_hash")) and not sys.stdin:
        log("no secret code set: run live\\install_manager.bat first")
        return
    if args.set_pin or not CFG.get("pin_hash"):
        import getpass
        while True:
            p1 = getpass.getpass("Choose a secret code for the app (at least 6 digits/letters): ")
            if len(p1) >= 6 and p1 == getpass.getpass("Type it again: "):
                set_pin(p1)
                print("Saved.")
                break
            print("Too short or not the same - try again.")
        if args.set_pin:
            return
    if not accounts() and (LIVE / "bots" / "bot1.json").exists():
        # the bot already running on this computer (MetaTrader logged in by you)
        accounts().append({"name": "bot1", "bot": "bot1", "label": "בוט 1", "login": None, "server": None,
                           "terminal": r"C:\MT5_bot1\terminal64.exe", "port": 8761, "enabled": True})
    save()
    vapid_public()
    try:
        srv = TUN.ExclusiveServer(("127.0.0.1", PORT), Handler)
    except OSError:
        log("another copy of the manager is already running: this one stops")
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"manager running on port {PORT}; accounts: {', '.join(a['name'] for a in accounts()) or 'none'}")

    def on_url(u):
        log(f"app connection: {u}")
        push_manager_state()
    TUN.tunnel(PORT, log, on_url, "manager")
    if CFG.pop("updated_to", None):
        save()
        push("העדכון הותקן", f"הכל רץ בגרסה {(UPD.local() or '?')[:7]}.")
    for a in accounts():
        if a.get("enabled"):
            free_port(a)
            start_bot(a)
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=update_check, daemon=True).start()
    threading.Thread(target=watchdog_secret, daemon=True).start()
    threading.Thread(target=heartbeat, daemon=True).start()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
