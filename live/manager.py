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
After 5 wrong codes a client (by its address) is locked for 15 minutes; 30 wrong codes from all
clients together lock everyone for the rest of those 15 minutes. Reading the live data stays open.
The computer's address is published encrypted with a random key (url_key) each phone fetches once.

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
import re
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
BOT_TEMPLATES = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4"}
JOURNAL = deque(maxlen=300)
PROCS = {}
LOCK = threading.RLock()
FAILS = {}          # per client (Cloudflare's CF-Connecting-IP): {"n", "until", "busy"}
GLOBAL_FAILS = deque(maxlen=200)   # times of wrong codes from every client (a cap for many clients)
PUSH_HOSTS = ("fcm.googleapis.com", "updates.push.services.mozilla.com", "web.push.apple.com", ".notify.windows.com")
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
    CFG.setdefault("url_key", secrets.token_hex(32))     # random key of the published computer address (not from a code)
    CFG.setdefault("paired", {})                         # which code holders already got url_key


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


def set_viewer_pin(pin):
    """A second, read-only code (e.g. for a family member): it may see and get notifications, never act."""
    salt = secrets.token_hex(16)
    CFG["viewer_salt"] = salt
    CFG["viewer_hash"] = hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), 200_000).hex()
    save()


def viewer_ok(pin):
    if not CFG.get("viewer_hash") or not pin:
        return False
    h = hashlib.pbkdf2_hmac("sha256", str(pin).encode(), bytes.fromhex(CFG["viewer_salt"]), 200_000).hex()
    return hmac.compare_digest(h, CFG["viewer_hash"])


def _hash(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", str(pin).encode(), bytes.fromhex(salt), 200_000).hex()


def check_codes(pin, client):
    """-> "owner", "viewer", None (wrong) or "locked". The attempt is reserved under the lock BEFORE the
    slow hash, so parallel wrong codes can never get more than 5 tries per client in 15 minutes; on top,
    at most 30 wrong codes from all clients together per 15 minutes. A client that is locked out does
    not lock out the others (the owner keeps control from his phone)."""
    now = time.time()
    with LOCK:
        rec = FAILS.setdefault(client, {"n": 0, "until": 0, "busy": 0})
        recent = sum(1 for t in GLOBAL_FAILS if now - t < 900)
        if now < rec["until"] or rec["n"] + rec["busy"] >= 5 or recent >= 30:
            return "locked"
        rec["busy"] += 1
    try:
        owner = bool(pin and CFG.get("pin_hash") and hmac.compare_digest(_hash(pin, CFG["pin_salt"]), CFG["pin_hash"]))
        viewer = (not owner) and bool(pin and CFG.get("viewer_hash")
                                      and hmac.compare_digest(_hash(pin, CFG["viewer_salt"]), CFG["viewer_hash"]))
    finally:
        with LOCK:
            rec["busy"] -= 1
    with LOCK:
        if owner or viewer:
            rec["n"] = 0
            return "owner" if owner else "viewer"
        rec["n"] += 1
        GLOBAL_FAILS.append(now)
        if rec["n"] >= 5:
            rec["until"], rec["n"] = now + 900, 0
            log(f"5 wrong secret codes from {client}: that client is locked for 15 minutes")
            threading.Thread(target=push, args=("קוד שגוי 5 פעמים",
                             "מישהו הקליד קוד שגוי 5 פעמים. הוא נחסם ל־15 דקות (המכשיר שלך לא נחסם)."),
                             kwargs={"urgent": True}, daemon=True).start()
        if len(FAILS) > 500:                 # forget old clients
            for k in [k for k, v in FAILS.items() if v["until"] < now and not v["busy"]][:250]:
                FAILS.pop(k, None)
    return None


def pin_ok(pin):
    """Owner code check for the command line (no client)."""
    return bool(pin and CFG.get("pin_hash") and hmac.compare_digest(_hash(pin, CFG["pin_salt"]), CFG["pin_hash"]))


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


PUSH_LOG = deque(maxlen=40)


def _kind(title):
    for k, w in (("open", "נפתחה"), ("close", "נסגר"), ("stop", "סטופ"), ("report", "דוח"), ("update", "עדכון"),
                 ("emergency", "חירום"), ("error", "שגיאה"), ("fail", "נכשל")):
        if w in title:
            return k
    return "other"


def push(title, body, urgent=False, action=None, tag=None, extra=None):
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
    data = json.dumps({"title": title, "body": body, "silent": quiet, "action": action, "tag": tag, **(extra or {})}, ensure_ascii=False)
    dead, res = [], []
    for s in subs:
        try:
            webpush(subscription_info=s, data=data, vapid_private_key=str(VAPID_FILE),
                    vapid_claims={"sub": "mailto:bots@trading-system.local"}, ttl=24 * 3600)
            res.append("ok")
        except WebPushException as e:
            code = getattr(e.response, "status_code", None)
            res.append(str(code or "err"))
            if code in (404, 410):
                dead.append(s)
            else:
                log(f"push failed: {e}")
        except Exception as e:  # noqa: BLE001
            res.append("err")
            log(f"push failed: {e}")
    # delivery record without any content (published): when, which kind, and each device's answer
    PUSH_LOG.append({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": _kind(title), "silent": quiet, "to": res})
    if dead:
        with LOCK:
            CFG["subs"] = [s for s in CFG["subs"] if s not in dead]
        save()


# ---------------- per-combo risk ("ניהול בוטים" in the app) ----------------
RISK_FILE = LIVE / "risk.json"      # {bot: {"mult": {combo: m}, "rejected": {combo: sample}, "log": [...]}}; read by the bots
RISK_LOCK = threading.Lock()


def risk_doc():
    try:
        return json.loads(RISK_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


KINDS = ("mult", "dir", "tp", "sl", "hours", "vol")


def _risk_value(kind, v):
    """Checks one approved value; None = back to the bot's normal behaviour for that kind."""
    try:
        if kind == "mult":
            m = round(float(v), 2)
            if m == 0 or 0.5 <= m <= 1.5:
                return None if m == 1.0 else m
        elif v is None:
            return None
        elif kind == "dir" and int(v) in (1, -1):
            return int(v)
        elif kind == "tp" and 0.3 <= float(v) < 1:
            return round(float(v), 2)
        elif kind == "sl" and 0.5 <= float(v) < 1:
            return round(float(v), 2)
        elif kind == "hours" and len(v) == 2 and 0 <= int(v[0]) < int(v[1]) <= 24:
            return [int(v[0]), int(v[1])]
        elif kind == "vol" and isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("max_pct", "min_pct") and 0 < float(next(iter(v.values()))) < 100:
            return {k: float(x) for k, x in v.items()}
    except (TypeError, ValueError):
        pass
    raise ValueError("ערך לא תקין")


def risk_change(bot, combo, kind="mult", value=None, reject=None):
    """The user's decision on one recommendation. kind: mult (1 = the bot's risk, 0.5..1.5, 0 = no new trades),
    dir, tp, sl, hours, vol (None = undo). reject: the sample id whose recommendation of this kind is declined."""
    if not re.fullmatch(r"bot\d{1,2}", str(bot)):
        raise ValueError("אין בוט כזה")
    if kind not in KINDS and not (reject is not None and kind == "remove"):
        raise ValueError("סוג המלצה לא מוכר")
    f = LIVE / "bots" / f"{bot}.json"
    if not f.exists() or combo not in json.loads(f.read_text(encoding="utf-8"))["combos"]:
        raise ValueError("האסטרטגיה לא נמצאת בבוט הזה")
    val = None if reject is not None else _risk_value(kind, value)
    with RISK_LOCK:
        d = risk_doc()
        b = d.setdefault(bot, {})
        for k in KINDS + ("rejected",):
            b.setdefault(k, {})
        b.setdefault("log", [])
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if reject is not None:
            b["rejected"][f"{kind}:{combo}"] = str(reject)[:20]
            b["log"].append({"t": now, "combo": combo, "kind": kind, "rejected": str(reject)[:20]})
        else:
            old = b[kind].get(combo)
            if val is None:
                b[kind].pop(combo, None)
            else:
                b[kind][combo] = val
            for k in [k for k in b["rejected"] if k.endswith(":" + combo) and k.split(":")[0] in (kind, "remove" if kind == "mult" else kind)]:
                b["rejected"].pop(k)
            b["log"].append({"t": now, "combo": combo, "kind": kind, "from": old, "to": val})
        b["log"] = b["log"][-300:]
        tmp = RISK_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, RISK_FILE)
    log(f"risk {bot} {combo} {kind}: " + (f"recommendation declined ({reject})" if reject is not None else f"{old} -> {val}"))
    threading.Thread(target=push_manager_state, daemon=True).start()
    return d[bot]


def risk_public():
    return {b: {k: v.get(k, {}) for k in KINDS + ("rejected",)} for b, v in risk_doc().items()}


# ---------------- bots ----------------
def accounts():
    return CFG["accounts"]


def acc_by(name):
    return next((a for a in accounts() if a["name"] == name), None)


def running(name):
    p = PROCS.get(name)
    return p is not None and p.poll() is None


def sync_config(a):
    """An added account runs its own copy of a bot's settings (live/bots/<name>.json). Refresh it from the
    bot it follows (combos, risk, rules) at every start, keeping only the account's own fields, so a change
    to bot2.json / bot3.json reaches the account with the next update."""
    bot = a.get("bot")
    if not bot or bot == a["name"]:
        return
    src, dst = LIVE / "bots" / f"{bot}.json", LIVE / "bots" / f"{a['name']}.json"
    try:
        cfg = json.loads(src.read_text(encoding="utf-8"))
        old = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
        cfg.update({k: old[k] for k in ("name", "label", "bot_label", "account", "live_tunnel") if k in old})
        cfg["name"] = a["name"]
        if cfg != old:
            dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"{a['name']}: settings refreshed from {bot}.json")
    except Exception as e:  # noqa: BLE001
        log(f"{a['name']}: could not refresh settings from {bot}.json: {e}")


def start_bot(a):
    if running(a["name"]):
        return
    sync_config(a)
    env = os.environ.copy()
    env.update(MANAGED="1", BOT_BASE=str(a.get("bot") or a["name"]).split("_")[0], BOT_PORT=str(a["port"]), MANAGER_KEY=CFG["local_key"], BOT_NICKNAME=a.get("nickname") or "")
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


# ---------------- MetaTrader window: hidden from the screen and taskbar, still trading ----------------
def mt5_windows(terminal):
    """Main windows of the MetaTrader started from this terminal64.exe (Windows only)."""
    if os.name != "nt" or not terminal:
        return []
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    target = os.path.normcase(os.path.abspath(terminal))
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        if not cls.value.startswith("MetaQuotes::MetaTrader"):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)          # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            buf, n = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) and os.path.normcase(buf.value) == target:
                found.append(hwnd)
            kernel32.CloseHandle(h)
        return True
    user32.EnumWindows(cb, 0)
    return found


def mt5_show(a, show):
    import ctypes
    n = 0
    for hwnd in mt5_windows(a.get("terminal")):
        visible = bool(ctypes.windll.user32.IsWindowVisible(hwnd))
        if visible != show:
            ctypes.windll.user32.ShowWindow(hwnd, 9 if show else 0)      # SW_RESTORE / SW_HIDE
            n += 1
    return n


MT5_CMD_ALGO = 32851          # MetaTrader 5's own menu command that switches the Algo Trading button


def algo_keeper():
    """A bot whose MetaTrader has Algo Trading switched off cannot trade (orders come back 10027). The bot
    reports the button's state; when it is off, the manager presses it (MetaTrader's own menu command sent
    to that terminal's window), at most once every 10 minutes per account, and checks it really came on."""
    import ctypes
    last = {}
    while True:
        try:
            for a in list(accounts()):
                if not a.get("enabled") or a.get("archived") or not running(a["name"]):
                    continue
                port = bot_port(a["name"])
                try:
                    live = requests.get(f"http://127.0.0.1:{port}/live", timeout=5).json()
                except Exception:  # noqa: BLE001
                    continue
                if live.get("algo", True) or time.time() - last.get(a["name"], 0) < 600:
                    continue
                last[a["name"]] = time.time()
                wins = mt5_windows(a.get("terminal"))
                for hwnd in wins:
                    ctypes.windll.user32.PostMessageW(hwnd, 0x0111, MT5_CMD_ALGO, 0)      # WM_COMMAND
                time.sleep(8)
                try:
                    ok = requests.get(f"http://127.0.0.1:{port}/live", timeout=5).json().get("algo")
                except Exception:  # noqa: BLE001
                    ok = None
                log(f"{a['name']}: Algo Trading was off in its MetaTrader; pressed it ({len(wins)} windows) -> {'on' if ok else 'still off'}")
                label = a.get("label", a["name"])
                if ok:
                    push(f"{label}: המסחר האוטומטי הופעל", "הכפתור Algo Trading היה כבוי ב־MetaTrader של החשבון. המחשב הדליק אותו, והבוט ממשיך לסחור.")
                else:
                    push(f"{label}: המסחר האוטומטי כבוי", "לא הצלחתי להדליק את Algo Trading מרחוק. הצג את חלון MetaTrader מהאפליקציה ולחץ על Algo Trading.", urgent=True)
        except Exception:  # noqa: BLE001
            log(f"algo keeper error:\n{traceback.format_exc()}")
        time.sleep(60)


def mt5_hider():
    """Keep hidden MetaTraders hidden (MetaTrader opens a new window when a bot restarts it)."""
    while True:
        try:
            for a in list(accounts()):
                if a.get("mt5_hidden"):
                    mt5_show(a, False)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


def stop_bot(name):
    p = PROCS.pop(name, None)
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(15)
        except Exception:  # noqa: BLE001
            p.kill()
        log(f"stopped {name}")


# ---------------- scalping watch bot (no money: reads bot 1's candles, reports virtual trades) ----------------
WATCH_NAME = "watch_own"
MARKS_FILE = LIVE / "watch_marks.json"
MARKS_LOCK = threading.Lock()


def watch_source():
    """The account whose MetaTrader the watch bot reads candles from: bot 1 (FTMO's own symbol names), running."""
    for a in accounts():
        if a["name"] == "bot1" and a.get("enabled") and running(a["name"]):
            return a
    return None


def start_watch():
    if running(WATCH_NAME) or not CFG.get("watch_own", True):
        return
    src = watch_source()
    if not src:
        return
    env = os.environ.copy()
    env.update(MANAGED="1", SOURCE_PORT=str(src["port"]), MANAGER_KEY=CFG["local_key"])
    py = sys.executable.replace("pythonw.exe", "python.exe")
    PROCS[WATCH_NAME] = subprocess.Popen([py, str(LIVE / "watch_own.py")], cwd=str(ROOT), env=env,
                                         creationflags=0x08000000 if os.name == "nt" else 0, stdout=subprocess.DEVNULL,
                                         stderr=open(LIVE / f"{WATCH_NAME}_stderr.log", "a"))
    log("started the scalping watch bot (no money)")


# bot 7 (user, 10.11.2026): economic releases, virtual (live/news_bot.py), reads bot 1's candles like the watch bot
NEWS_NAME = "news_bot"


def start_news():
    if running(NEWS_NAME) or not CFG.get("news_bot", True):
        return
    src = watch_source()
    if not src:
        return
    env = os.environ.copy()
    env.update(MANAGED="1", SOURCE_PORT=str(src["port"]), MANAGER_KEY=CFG["local_key"])
    py = sys.executable.replace("pythonw.exe", "python.exe")
    PROCS[NEWS_NAME] = subprocess.Popen([py, str(LIVE / "news_bot.py")], cwd=str(ROOT), env=env,
                                        creationflags=0x08000000 if os.name == "nt" else 0, stdout=subprocess.DEVNULL,
                                        stderr=open(LIVE / f"{NEWS_NAME}_stderr.log", "a"))
    log("started bot 7, economic releases (virtual, no money)")


def watch_mark(d):
    k = str(d.get("id") or "")
    if not k or len(k) > 80 or not re.fullmatch(r"[A-Za-z0-9./|-]+", k):
        raise ValueError("עסקה לא מוכרת")
    take = d.get("take")
    if take not in ("yes", "no", None):
        raise ValueError("סימון לא מוכר")
    why = str(d.get("why") or "")[:500]
    with MARKS_LOCK:
        try:
            marks = json.loads(MARKS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            marks = {}
        if take is None and not why:
            marks.pop(k, None)
        else:
            marks[k] = {"take": take, "why": why, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = MARKS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(marks, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(MARKS_FILE)
    return marks.get(k)


# the agents' manager ("מנהל הבוטים", 11.11.2026): the user's decisions and notes on the agents' cards, kept here and
# copied to branch live (live/agent_decisions_state.json) where the nightly agents read them; and once a day a
# notification when the manager has new proposals (reports/agents/manager.json)
AGENT_DEC = LIVE / "agent_decisions.json"
AGENT_LOCK = threading.Lock()
AGENTS_URL = "https://raw.githubusercontent.com/talshahino/trading-system/main/reports/agents/manager.json"


def agent_decide(d):
    k = str(d.get("id") or "")
    if not re.fullmatch(r"[0-9a-f]{12}", k):
        raise ValueError("הצעה לא מוכרת")
    dec = d.get("decision")
    if dec not in ("approved", "rejected", None):
        raise ValueError("החלטה לא מוכרת")
    note = str(d.get("note") or "")[:1000]
    with AGENT_LOCK:
        try:
            all_ = json.loads(AGENT_DEC.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            all_ = {}
        cur = all_.get(k, {})
        if dec is not None or d.get("clear"):
            cur["decision"] = dec
        if "note" in d:
            cur["note"] = note or None
        cur.update(bot=str(d.get("bot") or cur.get("bot") or "")[:20], title=str(d.get("title") or cur.get("title") or "")[:200],
                   at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        all_[k] = cur
        AGENT_DEC.write_text(json.dumps(all_, ensure_ascii=False, indent=1), encoding="utf-8")
    agent_publish()
    log(f"agents' card {k}: {dec or 'note'}")
    return cur


# the manager decides in the user's place (user, 11.11.2026): its decisions come in reports/agents/decisions.json
# (agents/autodecide.py); the risk / removal ones are applied here, keeping the value before so "undo" restores it.
# The app marks decisions read (they leave the list; the archive keeps 30 days) and can undo any of them.
AGENT_META = LIVE / "agent_meta.json"
DECISIONS_URL = "https://raw.githubusercontent.com/talshahino/trading-system/main/reports/agents/decisions.json"


def agent_meta():
    try:
        m = json.loads(AGENT_META.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        m = {}
    m.setdefault("read", [])
    m.setdefault("undo", [])
    m.setdefault("applied", {})
    return m


def agent_meta_save(m):
    m["read"], m["undo"] = m["read"][-2000:], m["undo"][-2000:]
    tmp = AGENT_META.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, AGENT_META)


def agent_publish():
    """The user's card decisions, read marks and undos -> branch live (the manager agent reads them)."""
    def go():
        with AGENT_LOCK:
            try:
                cards = json.loads(AGENT_DEC.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                cards = {}
            m = agent_meta()
        REP.push_state("agent_decisions", {"cards": cards, "read": m["read"], "undo": m["undo"],
                                           "applied": sorted(m["applied"])})
    threading.Thread(target=go, daemon=True).start()


def agent_read(d):
    ids = [str(x) for x in (d.get("ids") or []) if re.fullmatch(r"[0-9a-f]{12}", str(x))][:500]
    if not ids:
        raise ValueError("אין החלטות לסמן")
    with AGENT_LOCK:
        m = agent_meta()
        if d.get("unread"):
            m["read"] = [x for x in m["read"] if x not in ids]
        else:
            m["read"] += [x for x in ids if x not in m["read"]]
        agent_meta_save(m)
    agent_publish()
    return len(ids)


def agent_undo(d):
    k = str(d.get("id") or "")
    if not re.fullmatch(r"[0-9a-f]{12}", k):
        raise ValueError("החלטה לא מוכרת")
    with AGENT_LOCK:
        m = agent_meta()
        if k not in m["undo"]:
            m["undo"].append(k)
        if k not in m["read"]:
            m["read"].append(k)
        a = m["applied"].pop(k, None)
        agent_meta_save(m)
    if a:                                    # a risk change already applied: put back the value before it
        risk_change(a["bot"], a["combo"], a["kind"], a.get("prev"))
        log(f"manager's decision {k} undone by the user: {a['bot']} {a['combo']} {a['kind']} back to {a.get('prev')}")
    else:
        log(f"manager's decision {k} undone by the user")
    agent_publish()
    return {"undone": k, "restored": bool(a)}


def auto_apply():
    """Every 10 minutes: apply the manager's risk / removal decisions that are not applied and not undone."""
    time.sleep(90)
    while True:
        try:
            j = requests.get(DECISIONS_URL, params={"t": int(time.time() // 300)}, timeout=20).json()
            for x in j.get("decisions", []):
                a = x.get("action")
                if x.get("type") != "risk" or x.get("decision") != "approved" or not a:
                    continue
                with AGENT_LOCK:
                    m = agent_meta()
                    if x["id"] in m["applied"] or x["id"] in m["undo"]:
                        continue
                try:
                    cur = (risk_doc().get(a["bot"], {}).get(a["kind"]) or {}).get(a["combo"])
                    risk_change(a["bot"], a["combo"], a["kind"], a["value"])
                except Exception as e:  # noqa: BLE001
                    log(f"manager's decision {x['id']} not applied: {e}")
                    cur = "failed"
                with AGENT_LOCK:
                    m = agent_meta()
                    m["applied"][x["id"]] = {"bot": a["bot"], "combo": a["combo"], "kind": a["kind"],
                                             "prev": None if cur == "failed" else cur, "failed": cur == "failed",
                                             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    agent_meta_save(m)
                log(f"manager's decision applied: {a['bot']} {a['combo']} {a['kind']} -> {a['value']}")
                agent_publish()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(600)


def agents_notify():
    """Every morning at 05:00 (the computer's clock, Israel): the manager's decisions of the past day, to read and undo
    in "מנהל הבוטים" (user, 11.11.2026)."""
    while True:
        try:
            lt = time.localtime()
            today = time.strftime("%Y-%m-%d", lt)
            if lt.tm_hour == 5 and CFG.get("agents_report_day") != today:
                j = requests.get(DECISIONS_URL, params={"t": int(time.time() // 300)}, timeout=20).json()
                since = CFG.get("agents_report_t") or "1970"
                ds = [x for x in j.get("decisions", []) if x.get("at", "") > since and not x.get("undone")]
                unread = [x for x in j.get("decisions", []) if not x.get("read") and not x.get("undone")]
                if ds or unread or [u for u in CFG.get("auto_updates") or [] if u.get("at", "") > since]:
                    kinds = {}
                    for x in ds:
                        k = {"remove": "הסרות", "mult": "שינויי סיכון", "improve": "שיפורים", "new": "אסטרטגיות חדשות",
                             "info": "בדיקות של התראות"}.get(x.get("kind"), "שינויים")
                        kinds[k] = kinds.get(k, 0) + 1
                    ups = [u for u in CFG.get("auto_updates") or [] if u.get("at", "") > since]
                    if ups:
                        kinds["עדכוני בוטים שהותקנו"] = len(ups)
                    body = (" · ".join(f"{v} {k}" for k, v in kinds.items()) or "אין החלטות חדשות") + \
                           f". {len(unread)} לא נקראו. אפשר לבטל כל החלטה במנהל הבוטים."
                    push(f"דוח המנהל: {len(ds)} החלטות", body, tag="agents")
                with LOCK:
                    CFG["agents_report_day"] = today
                    CFG["agents_report_t"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
                save()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(300)


def watchdog():
    while True:
        try:
            if not WATCH.get("installing"):
                start_watch()
                start_news()
            for a in list(accounts()):
                if a.get("enabled") and not a.get("preparing") and not running(a["name"]):
                    if a.get("started") and time.time() - a["started"] < 120:
                        continue                              # it just stopped right after starting: wait
                    p = PROCS.get(a["name"])
                    if p is not None and p.returncode == 4:   # an older copy holds the port: take over
                        free_port(a)
                    elif a.get("started"):
                        log(f"{a['name']} stopped by itself: starting it again")
                        if not WATCH.get("installing"):
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
    template = Path(CFG.get("mt5_template") or r"C:\MT5_bot1")      # never from the request
    if not (template / "terminal64.exe").exists():
        raise ValueError(f"לא מצאתי MetaTrader להעתקה ב־{template}")
    dest = Path(rf"C:\MT5_{login}")
    cfg = json.loads((LIVE / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
    cfg.update(name=name, label=f"{nickname} · {BOT_TEMPLATES[bot]}", bot_label=BOT_TEMPLATES[bot], account=int(login), live_tunnel=False)
    (LIVE / "bots" / f"{name}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    port = max([PORT] + [a["port"] for a in accounts()]) + 1
    a = {"name": name, "bot": bot, "nickname": nickname, "label": f"{nickname} · {BOT_TEMPLATES[bot]}", "login": int(login), "server": server,
         "terminal": str(dest / "terminal64.exe"), "port": port, "enabled": True, "password": encrypt(password),
         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "preparing": True}
    with LOCK:
        accounts().append(a)
    save()
    log(f"account added: {a['label']} ({server}); preparing its own MetaTrader in {dest}")
    threading.Thread(target=_prepare_and_start, args=(a, template, dest), daemon=True).start()
    return a


def _copy_mt5(src, dst):
    """Copy a MetaTrader folder for a new account (its own portable copy: each account needs its own
    MetaTrader). Files the running MetaTrader keeps locked, logs, history caches and the tester are skipped."""
    skip = {"logs", "bases", "tester"}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d.lower() not in skip]
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.lower().endswith(".log"):
                continue
            try:
                shutil.copy2(Path(root) / f, dst / rel / f)
            except OSError:
                pass


def _prepare_and_start(a, template, dest):
    try:
        log(f"copying MetaTrader to {dest} ...")
        _copy_mt5(template, dest)
        if not (dest / "terminal64.exe").exists():
            raise RuntimeError(f"the copy in {dest} has no terminal64.exe")
        a.pop("preparing", None)
        a.pop("prepare_error", None)
        save()
        log(f"{a['name']}: MetaTrader ready, starting the bot (MetaTrader opens and logs in to {a['login']})")
        start_bot(a)
        push(f"{a.get('label', a['name'])} נוסף", "MetaTrader נפתח במחשב ומתחבר לחשבון. החשבון יופיע בחשבונות הפעילים תוך דקה־שתיים.")
    except Exception as e:  # noqa: BLE001
        a["prepare_error"] = str(e)[:200]
        a["enabled"] = False
        a.pop("preparing", None)
        save()
        log(f"{a['name']}: preparing MetaTrader failed: {e}")
        push("הוספת החשבון נכשלה", f"{e}", urgent=True)
    push_manager_state()


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
                    "preparing": bool(a.get("preparing")), "prepare_error": a.get("prepare_error"),
                    "mt5_hidden": bool(a.get("mt5_hidden"))})
    if CFG.get("watch_own", True):
        # the scalping watch bot's virtual $100,000 account, listed like the demo accounts (user, 06.11.2026)
        out.append({"name": WATCH_NAME, "label": "סקלפינג (צפייה) · חשבון וירטואלי", "nickname": "סקלפינג (צפייה)", "bot": WATCH_NAME,
                    "login": None, "server": "וירטואלי · בלי כסף", "enabled": True, "running": running(WATCH_NAME) if with_run else None,
                    "archived": False, "archived_at": None, "hidden": False, "created": None, "broker": "וירטואלי",
                    "platform": "צפייה בלבד", "target_pct": 10, "preparing": False, "prepare_error": None, "mt5_hidden": True,
                    "virtual": True})
        # bot 6 (user, 10.11.2026): the virtual futures account, computed by the same watch bot (live/fut_account.py)
        out.append({"name": "watch_fut", "label": "בוט 6 · חוזים עתידיים (וירטואלי)", "nickname": "בוט 6 · חוזים", "bot": "watch_fut",
                    "login": None, "server": "וירטואלי · FTMO Futures · בלי כסף", "enabled": True, "running": running(WATCH_NAME) if with_run else None,
                    "archived": False, "archived_at": None, "hidden": False, "created": None, "broker": "וירטואלי",
                    "platform": "צפייה בלבד", "target_pct": 6, "preparing": False, "prepare_error": None, "mt5_hidden": True,
                    "virtual": True})
    if CFG.get("news_bot", True):
        out.append({"name": NEWS_NAME, "label": "בוט 7 · פרסומים כלכליים (וירטואלי)", "nickname": "בוט 7 · פרסומים", "bot": NEWS_NAME,
                    "login": None, "server": "וירטואלי · בלי כסף", "enabled": True, "running": running(NEWS_NAME) if with_run else None,
                    "archived": False, "archived_at": None, "hidden": False, "created": None, "broker": "וירטואלי",
                    "platform": "צפייה בלבד", "target_pct": 10, "preparing": False, "prepare_error": None, "mt5_hidden": True,
                    "virtual": True})
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


AUTO_SETTLE = 600          # a new version installs by itself once it has been the latest for 10 minutes


def auto_update_on():
    """User, 11.11.2026: the manager installs the bots' updates by itself too. The user can go back to the previous
    version from the app, which also pauses the automatic installs until he turns them on again."""
    return CFG.get("auto_update", True) and not CFG.get("auto_update_paused")


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
                    UPDATE.update(sha=sha, title=title, seen=time.time())
                    log(f"update available: {sha[:7]} {title}")
                    if not auto_update_on():
                        # every new waiting version is announced (same tag: it replaces the earlier notice on the
                        # phone), and an approval counts only for the exact version it was given for
                        push("יש עדכון לבוטים", title, action="approve_update", tag="update", extra={"sha": sha[:7]})
            if (UPDATE["sha"] and UPDATE["sha"] != UPD.local() and auto_update_on()
                    and time.time() - UPDATE.get("seen", 0) >= AUTO_SETTLE and not INSTALL_LOCK.locked()
                    and UPDATE.get("tried") != UPDATE["sha"]):            # a failed version is not retried by itself
                UPDATE["tried"] = UPDATE["sha"]
                with LOCK:
                    CFG["prev_version"] = UPD.local()
                    CFG.setdefault("auto_updates", []).append({"sha": UPDATE["sha"], "prev": UPD.local(), "title": UPDATE["title"],
                                                               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    CFG["auto_updates"] = CFG["auto_updates"][-60:]
                save()
                log(f"installing update {UPDATE['sha'][:7]} by itself (the manager decides the updates too)")
                threading.Thread(target=install_update, daemon=True).start()
            UPDATE["checked"] = time.time()
        except Exception as e:  # noqa: BLE001
            log(f"update check failed: {e}")
        time.sleep(300)


INSTALL_LOCK = threading.Lock()


def install_update():
    # one install at a time: a second approval (notification button + app) must not unpack into the same
    # staging folder while the first one is checking it
    if not INSTALL_LOCK.acquire(blocking=False):
        log("update approval ignored: an install is already running")
        return
    try:
        _install_update()
    finally:
        INSTALL_LOCK.release()


def update_rollback():
    """Back to the version before the last install; automatic installs pause until the user turns them on again."""
    prev = CFG.get("prev_version")
    if not prev:
        raise ValueError("אין גרסה קודמת לחזור אליה")
    if INSTALL_LOCK.locked():
        raise ValueError("התקנה כבר רצה. נסה שוב בעוד דקה.")
    with LOCK:
        CFG["auto_update_paused"] = True
        CFG["rolled_back_from"] = UPD.local()
    save()
    UPDATE.update(sha=prev, title="חזרה לגרסה הקודמת", seen=time.time())
    log(f"rolling back to {prev[:7]} (user); automatic updates paused")
    threading.Thread(target=install_update, daemon=True).start()
    return prev[:7]


def update_auto(on):
    with LOCK:
        CFG["auto_update_paused"] = not bool(on)
    save()
    log(f"automatic updates {'on' if on else 'paused'} (user)")
    if not on and UPDATE["sha"] and UPDATE["sha"] != UPD.local():
        push("יש עדכון לבוטים", UPDATE["title"] or "", action="approve_update", tag="update", extra={"sha": UPDATE["sha"][:7]})
    return auto_update_on()


def _install_update():
    sha = UPDATE["sha"]
    if not sha:
        raise ValueError("אין עדכון שמחכה")
    log(f"installing update {sha[:7]}: stopping the bots")
    WATCH["installing"] = True                     # the watchdog must not report the planned stops
    # no message here: the user gets one "update installed" when everything runs again (or "update failed")
    names = [a["name"] for a in accounts() if running(a["name"])]
    stop_bot(WATCH_NAME)
    stop_bot(NEWS_NAME)
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
        WATCH["installing"] = False
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
                                    "update": {"sha": (UPDATE["sha"] or "")[:7], "title": UPDATE["title"]} if UPDATE["sha"] and not auto_update_on() and UPDATE["sha"] != UPD.local() else None,
                                    "auto_update": {"on": auto_update_on(), "prev": (CFG.get("prev_version") or "")[:7],
                                                    "pending": (UPDATE["sha"] or "")[:7] if UPDATE["sha"] and UPDATE["sha"] != UPD.local() else None,
                                                    "last": (CFG.get("auto_updates") or [{}])[-1]},
                                    "pin_set": bool(CFG.get("pin_hash")), "viewer_set": bool(CFG.get("viewer_hash")), "push_devices": len(CFG.get("subs", [])),
                                    "watchdog_push": WATCH["ok"],
                                    "risk": risk_public()})
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
        if path == "/subscribe" and not (self.headers.get("X-Pin") or d.get("pin")):
            # notifications only (view): a device without the secret code may subscribe; every action still needs it
            sub = d.get("subscription")
            if not sub or "endpoint" not in sub or len(json.dumps(sub)) > 2000 or not push_endpoint_ok(sub["endpoint"]):
                return self._send(400, {"error": "no subscription"})
            with LOCK:
                if sub not in CFG["subs"]:
                    if len(CFG["subs"]) >= 20:
                        return self._send(429, {"error": "יותר מדי מכשירים מקבלים התראות"})
                    CFG["subs"].append(sub)
            save()
            threading.Thread(target=push_manager_state, daemon=True).start()
            return self._send(200, {"ok": True})
        pin = self.headers.get("X-Pin") or d.get("pin")
        client = self.headers.get("CF-Connecting-IP") or self.client_address[0]
        role = check_codes(pin, client)
        if role == "locked":
            return self._send(429, {"error": "נעול ל־15 דקות אחרי 5 קודים שגויים"})
        if role is None:
            return self._send(401, {"error": "קוד שגוי"})
        if path == "/url_key":                                # both codes: the key of the published address
            with LOCK:
                first = not CFG["paired"].get(role)
                CFG["paired"][role] = True
            save()
            if first:
                log(f"{role} device got the address key; the code-derived copy is no longer published")
                threading.Thread(target=push_manager_state, daemon=True).start()
            return self._send(200, {"ok": True, "key": CFG["url_key"]})
        if role == "viewer":
            if path == "/auth":
                return self._send(200, {"ok": True, "role": "viewer"})
            if path != "/subscribe":
                return self._send(403, {"error": "קוד צפייה: אפשר לראות בלבד, בלי פעולות"})
        try:
            if path == "/auth":
                return self._send(200, {"ok": True, "role": "owner"})
            if path == "/pin_change":
                new = str(d.get("new") or "").strip()
                if len(new) < 8:
                    raise ValueError("הקוד החדש צריך לפחות 8 תווים")
                if not CFG["paired"].get("owner"):
                    raise ValueError("סגור ופתח את האפליקציה פעם אחת ונסה שוב")
                if CFG.get("viewer_hash") and hmac.compare_digest(_hash(new, CFG["viewer_salt"]), CFG["viewer_hash"]):
                    raise ValueError("הקוד החדש חייב להיות שונה מקוד הצפייה")
                set_pin(new)
                log("owner code changed")
                push("הקוד הסודי הוחלף", "הקוד הסודי של המערכת הוחלף עכשיו.", urgent=True)
                threading.Thread(target=push_manager_state, daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/viewer_pin":
                v = str(d.get("viewer") or "").strip()
                if len(v) < 6:
                    raise ValueError("קוד הצפייה צריך לפחות 6 תווים")
                if hmac.compare_digest(hashlib.pbkdf2_hmac("sha256", v.encode(), bytes.fromhex(CFG["pin_salt"]), 200_000).hex(), CFG["pin_hash"]):
                    raise ValueError("קוד הצפייה חייב להיות שונה מהקוד שלך")
                set_viewer_pin(v)
                CFG["paired"]["viewer"] = False          # the new code's device fetches the key on its first visit
                save()
                log("read-only code set")
                threading.Thread(target=push_manager_state, daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/viewer_pin_del":
                CFG.pop("viewer_hash", None)
                CFG.pop("viewer_salt", None)
                save()
                threading.Thread(target=push_manager_state, daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/subscribe":
                sub = d.get("subscription")
                if not sub or "endpoint" not in sub or not push_endpoint_ok(sub["endpoint"]):
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
            if path.startswith("/accounts/") and path.endswith("/close"):        # close one trade / all trades now
                a = acc_by(path.split("/")[2])
                if not a or not running(a["name"]):
                    raise ValueError("הבוט של החשבון הזה לא רץ כרגע")
                body = {"all": True} if d.get("all") else {"ticket": str(d.get("ticket") or "")}
                r = requests.post(f"http://127.0.0.1:{a['port']}/close", json=body, headers={"X-Local": CFG["local_key"]}, timeout=30)
                j = r.json() if r.status_code == 200 else {"error": f"הבוט ענה {r.status_code}"}
                log(f"{a['name']}: close from the app {body} -> {j}")
                if j.get("error") and not j.get("closed"):
                    raise ValueError(j["error"])
                return self._send(200, j)
            if path.startswith("/accounts/") and path.endswith("/mt5"):
                a = acc_by(path.split("/")[2])
                if not a:
                    raise ValueError("אין חשבון כזה")
                a["mt5_hidden"] = bool(d.get("hidden"))
                save()
                n = mt5_show(a, not a["mt5_hidden"])
                log(f"{a['name']}: MetaTrader {'hidden' if a['mt5_hidden'] else 'shown'} ({n} windows)")
                return self._send(200, {"ok": True, "windows": n})
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
                threading.Thread(target=push_manager_state, daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/logs":                          # behind the secret code only; never published
                which = str(d.get("which") or "manager")
                if which not in ("manager", WATCH_NAME, NEWS_NAME) and not acc_by(which):
                    raise ValueError("אין לוג כזה")
                f = LIVE / f"{which}.log"
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()[-400:] if f.exists() else []
                lines = [l for l in lines if "password" not in l.lower()]
                return self._send(200, {"ok": True, "lines": lines})
            if path == "/watch/mark":
                return self._send(200, {"ok": True, "mark": watch_mark(d)})
            if path == "/risk/set":
                kind = str(d.get("kind") or "mult")
                risk_change(d.get("bot"), str(d.get("combo") or ""), kind, d.get("value", d.get("mult")))
                return self._send(200, {"ok": True, "risk": risk_public()})
            if path == "/agent/decide":
                return self._send(200, {"ok": True, "card": agent_decide(d)})
            if path == "/update/rollback":
                return self._send(200, {"ok": True, "to": update_rollback()})
            if path == "/update/auto":
                return self._send(200, {"ok": True, "on": update_auto(d.get("on"))})
            if path == "/agent/read":
                return self._send(200, {"ok": True, "n": agent_read(d)})
            if path == "/agent/undo":
                return self._send(200, {"ok": True, **agent_undo(d)})
            if path == "/risk/reject":
                risk_change(d.get("bot"), str(d.get("combo") or ""), str(d.get("kind") or "mult"), reject=d.get("sample") or "")
                return self._send(200, {"ok": True, "risk": risk_public()})
            if path == "/update/approve":
                want = str(d.get("sha") or "")
                if not UPDATE["sha"]:
                    raise ValueError("אין עדכון שמחכה")
                if len(want) < 7 or not UPDATE["sha"].startswith(want):
                    raise ValueError("בינתיים הגיעה גרסה חדשה יותר. פתח את העדכון שוב ואשר אותה.")
                if INSTALL_LOCK.locked():
                    return self._send(200, {"ok": True, "already": True})
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


def push_endpoint_ok(ep):
    from urllib.parse import urlparse
    u = urlparse(str(ep))
    h = (u.hostname or "").lower()
    return u.scheme == "https" and any(h == x or (x.startswith(".") and h.endswith(x)) for x in PUSH_HOSTS)


def _seal(key, url, **extra):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    iv = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(iv, url.encode(), None)
    return {**extra, "iv": base64.b64encode(iv).decode(), "ct": base64.b64encode(ct).decode()}


def seal_url(url):
    """The computer's address, published only encrypted (AES-256-GCM, fresh nonce each time) with a random
    32-byte key (url_key) that each phone gets once from the manager after entering its code. A key derived
    from a short code could be found offline by trying every code, so the code-derived copy is published
    only until the holder of that code has fetched url_key (the transition after this change)."""
    if not url:
        return []
    out = [_seal(bytes.fromhex(CFG["url_key"]), url, k=1)]
    for role, h, salt in (("owner", CFG.get("pin_hash"), CFG.get("pin_salt")),
                          ("viewer", CFG.get("viewer_hash"), CFG.get("viewer_salt"))):
        # the read-only code opens the address on every device, always (user, 06.11.2026: a family member uses it
        # on a computer and a phone); it gives viewing only - every action needs the owner's code
        if h and salt and (role == "viewer" or not CFG["paired"].get(role)):
            out.append(_seal(hashlib.sha256(bytes.fromhex(h) + b"url").digest(), url, s=salt))
    return out


def push_manager_state():
    """Tell the app where the manager is (public GitHub file). It also lists the app's notification
    subscriptions for GitHub's watchdog: without the private signing key (kept on this computer and,
    encrypted, as a GitHub secret) nobody can send anything to them."""
    state = {"url_enc": seal_url(TUN.URL["url"]), "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "accounts": public_accounts(with_run=False),
             "subs": CFG.get("subs", []), "push_log": list(PUSH_LOG),
             "risk": risk_public(),
             "auto_update": {"on": auto_update_on(), "prev": CFG.get("prev_version"), "rolled_back_from": CFG.get("rolled_back_from"),
                             "installs": (CFG.get("auto_updates") or [])[-10:]},
             "tunnel_error": None if TUN.URL["url"] else (TUN.URL.get("err") or "connecting")}
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
        last = (CFG.get("auto_updates") or [{}])[-1]
        if last.get("sha") == UPD.local():
            push("המנהל התקין עדכון", f"{last.get('title') or ''} · גרסה {(UPD.local() or '?')[:7]}. אפשר לחזור לגרסה הקודמת בלשונית ניהול.")
        else:
            push("העדכון הותקן", f"הכל רץ בגרסה {(UPD.local() or '?')[:7]}.")
    for a in accounts():
        if a.get("preparing"):                    # the manager stopped while copying MetaTrader: finish it
            threading.Thread(target=_prepare_and_start, args=(a, Path(CFG.get("mt5_template") or r"C:\MT5_bot1"),
                                                            Path(a["terminal"]).parent), daemon=True).start()
        elif a.get("enabled"):
            free_port(a)
            start_bot(a)
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=agents_notify, daemon=True).start()
    threading.Thread(target=auto_apply, daemon=True).start()
    threading.Thread(target=mt5_hider, daemon=True).start()
    threading.Thread(target=algo_keeper, daemon=True).start()
    threading.Thread(target=update_check, daemon=True).start()
    threading.Thread(target=watchdog_secret, daemon=True).start()
    threading.Thread(target=heartbeat, daemon=True).start()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
