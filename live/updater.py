"""Self-update on the user's approval.

Every 5 minutes the bot compares the newest code on GitHub (branch main) with the version it runs
(live/version.txt). When there is a new one it sends a phone notification with an "אשר עדכון"
button. The button (ntfy "http" action) posts "update <sha>" to a private command topic; the bot
reads that topic, downloads exactly that version, installs it over C:\\trading-system, installs any
new Python packages and restarts itself. Open positions stay at the broker with their stop and
target, and the restarted bot manages them again.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

from live import reporter as REP

ROOT = Path(__file__).resolve().parent.parent
REPO = REP.REPO
VERSION = ROOT / "live" / "version.txt"
CMD_TOPIC = REP._CFG["topic"] + "-cmd"
_S = {"check": 0, "poll": 0, "since": int(time.time())}


def _headers():
    tok = REP.token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def remote():
    r = requests.get(f"https://api.github.com/repos/{REPO}/commits/main", headers=_headers(), timeout=20)
    r.raise_for_status()
    j = r.json()
    return j["sha"], j["commit"]["message"].splitlines()[0][:140]


BOT_CODE = ("live/", "agents/", "skills/", "config.py", "requirements.txt", "data/ftmo_symbols/")
# research-only strategy families: no bot trades them, so a change there is not a bot update
RESEARCH = tuple(f"agents/{x}_strategies.py" for x in
                 ("seasonal", "intermarket", "momentum", "event", "gap_breakout", "quantpedia", "book", "scout"))


def touches_bot(a, b):
    """Did anything the running bot uses change between versions a and b? (GitHub Actions commit
    reports and data all the time; those are not offered as updates.)"""
    try:
        r = requests.get(f"https://api.github.com/repos/{REPO}/compare/{a}...{b}", headers=_headers(), timeout=20)
        r.raise_for_status()
        files = [f["filename"] for f in r.json().get("files", [])]
        return any(f.startswith(BOT_CODE) and not f.endswith("_state.json") and f not in RESEARCH
                   and not f.startswith("data/agents/") for f in files) or len(files) >= 300
    except Exception:  # noqa: BLE001
        return True


def local():
    try:
        return VERSION.read_text().strip() or None
    except FileNotFoundError:
        return None


def startup(log, bot):
    """First run: the code was installed by update.bat just now, so it is the newest version."""
    if local() is None:
        try:
            VERSION.write_text(remote()[0])
        except Exception as e:  # noqa: BLE001
            log(f"version check not available: {e}")
    log(f"code version {(local() or '?')[:7]}; updates are offered on the phone and installed after approval")


def tick(log, bot, state, save_state, open_count, restart):
    now = time.time()
    if now - _S["check"] >= 300:
        _S["check"] = now
        try:
            sha, title = remote()
            if local() and sha != local() and state.get("offered_sha") != sha and not touches_bot(local(), sha):
                VERSION.write_text(sha)            # only reports/data changed: nothing to install
            elif local() and sha != local() and state.get("offered_sha") != sha and REP._CFG.get("ntfy") is not False:
                state["offered_sha"] = sha
                save_state()
                REP.notify_action(f"{bot}: יש עדכון", f"{title}\nעסקאות פתוחות: {open_count()} (נשארות עם הסטופ והיעד, הבוט ממשיך לנהל אותן אחרי העדכון)",
                                  "אשר עדכון", f"https://ntfy.sh/{CMD_TOPIC}", f"update {sha}")
                log(f"update available: {sha[:7]} {title}")
        except Exception as e:  # noqa: BLE001
            log(f"update check failed: {e}")
    if now - _S["poll"] >= 15 and REP._CFG.get("ntfy") is not False:
        _S["poll"] = now
        try:
            r = requests.get(f"https://ntfy.sh/{CMD_TOPIC}/json", params={"poll": "1", "since": str(_S["since"])}, timeout=15)
            for line in r.text.splitlines():
                m = json.loads(line)
                _S["since"] = max(_S["since"], int(m.get("time", 0)) + 1)
                msg = str(m.get("message", "")).strip()
                if msg.startswith("update ") and msg.split()[1] == state.get("offered_sha"):
                    install(msg.split()[1], log, bot, state, save_state, restart)
        except Exception as e:  # noqa: BLE001
            log(f"update command check failed: {e}")


MANIFEST = ROOT / "live" / "installed_files.txt"
PROTECT = ("live/manager.json", "live/vapid_private.pem", "live/github_token.txt", "live/version.txt",
           "live/installed_files.txt")


def _py():
    return sys.executable.replace("pythonw", "python")


def apply(sha):
    """Install version `sha` safely:
    1. download it and unpack it into a staging folder next to the program (nothing changed yet)
    2. check it there: every Python file compiles and the strategy engine imports
    3. install its Python packages (a failure stops here, the running version stays as it was)
    4. back up every file that will be replaced or removed, then swap the new files in
    5. any error while swapping: every file is put back from the backup
    6. files removed upstream since the last install are removed here too (from the install list)
    The version number is written only at the very end."""
    import py_compile
    import shutil
    r = requests.get(f"https://codeload.github.com/{REPO}/zip/{sha}", timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    top = z.namelist()[0].split("/")[0]
    stage = ROOT / "_update" / f"stage_{sha[:7]}"
    backup = ROOT / "_update" / f"backup_{sha[:7]}"
    for d in (stage, backup):
        shutil.rmtree(d, ignore_errors=True)
    files = {}
    for info in z.infolist():
        rel = info.filename[len(top) + 1:]
        if rel and not info.is_dir() and rel not in PROTECT:
            files[rel] = z.read(info)
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_bytes(files[rel])
    # 2. check before touching anything
    for rel in files:
        if rel.endswith(".py") and rel.startswith(("live/", "agents/")):
            py_compile.compile(str(stage / rel), doraise=True)
    chk = subprocess.run([_py(), "-c", "from agents import rules_agent; print(len(rules_agent.STRATEGIES))"],
                         cwd=str(stage), capture_output=True, text=True, timeout=300)
    if chk.returncode != 0:
        raise RuntimeError("the new version does not start: " + ((chk.stderr or chk.stdout).strip().splitlines() or ["?"])[-1][-200:])
    # 3. packages first: a failed install leaves the running version untouched
    pip = subprocess.run([_py(), "-m", "pip", "install", "-q", "-r", str(stage / "live" / "requirements.txt")],
                         capture_output=True, text=True, timeout=900)
    if pip.returncode != 0:
        raise RuntimeError("installing Python packages failed: " + ((pip.stderr or pip.stdout).strip().splitlines() or ["?"])[-1][-200:])
    old = set(MANIFEST.read_text(encoding="utf-8").split("\n")) - {""} if MANIFEST.exists() else set()
    gone = [rel for rel in old - set(files) if rel not in PROTECT and (ROOT / rel).is_file()]
    changed = [rel for rel, data in files.items() if not ((ROOT / rel).exists() and (ROOT / rel).read_bytes() == data)]
    created = []
    try:
        for rel in changed + gone:                    # 4. backup
            if (ROOT / rel).exists():
                (backup / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, backup / rel)
        for rel in changed:
            dest = ROOT / rel
            if not dest.exists():
                created.append(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".new")
            tmp.write_bytes(files[rel])
            os.replace(tmp, dest)
        for rel in gone:
            (ROOT / rel).unlink()
    except Exception:
        for rel in changed + gone:                    # 5. put everything back
            if (backup / rel).exists():
                (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup / rel, ROOT / rel)
        for rel in created:
            try:
                (ROOT / rel).unlink()
            except OSError:
                pass
        raise
    MANIFEST.write_text("\n".join(sorted(files)), encoding="utf-8")
    VERSION.write_text(sha)
    shutil.rmtree(stage, ignore_errors=True)
    for d in (ROOT / "_update").glob("backup_*"):         # keep only this install's backup (the previous version)
        if d != backup:
            shutil.rmtree(d, ignore_errors=True)


def install(sha, log, bot, state, save_state, restart):
    log(f"installing update {sha[:7]}...")
    REP.notify(f"{bot}: מתעדכן", "מוריד את הגרסה החדשה ומפעיל מחדש. זה לוקח כחצי דקה.", tags=["arrows_counterclockwise"])
    try:
        apply(sha)
        state["updated_to"] = sha
        save_state()
    except Exception as e:  # noqa: BLE001
        log(f"update failed: {e}")
        REP.notify(f"{bot}: העדכון נכשל", f"{e}\nהבוט ממשיך לעבוד בגרסה הקודמת.", priority=4, tags=["warning"])
        state.pop("offered_sha", None)
        save_state()
        return
    log("update installed; restarting")
    restart()


def restart_self(log):
    """Started from botN.bat (loop): exit with code 3 and the .bat starts the bot again.
    Started hidden (pythonw): start a new copy of the bot and exit."""
    if os.environ.get("BOT_LOOP") == "1":
        os._exit(3)
    subprocess.Popen([sys.executable] + sys.argv, cwd=str(ROOT), creationflags=0x00000008 | 0x00000200)  # detached
    os._exit(0)


def announce(log, bot, state, save_state):
    if state.pop("updated_to", None):
        save_state()
        REP.notify(f"{bot}: עודכן בהצלחה", f"רץ בגרסה {(local() or '?')[:7]}.", tags=["white_check_mark"])
        log("update finished: running the new version")
