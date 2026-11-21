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


def changed_files(a, b):
    """The files changed between versions a and b (GitHub's compare), or None when it cannot be told."""
    try:
        r = requests.get(f"https://api.github.com/repos/{REPO}/compare/{a}...{b}", headers=_headers(), timeout=20)
        r.raise_for_status()
        files = [f["filename"] for f in r.json().get("files", [])]
        return None if len(files) >= 300 else files
    except Exception:  # noqa: BLE001
        return None


def counts(f, used):
    """Does file f matter to a running bot? (used = live/loaded.py's record, None = not known)"""
    if f.endswith("_state.json") or f.startswith(("live/loaded_", "live/cache_watch/")):
        return False
    if used is not None:
        return f in used or f.startswith(("live/", "config.py", "requirements.txt", "data/ftmo_symbols/"))
    return f.startswith(BOT_CODE) and f not in RESEARCH and not f.startswith("data/agents/")


def is_setting(f):
    """A bot's settings file (live/bots/<bot>.json): put in place without stopping the bots (19.11.2026)."""
    return f.startswith("live/bots/") and f.endswith(".json") and "/" not in f[len("live/bots/"):]


def apply_settings(sha, files):
    """Only the settings files of version sha, each checked as JSON and written whole-or-nothing; the version number
    moves to sha (nothing else the bots use changed). Raises when one cannot be fetched or read - the full install
    then runs as before."""
    got = {}
    for f in files:
        r = requests.get(f"https://api.github.com/repos/{REPO}/contents/{f}", params={"ref": sha},
                         headers={**_headers(), "Accept": "application/vnd.github.raw"}, timeout=30)
        r.raise_for_status()
        json.loads(r.content.decode("utf-8"))
        got[f] = r.content
    for f, data in got.items():
        p = ROOT / f
        tmp = p.with_suffix(p.suffix + ".new")
        tmp.write_bytes(data)
        os.replace(tmp, p)
    VERSION.write_text(sha)
    return list(got)


def touches_bot(a, b):
    """Did anything the running bot uses change between versions a and b? (GitHub Actions commit
    reports and data all the time; those are not offered as updates.)"""
    try:
        r = requests.get(f"https://api.github.com/repos/{REPO}/compare/{a}...{b}", headers=_headers(), timeout=20)
        r.raise_for_status()
        files = [f["filename"] for f in r.json().get("files", [])]
        if len(files) >= 300:
            return True
        from live import loaded as LD
        used = LD.used()          # only what a running bot really loaded (live/loaded.py), its folder and settings
        return any(counts(f, used) for f in files)
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


def _not_installed(rel):
    """The agents' per-strategy result files (reports/agents/<bot>/exp/): never needed by the bots."""
    p = rel.split("/")
    return len(p) >= 5 and p[0] == "reports" and p[1] == "agents" and p[3] == "exp"


def _py():
    return sys.executable.replace("pythonw", "python")


def _blob_sha(data):
    """git's id of a file's content (the same as the tree's entry for an unchanged file)."""
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


FAST_MAX_FILES, FAST_MAX_BYTES = 600, 300 * 1024 * 1024


def _apply_fast(sha, stage):
    """Step 1 of apply() without the whole repository's zip (21.11.2026: the zip grew to ~3 GB with the agents'
    reports - an install stopped every bot for ~27 minutes): the version's file list (one call, git tree) and only the
    files whose content differs from what is here, each checked against its git id. Returns (files, changed) like the
    zip path - files = every installed path with its size, changed = the ones written to `stage`. Raises when anything
    is unclear (a truncated list, too many / too big changes, a download that does not match) - apply() then falls
    back to the zip, exactly as before."""
    import base64
    r = requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}", params={"recursive": "1"},
                     headers=_headers(), timeout=60)
    r.raise_for_status()
    t = r.json()
    if t.get("truncated"):
        raise RuntimeError("file list truncated")
    files, todo = {}, []
    for e in t.get("tree") or []:
        if e.get("type") != "blob" or e.get("mode") == "120000":
            continue
        rel = e["path"]
        if rel in PROTECT or _not_installed(rel):
            continue
        files[rel] = int(e.get("size") or 0)
        p = ROOT / rel
        try:
            same = p.is_file() and p.stat().st_size == files[rel] and _blob_sha(p.read_bytes()) == e["sha"]
        except OSError:
            same = False
        if not same:
            todo.append((rel, e["sha"]))
    if len(todo) > FAST_MAX_FILES or sum(files[x] for x, _ in todo) > FAST_MAX_BYTES:
        raise RuntimeError(f"{len(todo)} files changed - the full download is used")
    for rel, bsha in todo:
        g = requests.get(f"https://api.github.com/repos/{REPO}/git/blobs/{bsha}", headers=_headers(), timeout=120)
        g.raise_for_status()
        j = g.json()
        data = base64.b64decode(j["content"]) if j.get("encoding") == "base64" else str(j.get("content", "")).encode("utf-8")
        if _blob_sha(data) != bsha:
            raise RuntimeError(f"{rel}: download does not match")
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        (stage / rel).write_bytes(data)
    return files, [x for x, _ in todo]


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
    stage = ROOT / "_update" / f"stage_{sha[:7]}"
    backup = ROOT / "_update" / f"backup_{sha[:7]}"
    for d in (stage, backup):
        shutil.rmtree(d, ignore_errors=True)
    # the zip goes to disk, not memory, and the agents' result files are not installed into the bots' folder (manager,
    # 14.11.2026: the GitHub scout added 83 thousand of them, 2.9 GB - the zip of main grew to 1.35 GB, read whole into
    # the bots' own process at every install; the bots never read them, the agents have their own copy)
    fast = None
    try:
        stage.mkdir(parents=True, exist_ok=True)
        fast = _apply_fast(sha, stage)
    except Exception as e:  # noqa: BLE001 - the full download below, as before
        print(f"updater: the quick install is not used ({str(e)[-200:]}); downloading the whole version", flush=True)
        shutil.rmtree(stage, ignore_errors=True)
        fast = None
    if fast is not None:
        _install_staged(sha, stage, backup, *fast)
        return
    zpath = ROOT / "_update" / f"dl_{sha[:7]}.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"https://codeload.github.com/{REPO}/zip/{sha}", timeout=120, stream=True) as r:
        r.raise_for_status()
        with open(zpath, "wb") as out:
            for chunk in r.iter_content(1 << 20):
                out.write(chunk)
    z = zipfile.ZipFile(zpath)
    top = z.namelist()[0].split("/")[0]
    files = {}
    for info in z.infolist():
        rel = info.filename[len(top) + 1:]
        if rel and not info.is_dir() and rel not in PROTECT and not _not_installed(rel):
            files[rel] = info.file_size                      # the content stays in the staging folder
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_bytes(z.read(info))
    z.close()
    try:
        zpath.unlink()
    except OSError:
        pass
    changed = [rel for rel, size in files.items() if not ((ROOT / rel).is_file() and (ROOT / rel).stat().st_size == size
                                                          and (ROOT / rel).read_bytes() == (stage / rel).read_bytes())]
    _install_staged(sha, stage, backup, files, changed, full=True)


def _install_staged(sha, stage, backup, files, changed, full=False):
    """Steps 2-6 of apply(): the version is in `stage` - whole (full, the zip) or only its changed files (the quick
    path; the unchanged ones are the installed ones)."""
    import py_compile
    import shutil
    # 2. check before touching anything
    for rel in (files if full else changed):
        if rel.endswith(".py") and rel.startswith(("live/", "agents/")):
            py_compile.compile(str(stage / rel), doraise=True)
    if full:
        code = "from agents import rules_agent; print(len(rules_agent.STRATEGIES))"
    else:                                           # the changed files first, the installed ones for the rest
        code = ("import sys; sys.path[:0] = [%r, %r]; import agents; agents.__path__.insert(0, %r); "
                "from agents import rules_agent; print(len(rules_agent.STRATEGIES))") % (str(stage), str(ROOT), str(stage / "agents"))
    chk = subprocess.run([_py(), "-c", code], cwd=str(stage if full else ROOT), capture_output=True, text=True, timeout=300)
    if chk.returncode != 0:
        raise RuntimeError("the new version does not start: " + ((chk.stderr or chk.stdout).strip().splitlines() or ["?"])[-1][-200:])
    # 3. packages first: a failed install leaves the running version untouched
    if full or "live/requirements.txt" in changed:
        req = stage / "live" / "requirements.txt"
        pip = subprocess.run([_py(), "-m", "pip", "install", "-q", "-r", str(req)], capture_output=True, text=True, timeout=900)
        if pip.returncode != 0:
            raise RuntimeError("installing Python packages failed: " + ((pip.stderr or pip.stdout).strip().splitlines() or ["?"])[-1][-200:])
    old = set(MANIFEST.read_text(encoding="utf-8").split("\n")) - {""} if MANIFEST.exists() else set()
    gone = [rel for rel in old - set(files) if rel not in PROTECT and not _not_installed(rel) and (ROOT / rel).is_file()]
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
            tmp.write_bytes((stage / rel).read_bytes())
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
