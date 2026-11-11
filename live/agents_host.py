"""The agents on the bots' computer (user, 11.11.2026: "the agents' heavy tests run on my computer, free").

Runs inside the bots' manager (live/manager.py), in the background, at low priority and without any window, so
the bots' trading always comes first. Two layers, like the plan:
  continuous layer  every 10 minutes: every bot's health (agents/health.py) + the manager agent (agent_manager.py)
  smart layer       every 4 hours:   every bot's smart agent for its time budget (AGENTS below), then the manager
Everything runs in its own working copy, C:\\trading-agents (never in the bots' own folder), refreshed from GitHub
(branch main) before every run; the market-data cache stays there between runs, so data is downloaded once.
The results (reports/agents/, data/agents/) are published to GitHub in one commit, where the app reads them.
Status: reports/agents/host.json. While the computer runs the agents, the GitHub jobs (agents.yml, health.yml) see
it there and skip, and take over again by themselves if the computer stops.
Node.js (for the Dukascopy market data) is downloaded once into the working copy when the computer has none.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import requests

from live import reporter as REP

ROOT = Path(__file__).resolve().parent.parent              # C:\trading-system (the bots)
WORK = Path(os.environ.get("TS_AGENTS_DIR") or (ROOT.parent / "trading-agents"))
REPO = REP.REPO
API = f"https://api.github.com/repos/{REPO}"
PUBLISH = ("reports/agents/", "data/agents/")
HOST = "reports/agents/host.json"
LIGHT_EVERY, SMART_EVERY = 10 * 60, 4 * 3600
# the smart agents, in this order, with their minutes per smart run (the manager may change them in data/agents/plan.json,
# "pc": {"bot5": 40, ...}); an agent with nothing left to test ends at once
AGENTS = [("bot5", ["agent_bot5.py"], 40), ("bot6", ["agent_bot6.py"], 30),
          ("bot1", ["agent_portfolio.py", "--bot", "bot1"], 15), ("bot2", ["agent_portfolio.py", "--bot", "bot2"], 15),
          ("bot3", ["agent_portfolio.py", "--bot", "bot3"], 15), ("bot4", ["agent_portfolio.py", "--bot", "bot4"], 15),
          ("bot7", ["agent_portfolio.py", "--bot", "bot7"], 15),
          ("bot5-new", ["agent_portfolio.py", "--bot", "bot5"], 5), ("bot6-new", ["agent_portfolio.py", "--bot", "bot6"], 5)]
HEARTBEAT = 3 * 3600                                      # publish the continuous layer at least this often
LIGHT_PUBLISH = 30 * 60                                   # ... and at most this often, unless the manager decided something
NODE_VER = "v20.18.1"
_S = {"sha": None, "light": 0.0, "smart": 0.0, "pub": 0.0, "proc": None, "log": print, "err": None}
STATE = WORK / "host_state.json"


def _h():
    tok = REP.token()
    return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"} if tok else {}


def _py():
    return sys.executable.replace("pythonw.exe", "python.exe").replace("pythonw", "python")


def _load():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save(st):
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- the working copy ----------
def main_sha():
    r = requests.get(f"{API}/commits/main", headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json()["sha"]


def refresh(st):
    """Bring the working copy to the newest main (code and the latest published results). The cache stays."""
    sha = main_sha()
    if st.get("sha") == sha and (WORK / "agent_bot5.py").exists():
        return sha
    r = requests.get(f"{API}/zipball/{sha}", headers=_h(), timeout=300)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    top = z.namelist()[0].split("/")[0]
    for info in z.infolist():
        rel = info.filename[len(top) + 1:]
        if not rel or info.is_dir() or rel.startswith(("live/", "docs/", "mt5/", "dashboard/")):
            continue
        dest = WORK / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = z.read(info)
        if not (dest.exists() and dest.read_bytes() == data):
            dest.write_bytes(data)
    req = (WORK / "requirements.txt").read_bytes() if (WORK / "requirements.txt").exists() else b""
    hreq = hashlib.sha1(req).hexdigest()
    if req and st.get("req") != hreq:
        subprocess.run([_py(), "-m", "pip", "install", "-q", "-r", str(WORK / "requirements.txt")], cwd=WORK,
                       capture_output=True, timeout=1800, creationflags=_flags())
        st["req"] = hreq
    st["sha"] = sha
    _save(st)
    return sha


def ensure_node():
    """npx for the Dukascopy data: the computer's own Node.js, or a private copy in the working copy."""
    if shutil.which("npx"):
        return None
    home = WORK / "node"
    if not (home / "npx.cmd").exists():
        url = f"https://nodejs.org/dist/{NODE_VER}/node-{NODE_VER}-win-x64.zip"
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        tmp = WORK / "node_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
        shutil.rmtree(home, ignore_errors=True)
        (tmp / f"node-{NODE_VER}-win-x64").rename(home)
        shutil.rmtree(tmp, ignore_errors=True)
    return str(home)


def snapshot():
    out = {}
    for d in PUBLISH:
        base = WORK / d
        if base.exists():
            for f in base.rglob("*"):
                if f.is_file():
                    out[f.relative_to(WORK).as_posix()] = hashlib.sha1(f.read_bytes()).hexdigest()
    return out


def _meaningful(rel, data):
    """The content without its time stamps (a file that only got a new 'updated' is not news)."""
    if not rel.endswith(".json"):
        return data
    try:
        j = json.loads(data.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return data
    if isinstance(j, dict):
        for k in ("updated", "updated_utc", "generated", "at"):
            j.pop(k, None)
    return json.dumps(j, sort_keys=True, default=str).encode()


# ---------- running ----------
def _flags():
    if os.name != "nt":
        return 0
    return 0x08000000 | 0x00004000          # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS


def kill_orphan(st):
    """A run left over from before the manager restarted (e.g. after an update) is stopped first."""
    pid = st.pop("pid", None)
    if pid and os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=_flags())
    _save(st)


def run(st, args, timeout, logname):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               TS_TOKEN_FILE=str(ROOT / "live" / "github_token.txt"))
    tok = REP.token()
    if tok:
        env["GH_TOKEN"] = tok
    node = ensure_node()
    if node:
        env["PATH"] = node + os.pathsep + env.get("PATH", "")
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / logname, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([_py(), "-u"] + args, cwd=WORK, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             creationflags=_flags())
        st["pid"] = p.pid
        _save(st)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, creationflags=_flags()) \
                if os.name == "nt" else p.kill()
            rc = "timeout"
    st.pop("pid", None)
    _save(st)
    if rc != 0:
        tail = (logs / logname).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(f"{' '.join(args[:1])}: {rc} · {' | '.join(tail)[-300:]}")


# ---------- publishing (one commit through the GitHub API, the computer has no git) ----------
def publish(files, message):
    """files: {path: bytes}. One commit on main; when main moved meanwhile, again on the new main (these files
    are generated here, so this run's copy wins)."""
    if not files:
        return None
    h = _h()
    blobs = {}
    for rel, data in files.items():
        r = requests.post(f"{API}/git/blobs", headers=h, timeout=120,
                          json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
        r.raise_for_status()
        blobs[rel] = r.json()["sha"]
    for attempt in range(5):
        head = requests.get(f"{API}/git/ref/heads/main", headers=h, timeout=30)
        head.raise_for_status()
        hsha = head.json()["object"]["sha"]
        base_tree = requests.get(f"{API}/git/commits/{hsha}", headers=h, timeout=30).json()["tree"]["sha"]
        tree = requests.post(f"{API}/git/trees", headers=h, timeout=60, json={
            "base_tree": base_tree,
            "tree": [{"path": rel, "mode": "100644", "type": "blob", "sha": s} for rel, s in blobs.items()]})
        tree.raise_for_status()
        c = requests.post(f"{API}/git/commits", headers=h, timeout=60,
                          json={"message": message, "tree": tree.json()["sha"], "parents": [hsha]})
        c.raise_for_status()
        u = requests.patch(f"{API}/git/refs/heads/main", headers=h, timeout=30, json={"sha": c.json()["sha"]})
        if u.status_code == 200:
            return c.json()["sha"]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"publishing failed: {u.status_code} {u.text[:200]}")


def host_doc(st, kind):
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    st[f"{kind}_at"] = now
    return {"pc": True, "computer": os.environ.get("COMPUTERNAME", "?"), "updated": now,
            "light_at": st.get("light_at"), "smart_at": st.get("smart_at"),
            "every_min": {"light": LIGHT_EVERY // 60, "smart": SMART_EVERY // 60},
            "smart_minutes": st.get("smart_minutes"), "agent_errors": st.get("agent_errors") or [],
            "last_error": st.get("last_error"),
            "errors_in_row": st.get("errors_in_row", 0)}


def _publish_changes(st, before, kind, msg):
    after = snapshot()
    changed = [rel for rel, hsh in after.items() if before.get(rel) != hsh and rel != HOST]
    files = {rel: (WORK / rel).read_bytes() for rel in changed}
    if kind == "light" and time.time() - _S["pub"] < HEARTBEAT:
        prev = st.get("pub_digest", {})
        news = {rel: hashlib.sha1(_meaningful(rel, b)).hexdigest() for rel, b in files.items()}
        decided = any(prev.get(rel) != d for rel, d in news.items() if rel.endswith("decisions.json"))
        if all(prev.get(rel) == d for rel, d in news.items()) or (time.time() - _S["pub"] < LIGHT_PUBLISH and not decided):
            files = {}                                  # nothing new (or published less than 30 minutes ago): no commit
    doc = host_doc(st, kind)
    if files or kind != "light" or time.time() - _S["pub"] >= HEARTBEAT:
        files[HOST] = json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")
        (WORK / HOST).write_bytes(files[HOST])
        publish(files, msg)
        _S["pub"] = time.time()
        st["pub_digest"] = {**st.get("pub_digest", {}),
                            **{rel: hashlib.sha1(_meaningful(rel, b)).hexdigest() for rel, b in files.items()}}
    _save(st)
    return len(files)


def light(st, refresh_first=True):
    """The continuous layer: every bot's health + the manager agent."""
    t0 = time.time()
    if refresh_first:
        refresh(st)
    before = snapshot()
    run(st, ["-m", "agents.health"], timeout=900, logname="health.log")
    run(st, ["agent_manager.py"], timeout=1200, logname="agent_manager.log")
    st["last_error"], st["errors_in_row"] = None, 0
    n = _publish_changes(st, before, "light", "Agents on the bots' computer: health")
    _S["light"] = time.time()
    _S["log"](f"agents (continuous layer) done in {time.time() - t0:.0f}s, {n} files published")


def smart(st):
    """The smart layer: every bot's agent in turn, each published as it ends; the continuous layer keeps its
    10 minutes in between (between two agents)."""
    log = _S["log"]
    t0 = time.time()
    refresh(st)
    try:
        plan = json.loads((WORK / "data/agents/plan.json").read_text(encoding="utf-8")).get("pc", {})
    except Exception:  # noqa: BLE001
        plan = {}
    errs, mins_all = [], {}
    for name, args, mins in AGENTS:
        mins = float(plan.get(name, mins))
        mins_all[name] = mins
        if mins <= 0:
            continue
        if time.time() - _S["light"] >= LIGHT_EVERY:
            try:
                light(st, refresh_first=False)
            except Exception as e:  # noqa: BLE001
                log(f"continuous layer failed: {e}")
        before = snapshot()
        try:
            run(st, args + ["--minutes", str(mins)], timeout=int(mins * 60 + 3600), logname=f"agent_{name}.log")
        except Exception as e:  # noqa: BLE001 - one agent's failure does not stop the others
            errs.append(f"{name}: {e}"[:300])
            log(f"agent {name} failed: {e}")
        st["smart_minutes"], st["agent_errors"] = mins_all, errs
        try:
            _publish_changes(st, before, "agent", f"Agents on the bots' computer: {name} smart layer")
        except Exception as e:  # noqa: BLE001
            log(f"publishing {name} failed: {e}")
    before = snapshot()
    run(st, ["agent_manager.py"], timeout=1200, logname="agent_manager.log")
    host_doc(st, "smart")                                # stamps smart_at
    st["last_error"], st["errors_in_row"] = None, 0
    n = _publish_changes(st, before, "smart", "Agents on the bots' computer: smart layer")
    _S["light"] = time.time()
    log(f"agents (smart layer) done in {time.time() - t0:.0f}s, {n} files published, {len(errs)} agents failed")


def cycle(st, kind):
    return smart(st) if kind == "smart" else light(st)


def loop(log=print, enabled=lambda: True):
    """The manager starts this in a thread."""
    _S["log"] = log
    time.sleep(90)                                   # let the bots start first
    st = _load()
    kill_orphan(st)
    _S["smart"] = _ts(st.get("smart_at")) or (time.time() - SMART_EVERY + 15 * 60)   # first time: health first
    names = [a[0] for a in AGENTS]
    if st.get("agents_set") != names:                # new agents installed: their first smart run in 15 minutes
        _S["smart"] = min(_S["smart"], time.time() - SMART_EVERY + 15 * 60)
        st["agents_set"] = names
        _save(st)
    while True:
        try:
            if enabled():
                now = time.time()
                kind = "smart" if now - _S["smart"] >= SMART_EVERY else "light" if now - _S["light"] >= LIGHT_EVERY else None
                if kind:
                    if kind == "smart":
                        _S["smart"] = time.time()           # every 4 hours from start to start
                    try:
                        cycle(st, kind)
                    finally:
                        if kind == "light":
                            _S["light"] = time.time()
        except Exception as e:  # noqa: BLE001
            st["last_error"] = f"{time.strftime('%d.%m %H:%M')} {e}"[:400]
            st["errors_in_row"] = st.get("errors_in_row", 0) + 1
            _save(st)
            log(f"agents failed: {e}\n{traceback.format_exc()[-600:]}")
            if st["errors_in_row"] in (3, 10):
                try:
                    doc = host_doc(st, "error")
                    publish({HOST: json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")},
                            "Agents on the bots' computer: error")
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(30)


def _ts(s):
    try:
        import calendar
        return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:  # noqa: BLE001
        return 0.0


def status():
    st = _load()
    return {"on": True, "dir": str(WORK), "light_at": st.get("light_at"), "smart_at": st.get("smart_at"),
            "last_error": st.get("last_error"), "running": bool(st.get("pid"))}
