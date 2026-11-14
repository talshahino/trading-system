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
import re
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
PUBLISH = ("reports/agents/", "data/agents/", "live/bots/")    # live/bots: approved cards built (agents/builder.py)
HOST = "reports/agents/host.json"
LIGHT_EVERY, SMART_EVERY = 10 * 60, 1 * 3600     # smart every hour (user, 12.11.2026: when there is work - work, no waiting)
AGENTS_REV = 2                                            # raise it: the agents' next smart run starts 15 minutes after the install
# the smart agents, in this order, with their minutes per smart run (the manager may change them in data/agents/plan.json,
# "pc": {"bot5": 40, ...}); an agent with nothing left to test ends at once
AGENTS = [("bot5", ["agent_bot5.py"], 40), ("bot6", ["agent_bot6.py"], 30),
          ("bot1", ["agent_portfolio.py", "--bot", "bot1"], 15), ("bot2", ["agent_portfolio.py", "--bot", "bot2"], 15),
          ("bot3", ["agent_portfolio.py", "--bot", "bot3"], 15), ("bot4", ["agent_portfolio.py", "--bot", "bot4"], 15),
          ("bot7", ["agent_portfolio.py", "--bot", "bot7"], 15), ("bot7-shocks", ["agents/shocks.py"], 25),
          ("bot5-new", ["agent_portfolio.py", "--bot", "bot5"], 5), ("bot6-new", ["agent_portfolio.py", "--bot", "bot6"], 5)]
HEARTBEAT = 3 * 3600                                      # publish the continuous layer at least this often
LIGHT_PUBLISH = 30 * 60                                   # ... and at most this often, unless the manager decided something
NODE_VER = "v20.18.1"
_S = {"sha": None, "light": 0.0, "smart": 0.0, "pub": 0.0, "proc": None, "log": print, "err": None, "smart_running": False}
import threading  # noqa: E402
_LOCK, _REFRESH, _LIGHT = threading.RLock(), threading.Lock(), threading.Lock()
# what the continuous layer writes - while a smart agent runs, only these are published by it (the agent's own files
# are published by the smart layer when the agent ends, never half-written)
import re  # noqa: E402
LIGHT_OWN = re.compile(r"^(reports/agents/(manager\.json|decisions\.json|watch\.json|manager_log\.json|losses/.*|[^/]+/(health|cards_health)\.json)"
                       r"|data/agents/(replies|priority|bot7_reset)\.json|data/agents/bot7_archive/|live/bots/)")
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
    with _LOCK:
        for _ in range(5):                    # the continuous layer's thread may change it at the same moment
            try:
                txt = json.dumps(st, ensure_ascii=False, indent=1)
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            return
        STATE.write_text(txt, encoding="utf-8")


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
        if not rel or info.is_dir() or rel.startswith(("docs/", "mt5/", "dashboard/")):   # live/: the bots' settings
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
                rel = f.relative_to(WORK).as_posix()
                # the gap method's run cache (reports/agents/bot5|bot6/tal: thousands of csv.gz) is the computer's own,
                # never published (manager, 12.11.2026: a first run on bot 6 made ~900 of them - one commit of 900 blobs)
                if f.is_file() and "/tal/" not in rel:
                    out[rel] = hashlib.sha1(f.read_bytes()).hexdigest()
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


# ---------- the data keys (user, 12.11.2026: stocks on the computer too) ----------
# Only GitHub has the Alpaca keys. The computer makes a key pair once (the private half never leaves C:\trading-agents\keys)
# and shows the public half in host.json; the workflow pc_keys.yml seals the keys for it into data/agents/pc_secrets.sealed,
# which only this computer can open. They are kept in memory and given only to the agents' own processes.
SEALED = "data/agents/pc_secrets.sealed"
_BOX = {}


def box_key():
    if "priv" not in _BOX:
        from nacl.public import PrivateKey
        f = WORK / "keys" / "pc_box.key"
        if f.exists():
            k = PrivateKey(base64.b64decode(f.read_text().strip()))
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            k = PrivateKey.generate()
            f.write_text(base64.b64encode(bytes(k)).decode())
        _BOX["priv"] = k
    return _BOX["priv"]


def box_pub():
    try:
        return base64.b64encode(bytes(box_key().public_key)).decode()
    except Exception:  # noqa: BLE001
        return None


def secrets():
    f = WORK / SEALED
    try:
        if not f.exists():
            return {}
        from nacl.public import SealedBox
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("for") != box_pub():
            return {}
        raw = SealedBox(box_key()).decrypt(base64.b64decode(d["sealed"]))
        return {k: v for k, v in json.loads(raw).items() if k.startswith("ALPACA_") and v}
    except Exception:  # noqa: BLE001
        return {}


# ---------- running ----------
def _flags():
    if os.name != "nt":
        return 0
    return 0x08000000 | 0x00004000          # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS


def _turn_back(st, r):
    """A run that a restart cut short (its process was gone after the restart, well inside its minutes) gets its turn
    back: its start stamp returns to the one before, so it goes first in the next round (manager, 13.11.2026: bot 2
    started 05:28, an install restarted the manager, bot 2 had 2 of its 30 minutes and then waited a whole round)."""
    name = (r or {}).get("name")
    if not name or not r.get("at"):
        return
    budget = max(60.0, float(r.get("timeout", 3600)) - 3600)
    if time.time() - float(r["at"]) < 0.5 * budget:
        last = st.setdefault("agent_last", {})
        if r.get("prev_last"):
            last[name] = r["prev_last"]
        else:
            last.pop(name, None)
        st.setdefault("agent_runs", {})[name] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(float(r["at"]))),
                                                 "sec": int(time.time() - float(r["at"])), "budget_sec": int(budget),
                                                 "ok": None, "err": "cut short by a restart - turn given back"}
        return True
    return False


def _publish_lost(st, r):
    """A run that ended (or was stopped) while the manager was down, past half its minutes: what it wrote is published
    before the next refresh writes main's copy over it (manager, 13.11.2026: bot 5's run of 07:18 UTC ended during the
    08:25 install - no commit, no error, its gap results were overwritten at 08:51 and the search looked stuck)."""
    name = (r or {}).get("name")
    if not name or not r.get("at"):
        return
    t_r = float(r["at"])
    st.setdefault("agent_tail", {})[name] = _tail(f"agent_{name}.log") + ["(ended while the manager restarted)"]
    st.setdefault("agent_runs", {})[name] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(t_r)),
                                             "sec": None, "budget_sec": int(r.get("timeout", 3600)) - 3600,
                                             "ok": None, "err": "ended while the manager restarted - published after"}
    after = snapshot()
    before = r.get("before") or {}
    bad = set()
    for rel, hsh in after.items():                    # a file cut half-way by a stop is not published
        if before.get(rel) != hsh and rel.endswith(".json"):
            try:
                json.loads((WORK / rel).read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                bad.add(rel)
    only = re.compile("^(?!" + "|".join(re.escape(b) + "$" for b in bad) + ")") if bad else None
    try:
        _publish_changes(st, before, "agent", f"Agents on the bots' computer: {name} smart layer (ended in a restart)",
                         only=only)
    except Exception as e:  # noqa: BLE001
        _S["log"](f"publishing {name} after the restart failed: {e}")
        hist = [x for x in st.get("agent_errors_24h") or [] if time.time() - (_ts(x.get("at")) or 0.0) < 86400]
        st["agent_errors_24h"] = (hist + [{"at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                                           "err": f"{name}: publishing after a restart failed: {e}"[:300]}])[-20:]


def kill_orphan(st):
    """A run left over from before the manager restarted (e.g. after an update) is stopped first."""
    pid = st.pop("pid", None)
    if pid and os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=_flags())
    _save(st)


def _alive(pid):
    """Is this pid still a running python (Windows: tasklist; never os.kill - on Windows it terminates)?"""
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:  # noqa: BLE001
            return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"], capture_output=True,
                             text=True, timeout=30, creationflags=_flags()).stdout.lower()
    except Exception:  # noqa: BLE001
        return False
    return "python" in out


def adopt_orphan(st):
    """A smart agent left running by a restart (every code install restarts the manager; its child processes live on)
    is waited for and then published - not killed (manager, 12.11.2026: 8+ installs a day killed bot 5's and bot 6's
    gap runs half-way, 18:50 / 18:58 UTC, so their results stopped updating). Past its own timeout it is stopped."""
    pid, r = st.get("pid"), st.get("run") or {}
    if r.get("name") and r.get("ended"):
        # the agent ended and the manager restarted before its publish (manager, 14.11.2026: bot 6's 68-minute run of
        # 23:16 UTC ended ~00:25, an install restarted the manager during the whole-bot replay that runs between the
        # agent's end and its publish - no commit, no error, its gap results stayed from 13.11 16:07)
        try:
            _publish_lost(st, r)
        except Exception as e:  # noqa: BLE001
            _S["log"](f"publishing the ended agent failed: {e}")
        st.pop("run", None)
        _S["smart_running"] = False
        return kill_orphan(st)
    if not (pid and r.get("name") and time.time() - float(r.get("at", 0)) < float(r.get("timeout", 0)) and _alive(pid)):
        gone = bool(pid and r.get("name") and not _alive(pid))
        if not _turn_back(st, r) and gone:
            try:
                _publish_lost(st, r)
            except Exception as e:  # noqa: BLE001
                _S["log"](f"publishing the ended agent failed: {e}")
        st.pop("run", None)
        _S["smart_running"] = False
        return kill_orphan(st)
    name, log = r["name"], _S["log"]
    log(f"agent {name} (pid {pid}) still runs after the restart: waiting for it")
    _S["smart_running"] = True
    try:
        while _alive(pid) and time.time() - float(r["at"]) < float(r["timeout"]):
            time.sleep(30)
        if _alive(pid):
            kill_orphan(st)
        st.pop("pid", None)
        st.setdefault("agent_tail", {})[name] = _tail(f"agent_{name}.log") + ["(continued across a manager restart)"]
        t_r = float(r.get("at", 0))
        st.setdefault("agent_runs", {})[name] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(t_r)),
                                                 "sec": int(time.time() - t_r), "budget_sec": int(r.get("timeout", 3600)) - 3600,
                                                 "ok": None, "err": None, "adopted": True}
        try:
            _publish_changes(st, r.get("before") or {}, "agent", f"Agents on the bots' computer: {name} smart layer")
        except Exception as e:  # noqa: BLE001
            log(f"publishing {name} failed: {e}")
            st["agent_errors"] = (st.get("agent_errors") or []) + [f"{name}: publishing failed: {e}"[:300]]
    finally:
        st.pop("run", None)
        _save(st)
        _S["smart_running"] = False


def run(st, args, timeout, logname, track=True):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               TS_TOKEN_FILE=str(ROOT / "live" / "github_token.txt"))
    tok = REP.token()
    if tok:
        env["GH_TOKEN"] = tok
    env.update(secrets())
    node = ensure_node()
    if node:
        env["PATH"] = node + os.pathsep + env.get("PATH", "")
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / logname, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([_py(), "-u"] + args, cwd=WORK, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             creationflags=_flags())
        if track:                             # (the smart agents; a restart stops a left-over one)
            st["pid"] = p.pid
            _save(st)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, creationflags=_flags()) \
                if os.name == "nt" else p.kill()
            rc = "timeout"
    if track:
        st.pop("pid", None)
        _save(st)
    if rc != 0:
        tail = (logs / logname).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(f"{' '.join(args[:1])}: {rc} · {' | '.join(tail)[-300:]}")


# ---------- publishing (one commit through the GitHub API, the computer has no git) ----------
REPLIES = "data/agents/replies.json"


def _later(a, b):
    """b over a, unless a is a dict with a newer 'at' (an answer written on main after this copy)."""
    if isinstance(a, dict) and isinstance(b, dict) and str(a.get("at") or "") > str(b.get("at") or ""):
        return a
    return b


def merge_replies(main_b, own_b):
    """replies.json is written both here and on main (the user's tasks, the nightly agents' answers): the whole file
    from this copy erased the user's new tasks (12.11.2026: three of his tasks lost in one health commit). Main is
    the base; this copy adds its tasks / answers / status, never removes."""
    m, o = json.loads(main_b.decode("utf-8")), json.loads(own_b.decode("utf-8"))
    out = dict(m)
    for k, v in o.items():
        if k != "_tasks":
            out[k] = _later(m[k], v) if k in m else v
    key = lambda t: str(t.get("task", ""))[:200]  # noqa: E731
    mine = {key(t): t for t in o.get("_tasks") or [] if isinstance(t, dict)}
    tasks, seen = [], set()
    for t in m.get("_tasks") or []:
        if not isinstance(t, dict):
            continue
        k, t = key(t), dict(t)
        seen.add(k)
        if k in mine:
            for f in ("status", "done"):
                a, b = t.get(f) or {}, mine[k].get(f) or {}
                if isinstance(a, dict) and isinstance(b, dict) and (a or b):
                    t[f] = {**a, **{bk: _later(a[bk], bv) if bk in a else bv for bk, bv in b.items()}}
        tasks.append(t)
    tasks += [t for k, t in mine.items() if k not in seen]
    out["_tasks"] = tasks
    return json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")


def _main_file(h, sha, rel):
    r = requests.get(f"{API}/contents/{rel}", headers=h, params={"ref": sha}, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    j = r.json()
    if j.get("content") and j.get("encoding") == "base64":
        return base64.b64decode(j["content"])
    raw = requests.get(f"{API}/contents/{rel}", headers={**h, "Accept": "application/vnd.github.raw"},
                       params={"ref": sha}, timeout=60)
    raw.raise_for_status()
    return raw.content


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
        if REPLIES in files:                          # merged into main's replies.json, never over it
            try:
                mb = _main_file(h, hsha, REPLIES)
                data = merge_replies(mb, files[REPLIES]) if mb else files[REPLIES]
                r = requests.post(f"{API}/git/blobs", headers=h, timeout=120,
                                  json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
                r.raise_for_status()
                blobs[REPLIES] = r.json()["sha"]
            except Exception:  # noqa: BLE001  - unreadable: leave main's file as it is this time
                blobs.pop(REPLIES, None)
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
            "agent_last": st.get("agent_last") or {}, "agent_tail": st.get("agent_tail") or {},
            "agent_runs": st.get("agent_runs") or {}, "agent_errors_24h": st.get("agent_errors_24h") or [],
            "box_pub": box_pub(), "alpaca": bool(secrets()),
            "last_error": st.get("last_error"),
            "errors_in_row": st.get("errors_in_row", 0)}


def _publish_changes(st, before, kind, msg, only=None):
    after = snapshot()
    changed = [rel for rel, hsh in after.items() if before.get(rel) != hsh and rel != HOST
               and (only is None or only.match(rel))]
    files = {rel: (WORK / rel).read_bytes() for rel in changed}
    if any(r.startswith("live/bots/") for r in files):
        try:
            moved = main_sha() != st.get("sha")
        except Exception:  # noqa: BLE001
            moved = True
        if moved:          # main changed since this copy (e.g. a nightly agent edited a bot): build again on the new one
            files = {r: v for r, v in files.items() if not r.startswith("live/bots/") and r != "data/agents/replies.json"}
    if kind == "light" and time.time() - _S["pub"] < HEARTBEAT:
        prev = st.get("pub_digest", {})
        news = {rel: hashlib.sha1(_meaningful(rel, b)).hexdigest() for rel, b in files.items()}
        decided = any(prev.get(rel) != d for rel, d in news.items() if rel.endswith("decisions.json") or rel.startswith("live/bots/"))
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
    with _LIGHT:                              # the smart round's refresh between agents waits for this cycle to end
        return _light(st, refresh_first)


def _light(st, refresh_first=True):
    t0 = time.time()
    busy = _S["smart_running"]                # a smart agent is running: no refresh under it, publish only own files
    if refresh_first and not busy:
        with _REFRESH:
            refresh(st)
    before = snapshot()
    kw = {"track": False}
    run(st, ["-m", "agents.health"], timeout=900, logname="health.log", **kw)
    run(st, ["agent_manager.py"], timeout=1200, logname="agent_manager.log", **kw)
    run(st, ["-m", "agents.builder"], timeout=300, logname="builder.log", **kw)    # the decisions into the bots, at once
    run(st, ["-m", "agents.watch"], timeout=300, logname="watch.log", **kw)        # every change followed, bad ones back
    run(st, ["-m", "agents.loss_review"], timeout=300, logname="loss_review.log", **kw)   # every loss day: a report + tasks
    run(st, ["-m", "agents.builder"], timeout=300, logname="builder2.log", **kw)   # a reverted card out of the bot now
    st["last_error"], st["errors_in_row"] = None, 0
    n = _publish_changes(st, before, "light", "Agents on the bots' computer: health", only=LIGHT_OWN if busy else None)
    _S["light"] = time.time()
    _S["log"](f"agents (continuous layer) done in {time.time() - t0:.0f}s, {n} files published")


def smart(st):
    """The smart layer: every bot's agent in turn, each published as it ends. The continuous layer runs in its own
    thread meanwhile, every 10 minutes, and never waits for an agent (12.11.2026: a 2-hour agent run held health,
    the manager, the builder and the watch back for 2 hours)."""
    log = _S["log"]
    t0 = time.time()
    with _REFRESH:
        refresh(st)
        _S["smart_running"] = True
    try:
        return _smart(st, log, t0)
    finally:
        _S["smart_running"] = False


def _smart(st, log, t0):
    try:
        plan = json.loads((WORK / "data/agents/plan.json").read_text(encoding="utf-8")).get("pc", {})
    except Exception:  # noqa: BLE001
        plan = {}
    errs, mins_all = [], {}
    # the agent that waited longest goes first (manager, 12.11.2026: every code install restarts the manager, which
    # stopped the round in bot 5's 120 minutes and started it again from bot 5 - 8 installs in one day and bots 1-4,
    # 6 and 7 got no turn for 5+ hours). Each agent is stamped when it starts, so a stopped round goes on with the
    # agents that did not get their turn.
    if "agent_last" not in st:                # first time: bot 5 ran again and again on 12.11 - the others first
        now_s = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        st["agent_last"] = {"bot5": now_s, "bot5-new": now_s}
    last = st["agent_last"]
    order = sorted(AGENTS, key=lambda a: _ts(last.get(a[0])) or 0.0)
    for name, args, mins in order:
        mins = float(plan.get(name, mins))
        mins_all[name] = mins
        if mins <= 0:
            continue
        # the newest main before every agent (manager, 13.11.2026: the round refreshed only at its start and the
        # continuous layer does not refresh under a running agent - a 6-hour round kept the old agent_manager.py, which
        # failed every 10 minutes from 11:18 to 14:16 UTC although the fix was on main since 12:25). Between agents
        # nothing of the round is unpublished; the continuous layer's cycle is waited for.
        try:
            with _LIGHT, _REFRESH:
                refresh(st)
        except Exception as e:  # noqa: BLE001
            log(f"refresh before {name} failed: {e}")
        prev_last = last.get(name)
        last[name] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        _save(st)
        before = snapshot()
        # kept so that a restart in the middle can wait for this agent and publish it (adopt_orphan)
        st["run"] = {"name": name, "at": time.time(), "timeout": int(mins * 60 + 3600), "before": before,
                     "prev_last": prev_last}
        t_a, err = time.time(), None
        try:
            run(st, args + ["--minutes", str(mins)], timeout=int(mins * 60 + 3600), logname=f"agent_{name}.log")
        except Exception as e:  # noqa: BLE001 - one agent's failure does not stop the others
            err = f"{name}: {e}"[:300]
            errs.append(err)
            log(f"agent {name} failed: {e}")
        if st.get("run"):                     # ended: a restart from here on still publishes it (adopt_orphan)
            st["run"]["ended"] = True
            _save(st)
        # every agent's last run, kept across rounds (manager, 13.11.2026: bot 2 ran 2 of its 30 minutes at 05:28 and
        # published nothing; agent_errors only holds the current round, and every code install starts a new round,
        # so a failure was gone from host.json before anyone saw it)
        st.setdefault("agent_runs", {})[name] = {"at": last[name], "sec": int(time.time() - t_a), "budget_sec": int(mins * 60),
                                                 "ok": err is None, "err": err}
        if err:
            hist = [x for x in st.get("agent_errors_24h") or [] if time.time() - (_ts(x.get("at")) or 0.0) < 86400]
            st["agent_errors_24h"] = (hist + [{"at": last[name], "err": err}])[-20:]
        # what the agent printed about its own failures (manager, 12.11.2026: bot 6's gap search ended without results
        # and without an error - an exception caught and printed inside the agent is invisible from here otherwise)
        st.setdefault("agent_tail", {})[name] = _tail(f"agent_{name}.log")
        # the daily whole-bot replay also inside a round (manager, 13.11.2026: every code install restarts the manager,
        # the round never reached its end and reports/agents/<bot>/portfolio.json stayed from 12.11 11:41 for 19 hours)
        if time.time() - (_ts(st.get("portfolio_at")) or 0.0) >= 3 * 3600:
            try:
                _portfolio_check(st)
            except Exception as e:  # noqa: BLE001
                log(f"portfolio check failed: {e}")
        st["smart_minutes"], st["agent_errors"] = mins_all, errs
        try:
            _publish_changes(st, before, "agent", f"Agents on the bots' computer: {name} smart layer")
        except Exception as e:  # noqa: BLE001
            log(f"publishing {name} failed: {e}")
            errs.append(f"{name}: publishing failed: {e}"[:300])     # shown in host.json on the next publish
            st["agent_errors"] = errs
            hist = [x for x in st.get("agent_errors_24h") or [] if time.time() - (_ts(x.get("at")) or 0.0) < 86400]
            st["agent_errors_24h"] = (hist + [{"at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                                               "err": f"{name}: publishing failed: {e}"[:300]}])[-20:]
        st.pop("run", None)                   # only now: published (or its failure recorded)
        _save(st)
    before = snapshot()
    _portfolio_check(st)                                 # the whole bots, as they are
    host_doc(st, "smart")                                # stamps smart_at (the manager / builder: the continuous layer)
    st["last_error"], st["errors_in_row"] = None, 0
    n = _publish_changes(st, before, "smart", "Agents on the bots' computer: smart layer")
    log(f"agents (smart layer) done in {time.time() - t0:.0f}s, {n} files published, {len(errs)} agents failed")


def _portfolio_check(st):
    run(st, ["-m", "agents.portfolio_check"], timeout=600, logname="portfolio_check.log")
    st["portfolio_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _save(st)


def _tail(logname):
    """The agent log's lines about failures (at most 6) and its last 2 lines, for host.json."""
    try:
        lines = (WORK / "logs" / logname).read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except Exception:  # noqa: BLE001
        return []
    bad = [x for x in lines if re.search(r"failed|Error|Traceback|no data", x)]
    return [x[:240] for x in bad[-6:] + lines[-2:]]


def cycle(st, kind):
    return smart(st) if kind == "smart" else light(st)


def loop(log=print, enabled=lambda: True):
    """The manager starts this in a thread."""
    _S["log"] = log
    time.sleep(90)                                   # let the bots start first
    st = _load()
    orphan = bool(st.get("run") and (st.get("pid") or (st.get("run") or {}).get("ended")))
    if orphan:
        _S["smart_running"] = True                   # the continuous layer: no refresh under the running agent
    else:
        kill_orphan(st)
    _S["smart"] = _ts(st.get("smart_at")) or (time.time() - SMART_EVERY + 15 * 60)   # first time: health first
    names = [a[0] for a in AGENTS] + [AGENTS_REV]
    if st.get("agents_set") != names:                # new agents installed: their first smart run in 15 minutes
        _S["smart"] = min(_S["smart"], time.time() - SMART_EVERY + 15 * 60)
        st["agents_set"] = names
        _save(st)
    threading.Thread(target=light_loop, args=(st, enabled), daemon=True).start()
    if orphan:
        try:
            adopt_orphan(st)
        except Exception as e:  # noqa: BLE001
            _S["smart_running"] = False
            log(f"waiting for the left-over agent failed: {e}")
    while True:
        try:
            if enabled():
                now = time.time()
                kind = "smart" if now - _S["smart"] >= SMART_EVERY else None
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


def light_loop(st, enabled):
    """The continuous layer, every 10 minutes, in its own thread (never waits for a smart agent)."""
    log = _S["log"]
    while True:
        try:
            if enabled() and time.time() - _S["light"] >= LIGHT_EVERY:
                try:
                    light(st)
                finally:
                    _S["light"] = time.time()
        except Exception as e:  # noqa: BLE001
            st["last_error"] = f"{time.strftime('%d.%m %H:%M')} {e}"[:400]
            st["errors_in_row"] = st.get("errors_in_row", 0) + 1
            _save(st)
            log(f"continuous layer failed: {e}\n{traceback.format_exc()[-600:]}")
        time.sleep(20)


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
