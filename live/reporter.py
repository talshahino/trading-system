"""Reporting for the live robot: phone notifications (ntfy app) and the dashboard state file.

- Notifications: every event is POSTed to https://ntfy.sh/<topic>; the phone app ntfy, subscribed to
  that topic, shows it at once. The topic is in live/notify.json.
- Dashboard: every few minutes the full account state is written to branch "live",
  file live/<bot>_state.json, through the GitHub API with the token in live/github_token.txt
  (fine-grained, this repository only, Contents: read and write). docs/index.html reads it.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
REPO = "talshahino/trading-system"
_CFG = json.loads((ROOT / "live" / "notify.json").read_text(encoding="utf-8"))
_TOKEN_FILE = ROOT / "live" / "github_token.txt"
_LAST = {}


def notify(title, message, priority=3, tags=None, key=None, every=0):
    """Phone notification. key+every: send the same kind of message at most once per `every` seconds."""
    if key and every and time.time() - _LAST.get(key, 0) < every:
        return
    if key:
        _LAST[key] = time.time()
    if priority < 5 and quiet_now():
        priority = 2          # night: shows in the app without sound or vibration; urgent alerts still ring
    try:
        # JSON publishing: Hebrew titles are not allowed in HTTP headers
        r = requests.post("https://ntfy.sh/", timeout=10, json={
            "topic": _CFG["topic"], "title": title, "message": message, "priority": int(priority), "tags": list(tags or [])})
        return None if r.status_code == 200 else f"ntfy answered {r.status_code}: {r.text[:120]}"
    except Exception as e:  # noqa: BLE001 - a notification must never stop the robot
        return f"notification failed: {e}"


def notify_action(title, message, label, url, body, priority=4):
    """Notification with a button that POSTs `body` to `url` (ntfy "http" action)."""
    try:
        r = requests.post("https://ntfy.sh/", timeout=10, json={
            "topic": _CFG["topic"], "title": title, "message": message, "priority": int(priority), "tags": ["arrow_up"],
            "actions": [{"action": "http", "label": label, "url": url, "method": "POST", "body": body, "clear": True}]})
        return None if r.status_code == 200 else f"ntfy answered {r.status_code}: {r.text[:120]}"
    except Exception as e:  # noqa: BLE001
        return f"notification failed: {e}"


def quiet_now():
    """Quiet hours (Israel time) from live/notify.json, default 21:00-06:00."""
    try:
        import pandas as pd        # pandas ships its own time-zone data (Windows has none for zoneinfo)
        a, b = _CFG.get("quiet_from", 21), _CFG.get("quiet_to", 6)
        h = pd.Timestamp.now(tz="Asia/Jerusalem").hour
        return (h >= a or h < b) if a > b else (a <= h < b)
    except Exception:  # noqa: BLE001
        return False


def token():
    try:
        return _TOKEN_FILE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def push_state(name, state: dict):
    """Write live/<name>_state.json on branch live. Returns an error text or None."""
    tok = token()
    if not tok:
        return "no GitHub token (live/github_token.txt): dashboard not updated"
    url = f"https://api.github.com/repos/{REPO}/contents/live/{name}_state.json"
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(url, params={"ref": "live"}, headers=h, timeout=15)
        sha = r.json().get("sha") if r.status_code == 200 else None
        body = {"message": f"{name} state", "branch": "live",
                "content": base64.b64encode(json.dumps(state, ensure_ascii=False, default=str).encode("utf-8")).decode()}
        if sha:
            body["sha"] = sha
        r = requests.put(url, headers=h, json=body, timeout=20)
        return None if r.status_code in (200, 201) else f"GitHub answered {r.status_code}: {r.text[:150]}"
    except Exception as e:  # noqa: BLE001
        return f"dashboard push failed: {e}"
