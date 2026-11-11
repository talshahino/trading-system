"""Reading files from branch "live" (the bots' live states, the user's decisions) the same way on GitHub and on
the bots' computer (user, 11.11.2026: the agents move to the bots' computer, which has no git).

GitHub Actions: a git checkout -> `git show origin/live:<path>`.
The bots' computer: no git -> the GitHub API with the token (env GH_TOKEN, or live/github_token.txt).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "talshahino/trading-system"
_FETCHED = {"live": False}


def _git():
    return shutil.which("git") and (ROOT / ".git").exists()


def _token():
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    for f in (ROOT / "live" / "github_token.txt", Path(os.environ.get("TS_TOKEN_FILE", "")) if os.environ.get("TS_TOKEN_FILE") else None):
        try:
            if f and f.is_file():
                return f.read_text(encoding="utf-8").strip() or None
        except Exception:  # noqa: BLE001
            pass
    return None


def _api(path, raw=False):
    import requests
    h = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
    tok = _token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    r = requests.get(f"https://api.github.com/repos/{REPO}/contents/{path}", params={"ref": "live"}, headers=h, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content if raw else r.json()


def _fetch():
    if not _FETCHED["live"]:
        subprocess.run(["git", "fetch", "-q", "origin", "+refs/heads/live:refs/remotes/origin/live"], cwd=ROOT, check=False)
        _FETCHED["live"] = True


def live_text(path):
    """The text of <path> on branch live ('' when missing or unreachable)."""
    try:
        if _git():
            _fetch()
            return subprocess.run(["git", "show", f"origin/live:{path}"], cwd=ROOT, capture_output=True,
                                  text=True, encoding="utf-8").stdout
        b = _api(path, raw=True)
        return b.decode("utf-8") if b else ""
    except Exception:  # noqa: BLE001
        return ""


def live_json(path, default=None):
    raw = live_text(path)
    try:
        return json.loads(raw) if raw.strip() else ({} if default is None else default)
    except Exception:  # noqa: BLE001
        return {} if default is None else default


def live_names(folder):
    """File paths directly inside <folder> on branch live, e.g. live_names('live') -> ['live/bot1_state.json', ...]."""
    folder = folder.rstrip("/")
    try:
        if _git():
            _fetch()
            return subprocess.run(["git", "ls-tree", "--name-only", "origin/live", folder + "/"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.split()
        return [x["path"] for x in (_api(folder) or []) if x.get("type") == "file"]
    except Exception:  # noqa: BLE001
        return []
