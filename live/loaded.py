"""Which of the repository's files a running bot really uses (19.11.2026: every change to any agents/ file - a study,
a report - restarted all the bots, bot 5 frozen ~10 minutes each time). Every process writes the files it has loaded
to live/loaded_<name>.json a few minutes after it starts and every hour after; the updater (touches_bot) restarts the
bots only when one of those files, or the bots' own folder and settings, changed."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def files_now():
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            p = Path(f).resolve().relative_to(ROOT)
        except ValueError:
            continue
        out.add(p.as_posix())
    return sorted(out)


def record(name, first=240, every=3600):
    def run():
        time.sleep(first)
        while True:
            try:
                (ROOT / "live" / f"loaded_{name}.json").write_text(json.dumps(files_now()), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
            time.sleep(every)
    threading.Thread(target=run, daemon=True, name="loaded").start()


def used():
    """The union of what every bot recorded, or None when nothing is recorded yet (then every code file counts)."""
    got = set()
    for f in (ROOT / "live").glob("loaded_*.json"):
        try:
            got |= set(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return got or None
