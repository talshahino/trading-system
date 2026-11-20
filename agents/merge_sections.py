"""Files written by two writers (the bots' computer and the nightly smart agents in the cloud) whose top-level sections
each carry their own "updated" time: merged section by section, the newer "updated" wins, a section only one side has
is kept (manager, 20.11.2026 night: data/agents/strategies/tal_gap_results.json - the bots' computer published bot 5's
run of 23:13 UTC at 23:37, the cloud's bot 6 smart layer committed its older whole copy at 23:47 and bot 5's section
went back to 14:29; the computer's publish had just as well put the cloud's bot5_smart of 22:22 back to 18:56).

    python -m agents.merge_sections <file> [<ref>]   merge the working copy with <ref>'s copy (default origin/main)
                                                     before committing it - run it in the cloud before every commit of
                                                     a file in SECTIONED
live/agents_host.publish merges with main's copy the same way (merge_bytes).
"""
from __future__ import annotations

import json
import subprocess
import sys

SECTIONED = ("data/agents/strategies/tal_gap_results.json",)


def _upd(v):
    return str(v.get("updated") or "") if isinstance(v, dict) else ""


def merge(main, own):
    """own over main per top-level key, unless main's section has a newer 'updated'."""
    out = dict(main)
    for k, v in own.items():
        if k in main and _upd(main[k]) > _upd(v):
            continue
        out[k] = v
    return out


def merge_bytes(main_b, own_b):
    m, o = json.loads(main_b.decode("utf-8")), json.loads(own_b.decode("utf-8"))
    if not (isinstance(m, dict) and isinstance(o, dict)):
        return own_b
    return json.dumps(merge(m, o), ensure_ascii=False, indent=1).encode("utf-8")


def main(argv):
    f = argv[0]
    ref = argv[1] if len(argv) > 1 else "origin/main"
    try:
        mb = subprocess.run(["git", "show", f"{ref}:{f}"], capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        print(f"{f}: not on {ref}, left as it is")
        return
    with open(f, "rb") as fh:
        ob = fh.read()
    m, o = json.loads(mb.decode("utf-8")), json.loads(ob.decode("utf-8"))
    took = [k for k in m if k in o and _upd(m[k]) > _upd(o[k])] + [k for k in m if k not in o]
    with open(f, "wb") as fh:
        fh.write(json.dumps(merge(m, o), ensure_ascii=False, indent=1).encode("utf-8"))
    print(f"{f}: merged with {ref}; newer there: {took or 'nothing'}")


if __name__ == "__main__":
    main(sys.argv[1:])
