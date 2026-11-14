"""The scout's per-combo results (reports/agents/<bot>/exp/new_*.json) packed into a few compressed shards
(manager, 14.11.2026: the GitHub scout wrote ~85 thousand of these files, 2.8 GB, main's zip grew from 0.58 to 1.35 GB
and the bots' computer's agents could not refresh main for 4+ hours - every code install restarted the refresh before
it ended). Same content, same readers: a single new_*.json file still wins over the pack when it is newer.

  reports/agents/<bot>/exp/pack/index.json     {"n": shards, "docs": {file name: [at, combo, costs, shard]}}
  reports/agents/<bot>/exp/pack/new_<k>.json.xz {file name: doc}

  python -m agents.exp_pack pack [--bot bot1 ...]      # every single new_*.json into the pack (the files removed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARD_RAW = 20 * 1024 * 1024            # about this much JSON per shard before compression
_IDX, _SH = {}, OrderedDict()           # caches: index per folder, the last few shards read


def pack_dir(exp_dir):
    return Path(exp_dir) / "pack"


def _shard_of(name, n):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % n


def index(exp_dir):
    k = str(exp_dir)
    if k not in _IDX:
        try:
            _IDX[k] = json.loads((pack_dir(exp_dir) / "index.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - no pack yet
            _IDX[k] = {"n": 0, "docs": {}}
    return _IDX[k]


def _shard(exp_dir, s):
    key = (str(exp_dir), s)
    if key in _SH:
        _SH.move_to_end(key)
        return _SH[key]
    p = pack_dir(exp_dir) / f"new_{s}.json.xz"
    try:
        d = json.loads(lzma.decompress(p.read_bytes()).decode("utf-8"))
    except Exception:  # noqa: BLE001
        d = {}
    _SH[key] = d
    while len(_SH) > 3:
        _SH.popitem(last=False)
    return d


def meta(exp_dir, name):
    """{"at", "combo", "costs"} of a packed doc (enough for the age / staleness checks), or None."""
    m = index(exp_dir)["docs"].get(name)
    return {"at": m[0], "combo": m[1], "costs": m[2]} if m else None


def packed(exp_dir, name):
    m = index(exp_dir)["docs"].get(name)
    return _shard(exp_dir, m[3]).get(name) if m else None


def _single(exp_dir, name):
    p = Path(exp_dir) / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def read(exp_dir, name):
    """The doc of exp/<name>: the single file or the pack, the newer of the two."""
    a = _single(exp_dir, name)
    m = meta(exp_dir, name)
    if a is not None and (m is None or str(a.get("at") or "") >= str(m["at"] or "")):
        return a
    return packed(exp_dir, name) if m else a


def read_meta(exp_dir, name):
    a = _single(exp_dir, name)
    m = meta(exp_dir, name)
    if a is not None and (m is None or str(a.get("at") or "") >= str(m["at"] or "")):
        return a
    return m or a


def iter_docs(exp_dir, prefix="new_"):
    """(name, doc) of every new_ result, single files and the pack, the newer copy of each. Streamed shard by shard
    (one shard in memory - all of bot 1's results at once would be ~3 GB of Python objects)."""
    exp_dir = Path(exp_dir)
    docs = index(exp_dir)["docs"]
    won = set()                                     # single files newer than (or missing from) the pack
    for f in sorted(exp_dir.glob(f"{prefix}*.json")):
        d = _single(exp_dir, f.name)
        m = docs.get(f.name)
        if d is not None and (m is None or str(d.get("at") or "") >= str(m[0] or "")):
            won.add(f.name)
            yield f.name, d
    by_shard = {}
    for name, m in docs.items():
        if name not in won:
            by_shard.setdefault(m[3], []).append(name)
    for s in sorted(by_shard):
        sh = _shard(exp_dir, s)
        for name in sorted(by_shard[s]):
            if name in sh:
                yield name, sh[name]


def _write_shard(pd_, s, d):
    data = lzma.compress(json.dumps(d, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
                                    default=str).encode("utf-8"), preset=6)
    tmp = pd_ / f"new_{s}.json.xz.tmp"
    tmp.write_bytes(data)
    tmp.replace(pd_ / f"new_{s}.json.xz")


def pack(exp_dir, log=print):
    """Every single new_*.json into the pack, one shard at a time (memory: one shard); only the shards that get new
    docs are written again; a new shard count (the pack grew) writes them all. The packed files are removed."""
    exp_dir = Path(exp_dir)
    singles = sorted(exp_dir.glob("new_*.json"))
    if not singles:
        return 0
    idx = index(exp_dir)
    n_old = int(idx.get("n") or 0)
    pd_ = pack_dir(exp_dir)
    raw = sum(f.stat().st_size for f in singles)
    raw_old = sum((pd_ / f"new_{s}.json.xz").stat().st_size * 5 for s in range(n_old)
                  if (pd_ / f"new_{s}.json.xz").exists())          # about the compression of these files
    n = max(n_old, math.ceil((raw + raw_old) / SHARD_RAW), 1)
    if n_old and n != n_old:
        n = max(n, 2 * n_old)                      # grow seldom: every growth writes every shard again
    groups = {}
    for f in singles:
        groups.setdefault(_shard_of(f.name, n), []).append(f)
    old = {}                                       # old shard -> its names, for a new shard count
    if n != n_old:
        for k, m in idx["docs"].items():
            old.setdefault(_shard_of(k, n), []).append((k, m[3]))
    touched = set(groups) if n == n_old else set(range(n))
    docs = dict(idx["docs"])
    pd_.mkdir(parents=True, exist_ok=True)
    done, new_shards = [], {}
    for s in sorted(touched):
        if n == n_old:
            d = dict(_shard(exp_dir, s))
        else:
            d = {}
            for k, so in old.get(s, []):
                v = _shard(exp_dir, so).get(k)
                if v is not None:
                    d[k] = v
        for f in groups.get(s, []):
            try:
                v = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a half-written file stays as it is
                continue
            cur = d.get(f.name)
            if cur is None or str(v.get("at") or "") >= str(cur.get("at") or ""):
                d[f.name] = v
            done.append(f)
        for k, v in d.items():
            docs[k] = [v.get("at"), v.get("combo"), v.get("costs"), s]
        if n == n_old:
            _write_shard(pd_, s, d)
            _SH.pop((str(exp_dir), s), None)
        else:                                      # old shards are still read: written under a temporary name
            data = lzma.compress(json.dumps(d, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
                                            default=str).encode("utf-8"), preset=6)
            (pd_ / f"new_{s}.json.xz.next").write_bytes(data)
            new_shards[s] = True
    for s in new_shards:
        (pd_ / f"new_{s}.json.xz.next").replace(pd_ / f"new_{s}.json.xz")
    (pd_ / "index.json").write_text(json.dumps({"n": n, "docs": dict(sorted(docs.items()))}, ensure_ascii=False,
                                               separators=(",", ":")), encoding="utf-8")
    _IDX.pop(str(exp_dir), None)
    for key in [k for k in _SH if k[0] == str(exp_dir)]:
        _SH.pop(key, None)
    for f in done:
        f.unlink()
    log(f"{exp_dir}: {len(done)} files packed, {n} shards ({len(touched)} written), {len(docs)} docs")
    return len(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pack"])
    ap.add_argument("--bot", nargs="*")
    a = ap.parse_args()
    base = ROOT / "reports" / "agents"
    bots = a.bot or sorted(p.name for p in base.iterdir() if (p / "exp").is_dir())
    for b in bots:
        pack(base / b / "exp")


if __name__ == "__main__":
    main()
