"""Classifier agent: reads each headline and scores it for one ticker.

With ANTHROPIC_API_KEY set it asks Claude; otherwise it falls back to a simple keyword
model (good enough to test the pipeline, not good enough to trade on).
Results are cached so the same headline is never paid for twice.
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import config

CATEGORIES = ["earnings", "guidance", "m&a", "fda/clinical", "analyst", "contract/product",
              "legal/regulatory", "management", "offering/dilution", "macro", "other"]

PROMPT = """You are a buy-side news analyst. Judge the likely SHORT-TERM (minutes to 1 hour)
price reaction of {symbol} to this news item, as it would be read the moment it is published.

Headline: {headline}
Summary: {summary}

Reply with JSON only:
{{"score": <float -1..1, negative = price likely falls, positive = rises, 0 = no effect>,
 "confidence": <float 0..1>,
 "category": <one of {categories}>,
 "new_information": <true if this is genuinely new, false if a recap/opinion/old news>}}"""

_cache_lock = threading.Lock()
_CACHE_FILE = config.CACHE_DIR / "classifications.jsonl"


def _load_cache():
    cache = {}
    if _CACHE_FILE.exists():
        for line in _CACHE_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["result"]
    return cache


def _save(key, result):
    with _cache_lock, open(_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "result": result}) + "\n")


def _ask_claude(symbol, headline, summary):
    body = {
        "model": config.CLAUDE_MODEL,
        "max_tokens": 150,
        "messages": [{"role": "user", "content": PROMPT.format(
            symbol=symbol, headline=headline, summary=(summary or "")[:600],
            categories=", ".join(CATEGORIES))}],
    }
    for attempt in range(8):
        t0 = time.time()
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=60,
                              headers={"x-api-key": config.ANTHROPIC_KEY,
                                       "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"})
        except (requests.ConnectionError, requests.Timeout):
            # Network blips happen on long runs; wait and try again (up to ~4 minutes).
            time.sleep(min(60, 2 ** attempt * 2))
            continue
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(min(60, 2 ** attempt * 2))
            continue
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        meta = {"input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "latency_ms": int((time.time() - t0) * 1000)}
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            out = json.loads(m.group(0))
        except (AttributeError, ValueError):
            # Unreadable answer: record it as "no opinion" instead of crashing the run.
            return {"score": 0.0, "confidence": 0.0, "category": "other",
                    "new_information": False, "model": config.CLAUDE_MODEL, **meta}
        return {**meta,
            "score": max(-1.0, min(1.0, float(out.get("score", 0)))),
            "confidence": max(0.0, min(1.0, float(out.get("confidence", 0)))),
            "category": out.get("category", "other") if out.get("category") in CATEGORIES else "other",
            "new_information": bool(out.get("new_information", True)),
            "model": config.CLAUDE_MODEL,
        }
    # Still failing after all retries: skip this one item (not cached, so the next run
    # retries it) instead of losing hours of work.
    return None


_POS = r"beat|beats|tops|surpass|raise[sd]? (guidance|outlook)|upgrade|approv|record|wins?|award|" \
       r"acquire[sd]? by|buyout|soar|jump|surge|partnership|exceed|strong|buyback|positive"
_NEG = r"miss|misses|cut[s]? (guidance|outlook)|lower[s]? (guidance|outlook)|downgrade|reject|recall|" \
       r"probe|investigation|lawsuit|plunge|fall|drop|weak|offering|dilut|delay|halt|fraud|negative|layoff"


_CATS = [("earnings", r"estimates|earnings|eps|revenue|quarter"), ("guidance", r"guidance|outlook"),
         ("analyst", r"upgrade|downgrade|price target|analyst"), ("fda/clinical", r"fda|trial|drug"),
         ("offering/dilution", r"offering|dilut"), ("legal/regulatory", r"sec |probe|investigation|lawsuit"),
         ("contract/product", r"contract|recall|product|award")]


def _keyword_category(text):
    return next((c for c, pat in _CATS if re.search(pat, text)), "other")


def _keyword_model(headline, summary):
    text = f"{headline} {summary}".lower()
    cat = _keyword_category(text)
    pos = len(re.findall(_POS, text))
    neg = len(re.findall(_NEG, text))
    if pos == neg:
        return {"score": 0.0, "confidence": 0.2, "category": cat,
                "new_information": True, "model": "keywords"}
    score = (pos - neg) / (pos + neg)
    return {"score": score, "confidence": min(1.0, 0.4 + 0.2 * (pos + neg)),
            "category": cat, "new_information": True, "model": "keywords"}


_USAGE_COLS = ["input_tokens", "output_tokens", "latency_ms"]


def classify_events(events_df, workers=8):
    """Adds score / confidence / category / new_information / model columns."""
    use_claude = bool(config.ANTHROPIC_KEY)
    cache = _load_cache()
    rows = events_df.to_dict("records")
    tag = config.CLAUDE_MODEL if use_claude else "keywords"

    def work(row):
        key = f"{tag}|{row['id']}|{row['symbol']}"
        if key in cache:
            return cache[key]
        if use_claude:
            res = _ask_claude(row["symbol"], row["headline"], row.get("summary", ""))
            if res is None:
                return {"score": 0.0, "confidence": 0.0, "category": "other",
                        "new_information": False, "model": "unclassified"}
        else:
            res = _keyword_model(row["headline"], row.get("summary", ""))
        _save(key, res)
        return res

    print(f"  classifying {len(rows)} events with {'Claude (' + tag + ')' if use_claude else 'keyword fallback'}")
    with ThreadPoolExecutor(max_workers=workers if use_claude else 1) as ex:
        results = list(ex.map(work, rows))
    out = events_df.copy()
    for col in ["score", "confidence", "category", "new_information", "model"]:
        out[col] = [r[col] for r in results]
    for col in _USAGE_COLS:
        out[col] = [r.get(col, 0) for r in results]
    out["from_cache"] = [f"{tag}|{r['id']}|{r['symbol']}" in cache for r in rows]
    return out


def cost_summary(events_df):
    """Real token usage from the API responses, turned into a monthly cost estimate."""
    d = events_df[events_df["model"] != "keywords"]
    if d.empty or d["input_tokens"].sum() == 0:
        return None
    per_call = (d["input_tokens"] * config.CLAUDE_INPUT_USD_PER_MTOK +
                d["output_tokens"] * config.CLAUDE_OUTPUT_USD_PER_MTOK) / 1e6
    months = d["time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    monthly_events = d.groupby(months).size()
    avg_cost = per_call.mean()
    lat = d.loc[d["latency_ms"] > 0, "latency_ms"]
    return {
        "calls": len(d),
        "avg_input_tokens": d["input_tokens"].mean(),
        "avg_output_tokens": d["output_tokens"].mean(),
        "avg_cost_per_call": avg_cost,
        "total_cost": per_call.sum(),
        "events_per_month": monthly_events.mean(),
        "cost_per_month": monthly_events.mean() * avg_cost,
        "median_latency_ms": lat.median() if len(lat) else float("nan"),
        "p90_latency_ms": lat.quantile(0.9) if len(lat) else float("nan"),
    }
