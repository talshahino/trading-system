"""Offering agent: a second, more detailed read of every offering/dilution headline.

The rule that survived the clean backtest trades offering news. Offerings differ a lot:
a deeply discounted registered direct with warrants in a small biotech is not the same
event as a shelf filing or a convertible note by a large company. Claude extracts the
details, and the report shows where the rule actually earns its money.

Only offering/dilution events are sent (about 1-2% of all news), so this costs ~$1.
Results are cached like the main classification.
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import config

TYPES = ["priced_offering", "registered_direct", "atm_program", "shelf_filing", "convertible_debt",
         "private_placement", "holder_secondary", "withdrawn_or_terminated", "buyback_or_other", "unclear"]
TYPES_HE = {"priced_offering": "הנפקה מתומחרת", "registered_direct": "Registered Direct",
            "atm_program": "תוכנית ATM", "shelf_filing": "תשקיף מדף (עוד לא הנפקה)",
            "convertible_debt": "אג״ח להמרה", "private_placement": "הקצאה פרטית (PIPE)",
            "holder_secondary": "מכירה של בעלי מניות קיימים", "withdrawn_or_terminated": "בוטלה / נמשכה",
            "buyback_or_other": "רכישה עצמית / אחר", "unclear": "לא ברור"}

PROMPT = """This news item about {symbol} was tagged as an equity offering / dilution event.
Extract the details. Use null when the text does not say.

Headline: {headline}
Summary: {summary}

Reply with JSON only:
{{"offering_type": <one of {types}>,
 "discount_pct": <number: % discount of the offering price to the last close, or null>,
 "warrants": <true if warrants are included, else false>,
 "size_pct_of_market_cap": <number: offering size as % of the company's market value, or null>,
 "stage": <one of "announced", "priced", "closed", "filing", "cancelled">}}"""

_FILE = config.CACHE_DIR / "offering_details.jsonl"
_lock = threading.Lock()


def _load():
    out = {}
    if _FILE.exists():
        for line in _FILE.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec["result"]
            except (ValueError, KeyError):
                continue
    return out


def _ask(symbol, headline, summary):
    body = {"model": config.CLAUDE_MODEL, "max_tokens": 150,
            "messages": [{"role": "user", "content": PROMPT.format(
                symbol=symbol, headline=headline, summary=(summary or "")[:800], types=", ".join(TYPES))}]}
    headers = {"x-api-key": config.ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    for attempt in range(6):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(60, 2 ** attempt * 2))
            continue
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(min(60, 2 ** attempt * 2))
            continue
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            out = json.loads(m.group(0))
        except ValueError:
            return None
        t = out.get("offering_type")
        num = lambda v: float(v) if isinstance(v, (int, float)) else None
        return {"offering_type": t if t in TYPES else "unclear",
                "discount_pct": num(out.get("discount_pct")),
                "warrants": bool(out.get("warrants")),
                "size_pct_of_market_cap": num(out.get("size_pct_of_market_cap")),
                "stage": str(out.get("stage") or "")}
    return None


def enrich(events, workers=8):
    """Adds offering_type / discount_pct / warrants / size_pct_of_market_cap / stage to
    offering-category rows (other rows get NaN)."""
    out = events.copy()
    cols = ["offering_type", "discount_pct", "warrants", "size_pct_of_market_cap", "stage"]
    for c in cols:
        out[c] = None
    if not config.ANTHROPIC_KEY or "category" not in out:
        return out
    idx = out.index[out["category"] == "offering/dilution"]
    if len(idx) == 0:
        return out
    cache = _load()
    key = lambda r: f"{config.CLAUDE_MODEL}|offer|{r['id']}|{r['symbol']}"
    rows = [(i, out.loc[i]) for i in idx]
    todo = [(i, r) for i, r in rows if key(r) not in cache]
    print(f"  offering details: {len(rows)} offering events, {len(todo)} new for Claude")

    def work(item):
        i, r = item
        res = _ask(r["symbol"], r["headline"], r.get("summary", ""))
        if res is not None:
            with _lock, open(_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key(r), "result": res}) + "\n")
        return key(r), res

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for k, res in ex.map(work, todo):
                if res is not None:
                    cache[k] = res
    for i, r in rows:
        res = cache.get(key(r))
        if res:
            for c in cols:
                out.at[i, c] = res.get(c)
    return out
