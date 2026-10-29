"""Insider agent: open-market purchases by company insiders, from the SEC's free quarterly
Form 3/4/5 data sets (https://www.sec.gov/dera/data/form-345).

Why this signal: insiders sell for many reasons (taxes, a house, diversification) but buy
with their own money on the open market for one reason. Several insiders buying the same
stock within a few weeks ("cluster buying") is one of the best documented effects in the
academic literature. The data is structured, so no AI reads it and there is no model
"memory" problem: the backtest can use ten years.

Only purchases (transaction code P, acquired) on Form 4 are kept. Each quarter is reduced
to a small CSV of purchases and cached; the raw zip is not kept.

Point in time: a signal exists only from the FILING date (when the public could see it),
never from the trade date. Entry is the next session's open after the filing date.
"""
import io
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

import config

BASES = ["https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets",
         "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets"]
NY = "America/New_York"


def _ua():
    if not config.SEC_CONTACT_EMAIL:
        raise RuntimeError("Missing SEC_CONTACT_EMAIL: the SEC requires a contact email for automated "
                           "downloads. Add it as a repository secret named SEC_CONTACT_EMAIL.")
    return {"User-Agent": f"trading-system research {config.SEC_CONTACT_EMAIL}",
            "Accept-Encoding": "gzip, deflate"}


def _download(q):
    """Raw zip bytes for a quarter like '2024q1', or None if the SEC has not published it."""
    last = None
    for base in BASES:
        url = f"{base}/{q}_form345.zip"
        for attempt in range(5):
            config.check_time_budget()
            try:
                r = requests.get(url, headers=_ua(), timeout=180)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 404:
                break
            if r.status_code in (403, 429, 500, 502, 503):
                last = f"HTTP {r.status_code}"
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            time.sleep(0.5)  # stay far below the SEC's 10 requests/second
            return r.content
    if isinstance(last, str) and "403" in last:
        raise RuntimeError(f"SEC refused the download of {q} (403). Check SEC_CONTACT_EMAIL.")
    return None


def _table(zf, name, cols):
    member = next((m for m in zf.namelist()
                   if re.sub(r"\.[a-z]+$", "", m.split("/")[-1], flags=re.I).upper() == name), None)
    if member is None:
        raise RuntimeError(f"{name} not found in the SEC zip (files: {zf.namelist()[:10]})")
    with zf.open(member) as f:
        df = pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c.upper() in cols,
                         on_bad_lines="skip", quoting=3, encoding="utf-8", encoding_errors="replace")
    df.columns = [c.upper() for c in df.columns]
    return df


def parse_quarter(raw):
    """One row per open-market purchase on an original Form 4."""
    zf = zipfile.ZipFile(io.BytesIO(raw))
    sub = _table(zf, "SUBMISSION", {"ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                                    "ISSUERCIK", "ISSUERTRADINGSYMBOL"})
    own = _table(zf, "REPORTINGOWNER", {"ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP",
                                        "RPTOWNER_TITLE"})
    tr = _table(zf, "NONDERIV_TRANS", {"ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                       "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
                                       "SHRS_OWND_FOLWNG_TRANS"})
    tr = tr[(tr["TRANS_CODE"].str.strip() == "P") & (tr["TRANS_ACQUIRED_DISP_CD"].str.strip() == "A")]
    sub = sub[sub["DOCUMENT_TYPE"].str.strip() == "4"]          # originals only, no amendments
    for c in ("TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    # A filing can list several owners (joint filings): one owner per filing, roles combined.
    own["REL"] = own["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()
    own["TITLE"] = own["RPTOWNER_TITLE"].fillna("").str.lower()
    own = own.groupby("ACCESSION_NUMBER").agg(owner=("RPTOWNERCIK", "first"), rel=("REL", " ".join),
                                              title=("TITLE", " ".join)).reset_index()
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"],
        "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
        "trans_date": pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
        "symbol": df["ISSUERTRADINGSYMBOL"].fillna("").str.upper().str.strip(),
        "issuer_cik": df["ISSUERCIK"],
        "owner": df["owner"].fillna(df["ACCESSION_NUMBER"]),
        "shares": df["TRANS_SHARES"], "price": df["TRANS_PRICEPERSHARE"],
        "owned_after": df["SHRS_OWND_FOLWNG_TRANS"],
        "officer": df["rel"].fillna("").str.contains("OFFICER"),
        "director": df["rel"].fillna("").str.contains("DIRECTOR"),
        "ten_pct": df["rel"].fillna("").str.contains("TEN"),
        "ceo_cfo": df["title"].fillna("").str.contains(r"chief executive|\bceo\b|chief financial|\bcfo\b|president"),
    })
    out["value"] = out["shares"] * out["price"]
    return out.dropna(subset=["filing_date", "value"])


def fetch_purchases(start, end):
    """All Form 4 open-market purchases filed between two dates (cached per quarter)."""
    qs = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="Q")
    now_q = pd.Timestamp.now().to_period("Q")
    frames = []
    for q in qs:
        tag = f"{q.year}q{q.quarter}"
        f = config.CACHE_DIR / f"insider_purchases_{tag}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, parse_dates=["filing_date", "trans_date"])
            print(f"  insider {tag}: {len(df)} purchases (cached)")
        else:
            raw = _download(tag)
            if raw is None:
                print(f"  insider {tag}: not published yet")
                continue
            df = parse_quarter(raw)
            if q < now_q:
                df.to_csv(f, index=False)
            print(f"  insider {tag}: {len(df)} purchases (downloaded)")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["filing_date"] >= pd.Timestamp(start)) & (df["filing_date"] <= pd.Timestamp(end))]
    return df


def build_events(purchases):
    """One event per (stock, filing date) with the 30-day cluster as known on that date.

    Mapped onto the backtest's columns so the same rule search, luck test and reports work:
      score      = how many different insiders bought in the last 30 days
                   (1 -> 0.3, 2 -> 0.5, 3 -> 0.7, 4+ -> 0.85; so the threshold grid means
                   "at least 1 / 2 / 3 / 4 insiders"). Always positive: we only buy.
      confidence = total bought in those 30 days (< $100K -> 0.3, < $500K -> 0.7, more -> 0.9;
                   so the grid means "any / $100K+ / $500K+").
      new_information = someone raised their holding by 10% or more (a real bet, not a token).
      category   = the most senior buyer: ceo/cfo, officer, director, 10% owner.
    """
    p = purchases.copy()
    p = p[p["value"] >= config.INSIDER_MIN_VALUE]
    p = p[p["symbol"].str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")]
    p = p[~p["symbol"].isin(config.ETFS)]
    before = p["owned_after"] - p["shares"]
    p["pct_increase"] = np.where(before > 0, p["shares"] / before, 1.0)
    p = p.sort_values("filing_date")
    rows = []
    win = pd.Timedelta(days=config.INSIDER_CLUSTER_DAYS)
    for sym, g in p.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        dates = g["filing_date"].to_numpy()
        for d in np.unique(dates):
            w = g[(g["filing_date"] <= d) & (g["filing_date"] > d - win)]
            today = g[g["filing_date"] == d]
            n = w["owner"].nunique()
            value = w["value"].sum()
            role = ("ceo/cfo" if w["ceo_cfo"].any() else "officer" if w["officer"].any()
                    else "director" if w["director"].any() else "10% owner/other")
            rows.append({
                "id": f"{sym}-{pd.Timestamp(d).date()}", "symbol": sym,
                # Filing date, after the close: the first possible entry is the next open.
                "time": pd.Timestamp(d).tz_localize(NY) + pd.Timedelta(hours=16),
                "headline": f"{n} insider(s) bought ${value:,.0f} in {config.INSIDER_CLUSTER_DAYS} days",
                "n_insiders": n, "cluster_value": value, "value_today": today["value"].sum(),
                "max_pct_increase": w["pct_increase"].max(),
                "score": {1: 0.3, 2: 0.5, 3: 0.7}.get(n, 0.85),
                "confidence": 0.3 if value < 1e5 else 0.7 if value < 5e5 else 0.9,
                "new_information": bool(w["pct_increase"].max() >= 0.10),
                "category": role, "model": "rules",
            })
    ev = pd.DataFrame(rows)
    if not ev.empty:
        ev["time"] = ev["time"].dt.tz_convert("UTC")
        ev["year"] = ev["time"].dt.year
    return ev


N_LABEL = {0.3: "1", 0.5: "2", 0.7: "3", 0.85: "4+"}
VALUE_LABEL = {0.0: "כל סכום", 0.6: "$100K ומעלה", 0.8: "$500K ומעלה"}
ROLE_HE = {"ceo/cfo": "מנכ״ל/סמנכ״ל כספים", "officer": "נושא משרה", "director": "דירקטור",
           "10% owner/other": "בעל 10%+ / אחר"}


def rule_text(rule):
    """Hebrew description of a rule in insider terms."""
    n = N_LABEL.get(rule["threshold"], rule["threshold"])
    who = "לפחות מנהל אחד קנה" if n == "1" else f"לפחות {n} מנהלים שונים קנו"
    parts = [f"{who} ב־{config.INSIDER_CLUSTER_DAYS} הימים האחרונים",
             f"סכום כולל: {VALUE_LABEL.get(rule['min_conf'], rule['min_conf'])}"]
    if rule.get("require_new"):
        parts.append("לפחות אחד מהם הגדיל את ההחזקה שלו ב־10% ומעלה")
    if rule.get("category"):
        parts.append(f"הקונה הבכיר ביותר: {ROLE_HE.get(rule['category'], rule['category'])}")
    return ", ".join(parts)
