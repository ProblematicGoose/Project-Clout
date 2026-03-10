import time
import pandas as pd
import urllib.parse
import urllib.request
import json
from datetime import datetime, timedelta, date
from dash.exceptions import PreventUpdate
from flask import session, request
import concurrent.futures
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import os
from concurrent.futures import TimeoutError as FuturesTimeoutError
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)


# -----------------------------
# Config / Endpoints
# -----------------------------
BASE_URL = "https://sentiment-dashboard.ngrok.app"
SCORECARD_URL = f"{BASE_URL}/api/scorecard"
PHOTOS_URL = f"{BASE_URL}/api/subject-photos"
TIMESERIES_URL = f"{BASE_URL}/api/timeseries"
TRAITS_URL = f"{BASE_URL}/api/traits"
BILL_SENTIMENT_URL = f"{BASE_URL}/api/bill-sentiment"
TOP_ISSUES_URL = f"{BASE_URL}/api/top-issues"
COMMON_GROUND_URL = f"{BASE_URL}/api/common-ground-issues"
MENTION_COUNT_URL = f"{BASE_URL}/api/mention-counts"
MOMENTUM_URL = f"{BASE_URL}/api/momentum"
MENTION_COUNTS_DAILY_URL = f"{BASE_URL}/api/mention-counts-daily"
LATEST_COMMENTS_URL = f"{BASE_URL}/api/latest-comments"  # NEW
CONSTITUENT_ASKS_URL = f"{BASE_URL}/api/constituent-asks"
WEEKLY_STRATEGY_URL = f"{BASE_URL}/api/weekly-strategy"
SUBJECT_BUNDLE_URL = f"{BASE_URL}/api/subject-bundle"
SUBJECTS_URL = f"{BASE_URL}/api/subjects"
ELECTION_OUTLOOK_URL = f"{BASE_URL}/api/election-outlook"

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mssql+pyodbc://sa:Is2*2set@3.tcp.ngrok.io,27671/Clout?driver=ODBC+Driver+17+for+SQL+Server"
)

# Create the engine
engine = create_engine(DATABASE_URL)


# -----------------------------
# SQL fallback / augmentation: Truth Social latest comments
# -----------------------------
def _get_table_columns(table_name: str, schema: str = "dbo") -> set[str]:
    """Return a set of column names for a table (best-effort)."""
    try:
        q = text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :t
        """)
        with engine.begin() as conn:
            rows = conn.execute(q, {"schema": schema, "t": table_name}).fetchall()
        return {r[0] for r in rows}
    except Exception as e:
        print(f"[_get_table_columns] error for {schema}.{table_name}: {e}")
        return set()

def fetch_truthsocial_latest_comments_df(subject: str | None = None, limit: int = 10) -> pd.DataFrame:
    """Fetch latest Truth Social comments from dbo.CombinedTruthSocialComments (best-effort)."""
    cols = _get_table_columns("CombinedTruthSocialComments", schema="dbo")
    if not cols:
        return pd.DataFrame(columns=["Source", "Comment", "CreatedUTC", "URL"])

    def pick(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in cols:
                return c
        return None

    subject_col = pick(["Subject", "FullName", "Name", "ElectedOfficial"])
    comment_col = pick(["Comment", "CommentText", "Text", "Body", "Content", "Message"])
    created_col = pick(["CreatedUTC", "CreatedAtUTC", "CreatedAt", "Timestamp", "Created", "CreatedDateUTC"])
    url_col     = pick(["URL", "Permalink", "Link", "CommentURL", "PostURL", "PostLink"])

    # If we can't find the minimum fields, bail cleanly
    if not comment_col or not created_col:
        return pd.DataFrame(columns=["Source", "Comment", "CreatedUTC", "URL"])

    # Build a safe SELECT with the columns we found
    where = ""
    params = {"n": int(limit)}
    if subject and subject_col:
        where = f"WHERE {subject_col} = :subject"
        params["subject"] = str(subject)

    sql = f"""
        SELECT TOP (:n)
            'TruthSocial' AS Source,
            CAST({comment_col} AS NVARCHAR(MAX)) AS Comment,
            {created_col} AS CreatedUTC
            {',' if url_col else ''} {url_col + ' AS URL' if url_col else ''}
        FROM dbo.CombinedTruthSocialComments
        {where}
        ORDER BY {created_col} DESC
    """

    try:
        with engine.begin() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        print(f"[fetch_truthsocial_latest_comments_df] error: {e}")
        return pd.DataFrame(columns=["Source", "Comment", "CreatedUTC", "URL"])

    # Ensure expected columns exist
    for c in ["Source", "Comment", "CreatedUTC", "URL"]:
        if c not in df.columns:
            df[c] = None
    return df[["Source", "Comment", "CreatedUTC", "URL"]]

def fetch_latest_comments_combined(subject: str | None, limit: int = 10, timeout: int = 4) -> pd.DataFrame:
    """Combine /api/latest-comments with Truth Social table (CombinedTruthSocialComments)."""
    params = {"limit": int(limit)}
    if subject:
        params["subject"] = subject

    api_df = fetch_df_with_params(LATEST_COMMENTS_URL, params, timeout=timeout)
    ts_df  = fetch_truthsocial_latest_comments_df(subject=subject, limit=limit)

    # Normalize API columns to match
    if not api_df.empty:
        if "Timestamp" in api_df.columns and "CreatedUTC" not in api_df.columns:
            api_df["CreatedUTC"] = api_df["Timestamp"]
        if "CreatedUTC" not in api_df.columns and "CreatedAt" in api_df.columns:
            api_df["CreatedUTC"] = api_df["CreatedAt"]
        if "Comment" not in api_df.columns:
            for c in ("Text", "Body", "Content", "Message"):
                if c in api_df.columns:
                    api_df["Comment"] = api_df[c]
                    break
        if "URL" not in api_df.columns:
            for c in ("Link", "Permalink", "CommentURL", "PostURL"):
                if c in api_df.columns:
                    api_df["URL"] = api_df[c]
                    break
        if "Source" not in api_df.columns:
            api_df["Source"] = "Unknown"

        api_df = api_df.copy()
        for c in ["Source", "Comment", "CreatedUTC", "URL"]:
            if c not in api_df.columns:
                api_df[c] = None
        api_df = api_df[["Source", "Comment", "CreatedUTC", "URL"]]

    combined = pd.concat([api_df, ts_df], ignore_index=True) if (not api_df.empty or not ts_df.empty) else pd.DataFrame(columns=["Source","Comment","CreatedUTC","URL"])

    # Sort by timestamp desc (best-effort) and trim to limit
    if not combined.empty:
        combined["_ts"] = pd.to_datetime(combined["CreatedUTC"], errors="coerce")
        combined = combined.sort_values("_ts", ascending=False).drop(columns=["_ts"])
        combined = combined.head(int(limit))

    return combined


from flask_sql_api import flask_app  # your shared Flask app

import dash
app = dash.Dash(
    __name__,
    server=flask_app,
    suppress_callback_exceptions=True,  # ← allow callbacks to target components added later
)
server = flask_app



# -----------------------------
# Lightweight in-memory TTL cache (speeds up subject switches)
# -----------------------------
_TTL_SECONDS = 300  # adjust if you want longer/shorter freshness
_CACHE: dict[str, tuple[float, object]] = {}

from flask import session

def current_user_id() -> str | None:
    if "user_id" in session:
        return session["user_id"]
    
    # Read from WordPress-set cookie
    user_cookie = request.cookies.get("clout_user")
    if user_cookie:
        session["user_id"] = user_cookie
        return user_cookie

    return None
    
def _cache_get(key: str):
    rec = _CACHE.get(key)
    if not rec:
        return None
    ts, value = rec
    if time.time() - ts < _TTL_SECONDS:
        return value
    return None

def _cache_set(key: str, value: object):
    _CACHE[key] = (time.time(), value)
    # Limit cache to 100 entries max
    if len(_CACHE) > 100:
        oldest = sorted(_CACHE.items(), key=lambda kv: kv[1][0])[:10]
        for k, _ in oldest:
            _CACHE.pop(k, None)

# -----------------------------
# Helpers
# -----------------------------
def fetch_subject_bundle(subject: str, start_date: str | None = None, end_date: str | None = None, timeout: int = 8) -> dict:
    if not subject:
        return {}

    subject_norm = str(subject).strip()
    params = {"subject": subject_norm}
    if start_date: params["start_date"] = str(start_date)
    if end_date:   params["end_date"]   = str(end_date)

    query = urllib.parse.urlencode(params)
    url = f"{SUBJECT_BUNDLE_URL}?{query}"
    print(f"[bundle] GET {url}")

    cache_key = f"BUNDLE::{subject_norm}::{params.get('start_date','')}::{params.get('end_date','')}"
    cached = _cache_get(cache_key)
    if isinstance(cached, dict):
        return cached

    # 1) Try the real bundle endpoint
    data = fetch_json(url, timeout=timeout)
    if data:
        _cache_set(cache_key, data)   # ← cache server bundle
        return data

    # 2) Fallback: assemble bundle from existing endpoints
    try:
        ts_df = fetch_timeseries_df(subject_norm, params.get("start_date"), params.get("end_date"))
        mo_df = fetch_momentum_df(subject_norm, params.get("start_date"), params.get("end_date"))
        ws_df = fetch_df_with_params(WEEKLY_STRATEGY_URL, {"subjects": subject_norm, "latest": "1"}, timeout=timeout)
        asks_df = fetch_df_with_params(CONSTITUENT_ASKS_URL, {"subjects": subject_norm, "top_n": 5, "latest": "1"}, timeout=timeout)
        photos_df = fetch_df(PHOTOS_URL)
        comments_df = fetch_latest_comments_combined(subject_norm, limit=5, timeout=timeout)

        bundle = {
            "timeseries": ([] if ts_df.empty else [
                {"Date": pd.to_datetime(r["SentimentDate"]).date().isoformat(),
                 "Score": int(r["NormalizedSentimentScore"])}
                for _, r in ts_df.iterrows()
            ]),
            "momentum": ([] if mo_df.empty else [
                {"Date": pd.to_datetime(r["ActivityDate"]).date().isoformat(),
                 "MentionCount": int((r["MentionCount"] or 0)),
                 "AvgSentiment": float((r["AvgSentiment"] or 0.0))}
                for _, r in mo_df.iterrows()
            ]),
            "strategy": (ws_df.iloc[0].to_dict() if not ws_df.empty else None),
            "asks": ([] if asks_df.empty else asks_df.to_dict(orient="records")),
            "photos": ([] if photos_df.empty else photos_df[photos_df["Subject"].astype(str).eq(subject_norm)].head(3).to_dict(orient="records")),
            "latest_comments": ([] if comments_df.empty else comments_df.to_dict(orient="records")),
        }

        _cache_set(cache_key, bundle)  # ← cache fallback bundle
        return bundle
    except Exception as e:
        print(f"[bundle:fallback] error: {e}")
        return {}


# -----------------------------
# Helpers (resilient caching)
# -----------------------------

def fetch_df(url: str, timeout: int = 6) -> pd.DataFrame:
    """Fetch JSON → DataFrame using background thread caching."""
    fresh_key = f"DF::{url}"
    cached = _cache_get(fresh_key)
    if isinstance(cached, pd.DataFrame):
        return cached

    def _fetch():
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                df = pd.DataFrame(json.load(r))
                _cache_set(fresh_key, df)
                return df
        except Exception as e:
            print(f"[fetch_df] inner error for {url}: {e}")
            stale = _CACHE.get(fresh_key)
            if isinstance(stale, tuple) and isinstance(stale[1], pd.DataFrame):
                return stale[1]
            return pd.DataFrame()

    future = _executor.submit(_fetch)
    try:
        return future.result(timeout=timeout + 2)
    except FuturesTimeoutError:
        print(f"[fetch_df] TIMEOUT for {url}")
        stale = _CACHE.get(fresh_key)
        if isinstance(stale, tuple) and isinstance(stale[1], pd.DataFrame):
            return stale[1]
        return pd.DataFrame()
    except Exception as e:
        print(f"[fetch_df] outer error for {url}: {e}")
        stale = _CACHE.get(fresh_key)
        if isinstance(stale, tuple) and isinstance(stale[1], pd.DataFrame):
            return stale[1]
        return pd.DataFrame()

    # submit async
    future = _executor.submit(_fetch)
    return future.result(timeout=timeout + 2)


def fetch_json(url: str, timeout: int = 6) -> dict:
    """Fetch JSON dict with caching and background I/O."""
    fresh_key = f"JSON::{url}"
    cached = _cache_get(fresh_key)
    if isinstance(cached, dict):
        return cached

    def _fetch():
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = json.load(r)
                _cache_set(fresh_key, data)
                return data
        except Exception as e:
            print(f"[fetch_json] inner error for {url}: {e}")
            stale = _CACHE.get(fresh_key)
            if isinstance(stale, tuple) and isinstance(stale[1], dict):
                return stale[1]
            return {}

    future = _executor.submit(_fetch)
    try:
        return future.result(timeout=timeout + 2)
    except FuturesTimeoutError:
        print(f"[fetch_json] TIMEOUT for {url}")
        stale = _CACHE.get(fresh_key)
        if isinstance(stale, tuple) and isinstance(stale[1], dict):
            return stale[1]
        return {}
    except Exception as e:
        print(f"[fetch_json] outer error for {url}: {e}")
        stale = _CACHE.get(fresh_key)
        if isinstance(stale, tuple) and isinstance(stale[1], dict):
            return stale[1]
        return {}

    future = _executor.submit(_fetch)
    return future.result(timeout=timeout + 2)


def fetch_df_with_params(base_url: str, params: dict, timeout: int = 6) -> pd.DataFrame:
    """Cached GET with query parameters and timeout control."""
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{base_url}?{query}" if query else base_url
    return fetch_df(url, timeout=timeout)



def start_of_today() -> datetime:
    now = datetime.now()
    return datetime(year=now.year, month=now.month, day=now.day)




# --- MOMENTUM MATH (helper) ---
import numpy as np

def compute_momentum(df: pd.DataFrame, ema_fast: int = 7, ema_slow: int = 21, dead_zone: float = 0.05) -> pd.DataFrame:
    """
    Input: df with columns ['Subject', 'ActivityDate', 'MentionCount', 'AvgSentiment'] (daily aggregates).
    Steps:
      1) Dead-zone tiny sentiment values to 0 (|x| < dead_zone).
      2) base_t = log1p(mentions_t) * sentiment_t
      3) momentum = EMA( base_t, span=ema_fast )
      4) z-score momentum vs rolling 90d per subject
      5) accel = EMA_fast - EMA_slow (MACD-style)
    Output: original columns + ['base','ema_fast','ema_slow','momentum','z_momentum','accel']
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Subject", "ActivityDate", "MentionCount", "AvgSentiment",
            "base", "ema_fast", "ema_slow", "momentum", "z_momentum", "accel"
        ])

    out = df.copy()
    # Ensure Subject exists (some endpoints omit it)
    if "Subject" not in out.columns:
        out["Subject"] = "(single)"
    # Ensure types and ordering
    out["ActivityDate"] = pd.to_datetime(out["ActivityDate"], errors="coerce")
    out = out.dropna(subset=["ActivityDate"])
    out = out.sort_values(["Subject", "ActivityDate"])

    # Dead-zone tiny sentiment noise
    def _deadzone(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        return 0.0 if (-dead_zone < x < dead_zone) else x

    out["AvgSentiment"] = out["AvgSentiment"].map(_deadzone).astype(float)
    out["MentionCount"] = pd.to_numeric(out["MentionCount"], errors="coerce").fillna(0.0)

    # Base = log1p(mentions) * sentiment
    out["base"] = np.log1p(out["MentionCount"]) * out["AvgSentiment"]

    # Per-subject EMA/z-score/accel
    frames = []
    for subject, g in out.groupby("Subject", sort=False):
        g = g.sort_values("ActivityDate")
        g["ema_fast"] = g["base"].ewm(span=ema_fast, adjust=False).mean()
        g["ema_slow"] = g["base"].ewm(span=ema_slow, adjust=False).mean()
        g["momentum"] = g["ema_fast"]

        roll = g["momentum"].rolling(90, min_periods=15)
        mu = roll.mean()
        sigma = roll.std(ddof=0).replace(0, np.nan)
        g["z_momentum"] = (g["momentum"] - mu) / sigma

        g["accel"] = g["ema_fast"] - g["ema_slow"]
        frames.append(g)

    out = pd.concat(frames, ignore_index=True) if frames else out
    return out



# --- FETCH HELPER (Momentum endpoint) ---
import requests

def fetch_momentum_df(subject: str | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """
    Calls your Flask endpoint /api/momentum and normalizes the response to:
      ['Subject', 'ActivityDate', 'MentionCount', 'AvgSentiment']
    It gracefully handles slight column-name differences.
    """
    # Try to use an existing constant if your app defines it; else build from env/default
    url = None
    try:
        url = MOMENTUM_URL  # if your app already defines this
    except NameError:
        api_base = os.environ.get("API_BASE", "http://localhost:5050")
        url = f"{api_base.rstrip('/')}/api/momentum"

    params = {}
    if start_date: params["start_date"] = start_date
    if end_date:   params["end_date"]   = end_date
    # (API supports optional subject filter; if not, we'll filter client-side)
    if subject:    params["subject"]    = subject

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame(columns=["Subject", "ActivityDate", "MentionCount", "AvgSentiment"])

    # Normalize column names
    # ActivityDate could come as 'ActivityDate', 'Date', or similar
    if "ActivityDate" not in df.columns:
        for c in ("Date", "MomentumDate", "CreatedUTC"):
            if c in df.columns:
                df = df.rename(columns={c: "ActivityDate"})
                break

    # Mention count sometimes capitalized differently
    if "MentionCount" not in df.columns:
        for c in ("Mentions", "Count"):
            if c in df.columns:
                df = df.rename(columns={c: "MentionCount"})
                break

    # Average sentiment sometimes 'AvgSentiment' or 'AverageSentiment'
    if "AvgSentiment" not in df.columns:
        for c in ("AverageSentiment", "Avg_Sentiment", "SentimentAvg"):
            if c in df.columns:
                df = df.rename(columns={c: "AvgSentiment"})
                break

    # Keep only the fields we need; coerce types
    keep = ["Subject", "ActivityDate", "MentionCount", "AvgSentiment"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep].copy()
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], errors="coerce")
    df["MentionCount"] = pd.to_numeric(df["MentionCount"], errors="coerce")
    df["AvgSentiment"] = pd.to_numeric(df["AvgSentiment"], errors="coerce")

    # If API didn’t filter by subject, do it here
    if subject:
        df = df[df["Subject"].astype(str) == str(subject)]

    return df.dropna(subset=["ActivityDate"])

# --- FETCH HELPER (Sentiment Over Time) ---
def fetch_timeseries_df(subject: str, start_ts, end_ts) -> pd.DataFrame:
    """
    Calls /api/timeseries with subject + date range and normalizes columns to:
      ['SentimentDate', 'Subject', 'NormalizedSentimentScore']
    start_ts/end_ts can be pandas Timestamps or 'YYYY-MM-DD' strings.
    """
    # clamp to YYYY-MM-DD for the API
    def _fmt(d):
        if hasattr(d, "date"):
            return str(d.date())
        s = str(d)
        return s[:10]  # 'YYYY-MM-DD'

    params = {
        "subject": subject,
        "start_date": _fmt(start_ts),
        "end_date": _fmt(end_ts),
    }

    # Prefer your existing helper; fall back to requests if it's not defined
    try:
        df = fetch_df_with_params(TIMESERIES_URL, params, timeout=5)  # your app helper
    except NameError:
        r = requests.get(TIMESERIES_URL, params=params, timeout=5)
        r.raise_for_status()
        df = pd.DataFrame(r.json())

    if df.empty:
        return pd.DataFrame(columns=["SentimentDate", "Subject", "NormalizedSentimentScore"])

    # Normalize column names from API variants
    if "SentimentDate" not in df.columns:
        for c in ("Date", "CreatedUTC", "ActivityDate"):
            if c in df.columns:
                df = df.rename(columns={c: "SentimentDate"})
                break
    if "NormalizedSentimentScore" not in df.columns:
        for c in ("Score", "AvgSentiment", "AverageSentiment"):
            if c in df.columns:
                df = df.rename(columns={c: "NormalizedSentimentScore"})
                break

    # Coerce + sort
    df["SentimentDate"] = pd.to_datetime(df["SentimentDate"], errors="coerce")
    df = df.dropna(subset=["SentimentDate"])
    df["NormalizedSentimentScore"] = pd.to_numeric(df["NormalizedSentimentScore"], errors="coerce")
    return df[["SentimentDate", "Subject", "NormalizedSentimentScore"]].sort_values("SentimentDate")


# --- DATE RANGE HELPER (for momentum controls) ---

def date_range_from_mode(mode: str | None, custom_start: str | None, custom_end: str | None):
    """
    Returns (start, end) as pandas Timestamps (UTC-normalized midnight).
    Supports:
      Legacy: "Today", "This Week", "This Month", "This Year", "Custom"
      New:    "7d", "30d", "90d", "custom"
    Defaults to last 30 days on unknown mode or malformed dates.
    """
    m = (mode or "").strip().lower()
    today = pd.Timestamp.utcnow().normalize()

    def _safe_custom():
        try:
            s = pd.to_datetime(custom_start).normalize()
            e = pd.to_datetime(custom_end).normalize()
            if pd.isna(s) or pd.isna(e):
                raise ValueError
            return min(s, e), max(s, e)
        except Exception:
            return today - pd.Timedelta(days=30), today

    # New options
    if m in {"7d", "last7", "last_7_days"}:
        start = today - pd.Timedelta(days=7);  end = today
    elif m in {"30d", "last30", "last_30_days"}:
        start = today - pd.Timedelta(days=30); end = today
    elif m in {"90d", "last90", "last_90_days"}:
        start = today - pd.Timedelta(days=90); end = today
    elif m == "custom":
        start, end = _safe_custom()

    # Legacy options
    elif m == "today":
        start = today; end = today
    elif m in {"this week", "week"}:
        start = today - pd.Timedelta(days=today.weekday())  # Monday-start; tweak if Sunday-start preferred
        end = today
    elif m in {"this month", "month"}:
        start = today.replace(day=1); end = today
    elif m in {"this year", "year"}:
        start = today.replace(month=1, day=1); end = today
    elif m in {"custom (legacy)"}:
        start, end = _safe_custom()
    else:
        # Default
        start = today - pd.Timedelta(days=30); end = today

    if start > end:
        start, end = end, start
    return start, end



def coerce_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_first(series, default=None):
    try:
        v = series.iloc[0]
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default
    

def fetch_election_outlook(subject: str | None, timeout: int = 6) -> dict:
    if not subject:
        return {}
    params = {"subject": str(subject).strip()}
    query = urllib.parse.urlencode(params)
    url = f"{ELECTION_OUTLOOK_URL}?{query}"
    data = fetch_json(url, timeout=timeout)
    return data if isinstance(data, dict) else {}


def party_color(party: str | None) -> str:
    p = (party or "").strip().lower()
    if p == "democrat":
        return "#1f77b4"
    if p == "republican":
        return "#d62728"
    if p == "independent":
        return "#7f7f7f"
    return "#6c757d"


def probability_color(probability: float | int | None) -> str:
    try:
        p = float(probability or 0)
    except Exception:
        p = 0.0
    if p >= 0.40:
        return "#2ca02c"
    if p >= 0.15:
        return "#ff7f0e"
    return "#d62728"


def make_election_gauge(probability: float | int | None, label: str, title: str) -> go.Figure:
    try:
        p = max(0.0, min(1.0, float(probability or 0.0)))
    except Exception:
        p = 0.0

    bar_color = probability_color(p)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100.0,
            domain={"x": [0.00, 0.84], "y": [0.0, 1.0]},  # centers the gauge
            number={"suffix": "%", "font": {"size": 34}},
            title={
                "text": f"{title}<br><span style='font-size:14px;color:#666'>{label}</span>"
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 15], "color": "#fde2e2"},
                    {"range": [15, 40], "color": "#fff1db"},
                    {"range": [40, 100], "color": "#e4f5e8"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 4},
                    "thickness": 0.8,
                    "value": p * 100.0,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=35, r=35, t=70, b=20),
        height=280,
    )
    return fig



def election_outlook_card(payload: dict | None):
    payload = payload or {}
    subject_info = payload.get("Subject") or {}
    default_mode = "general"
    return html.Div(
        [
            dcc.Loading(
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("Election Outlook", className="center-text", style={"marginBottom": "6px"}),
                                html.Div(
                                    [
                                        html.Span(payload.get("RaceType") or "", style={"fontWeight": "600"}),
                                        html.Span(" • "),
                                        html.Span(subject_info.get("Party") or ""),
                                        html.Span(" • "),
                                        html.Span(payload.get("LastUpdated") or ""),
                                    ],
                                    style={"textAlign": "center", "color": "#666", "fontSize": "12px", "marginBottom": "10px"},
                                ),
                            ]
                        ),
                        dcc.RadioItems(
                            id="election-outlook-toggle",
                            options=[
                                {"label": "Nomination Outlook", "value": "nomination"},
                                {"label": "General Election Outlook", "value": "general"},
                            ],
                            value=default_mode,
                            inline=True,
                            style={"textAlign": "center", "marginBottom": "10px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="election-outlook-gauge",
                                            config={"displayModeBar": False},
                                            style={"height": "300px", "width": "100%"},
                                        ),
                                    ],
                                    style={
                                        "flex": "0 0 320px",
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(id="election-outlook-primary"),
                                        html.Details(
                                            [
                                                html.Summary("Race Competitors", style={"cursor": "pointer", "fontWeight": "600", "marginTop": "14px", "marginBottom": "8px"}),
                                                html.Div(id="election-outlook-competitors"),
                                            ],
                                            open=False,
                                            style={"marginTop": "8px"},
                                        ),
                                        html.H4("Path to Victory", style={"marginTop": "14px", "marginBottom": "8px"}),
                                        html.Div(id="election-outlook-path", style={"fontSize": "15px", "lineHeight": "1.45"}),
                                    ],
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                            ],
                            style={"display": "flex", "gap": "18px", "alignItems": "flex-start", "flexWrap": "wrap"},
                        ),
                    ]
                ),
                type="default",
            ),
        ],
        id="election-outlook-card",
        className="dashboard-card",
        style={"gridColumn": "1 / -1", "display": "none"},
    )

 # Radio options for date ranges used by all charts
TIME_MODES = [
    #{"label": "Today",      "value": "Today"},
    #{"label": "This Week",  "value": "This Week"},
    #{"label": "This Month", "value": "This Month"},
    #{"label": "This Year",  "value": "This Year"},
    {"label": "Custom",     "value": "Custom"},
    # If you want quick-shortcuts too, uncomment any of these:
    {"label": "Last 7d",  "value": "7d"},
    {"label": "Last 30d", "value": "30d"},
    {"label": "Last 90d", "value": "90d"},
]   


# -----------------------------
# Layout (single grid)
# -----------------------------
app.layout = html.Div([
    html.H1("Sentiment Dashboard", style={"textAlign": "center", "paddingTop": "20px"}),
    # Persist per-browser as a fallback when user is not authenticated
    dcc.Store(id="local-default-subject", storage_type="local"),
    # Lazy-loaded election payload
    dcc.Store(id="election-outlook-store", data={}),

    # One-time page-load trigger
    dcc.Interval(id="page-load-once", max_intervals=1, interval=250),
    

    html.Div([
        dcc.Dropdown(
            id="subject-dropdown",
            options=[],                # start empty
            value=None,                # no default yet
            style={"width": "100%"},
            placeholder="Select a subject...",
        )
    ], className="dropdown-wrapper"),

    html.Div(id="dashboard-grid", className="dashboard-grid"),
])


# -----------------------------
# Chart card components we always render inside the grid
# -----------------------------

def chart_cards():
    return [
        html.Div([
            html.H2("Sentiment Over Time", className="center-text"),
            html.Div([
                dcc.RadioItems(
                    id="sentiment-range-mode",
                    options=TIME_MODES,
                    value="30d",
                    inline=True,
                ),
                dcc.DatePickerRange(
                    id="sentiment-custom-range",
                    display_format="YYYY-MM-DD",
                    start_date=(date.today() - timedelta(days=30)),
                    end_date=date.today(),
                ),
            ], className="control-row"),
            dcc.Loading(dcc.Graph(id="sentiment-graph", style={"height": "400px"})),
        ], className="dashboard-card"),

        html.Div([
            html.H2("Mentions by Subject", className="center-text"),
            html.Div([
                dcc.RadioItems(
                    id="mentions-range-mode",
                    options=TIME_MODES,
                    value="30d",
                    inline=True,
                ),
                dcc.DatePickerRange(
                    id="mentions-custom-range",
                    display_format="YYYY-MM-DD",
                    start_date=(date.today() - timedelta(days=30)),
                    end_date=date.today(),
                ),
            ], className="control-row"),
            dcc.Loading(dcc.Graph(id="mentions-graph", style={"height": "400px"})),
        ], className="dashboard-card"),

        html.Div([
            html.H2("Momentum by Subject", className="center-text"),
            html.Div([
                dcc.RadioItems(
                    id="momentum-range-mode",
                    options=TIME_MODES,
                    value="30d",
                    inline=True,
                ),
                dcc.DatePickerRange(
                    id="momentum-custom-range",
                    display_format="YYYY-MM-DD",
                    start_date=(date.today() - timedelta(days=30)),
                    end_date=date.today(),
                ),
            ], className="control-row"),
            dcc.Loading(dcc.Graph(id="momentum-graph", style={"height": "400px"})),
        ], className="dashboard-card"),
    ]

def latest_comments_card(subject: str | None):
    """
    Big table at the bottom showing newest 10 total comments across Reddit + Bluesky + YouTube + Truth Social.
    This version is non-blocking: on error/timeout, it shows cached or a friendly placeholder.
    """
    # Combine your API's latest-comments feed with Truth Social rows from dbo.CombinedTruthSocialComments.
    # Use a shorter timeout just for this hit so it can't stall the whole dashboard render.
    df = fetch_latest_comments_combined(subject, limit=10, timeout=4)

    if "Timestamp" not in df.columns and "CreatedUTC" in df.columns:
        df["Timestamp"] = df["CreatedUTC"]

    # Ensure columns
    for col in ["Source", "Comment", "Timestamp", "URL"]:
        if col not in df.columns:
            df[col] = None

    # Build table rows
    rows = []
    for _, r in df.iterrows():
        src = (r.get("Source") or "")
        cmt = (r.get("Comment") or "")
        ts  = (r.get("Timestamp") or "")
        url = (r.get("URL") or "")
        link = html.A("link", href=url, target="_blank") if isinstance(url, str) and url.strip() else ""
        rows.append(
            html.Tr([
                html.Td(src, style={"width": "8%", "verticalAlign": "top"}),
                html.Td(cmt, style={"verticalAlign": "top", "whiteSpace": "normal"}),
                html.Td(ts,  style={"width": "14%", "verticalAlign": "top"}),
                html.Td(link, style={"width": "8%", "verticalAlign": "top"}),
            ])
        )

    # Friendly placeholder if empty after timeout/error
    if not rows:
        rows = [html.Tr([html.Td("No recent comments available.", colSpan=4)])]

    return html.Div(
        [
            html.H2("Latest Comments", className="center-text"),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("Source"),
                                html.Th("Comment"),
                                html.Th("Timestamp"),
                                html.Th("URL"),
                            ])
                        ),
                        html.Tbody(rows),
                    ],
                    style={"width": "100%"},
                ),
                style={
                    "maxHeight": "460px",
                    "overflowY": "auto",
                    "overflowX": "auto",
                    "border": "1px solid #ddd",
                    "padding": "6px",
                    "wordWrap": "break-word",
                    "whiteSpace": "normal",
                },
            ),
        ],
        className="dashboard-card",
        style={"gridColumn": "1 / -1", "fontSize": "16px"},
    )

def constituent_asks_card(subject: str | None, top_n: int = 5):
    """
    Renders a card listing up to top_n 'asks' for the selected subject
    from the latest 7-day window, using /api/constituent-asks.
    """
    if not subject:
        return html.Div(
            [html.H2("Voice of the People", className="center-text"),
             html.Div("Select a subject to view asks.")],
            className="dashboard-card",
        )

    # Fetch from API (latest window, specific subject)
    params = {
        "subjects": subject,
        "top_n": str(max(1, min(int(top_n), 10))),
        "latest": "1",
    }
    df = fetch_df_with_params(CONSTITUENT_ASKS_URL, params, timeout=6)

    # Ensure columns
    for col in ["Subject", "Ask", "SupportCount", "Confidence", "WeekStartUTC", "WeekEndUTC"]:
        if col not in df.columns:
            df[col] = None

    # Build items
    items = []
    if not df.empty:
        # Only rows belonging to this subject (defensive)
        sdf = df[df["Subject"].astype(str) == str(subject)].copy()
        # Add rank number display based on appearance order
        for i, (_, r) in enumerate(sdf.head(top_n).iterrows(), start=1):
            ask = (r.get("Ask") or "").strip()
            support = r.get("SupportCount")
            conf = r.get("Confidence")
            support_badge = html.Span(
                f"{int(support):,}" if pd.notna(support) else "0",
                style={
                    "background": "#eef",
                    "border": "1px solid #ccd",
                    "borderRadius": "10px",
                    "padding": "2px 8px",
                    "fontSize": "12px",
                    "marginLeft": "8px",
                },
                title="Supporting comments in the last 7 days",
            )
            conf_txt = (f"{float(conf):.2f}" if pd.notna(conf) else "—")
            conf_span = html.Span(
                f"(conf {conf_txt})",
                style={"color": "#888", "fontSize": "12px", "marginLeft": "6px"},
                title="Heuristic confidence based on cluster size",
            )
            items.append(
                html.Li(
                    [
                        html.Span(f"{i}. "),
                        html.Span(ask or "(no recent constituent asks)"),
                        support_badge,
                        conf_span,
                    ],
                    style={"marginBottom": "8px"}
                )
            )

    # Friendly placeholder
    if not items:
        items = [html.Li("(no recent constituent asks)")]  # shows when subject has zero asks in latest window

    return html.Div(
        [
            html.H2("Voice of the People (last 7 days)", className="center-text"),
            html.Div(
                html.Ul(items, style={"fontSize": "18px", "lineHeight": "1.4"}),
                style={"minHeight": "120px"},
            ),
        ],
        className="dashboard-card",
    )

def weekly_strategy_card(subject: str | None):
    """
    Renders a 'Weekly Strategy' card for the selected subject from /api/weekly-strategy.
    Shows a headline (StrategySummary) and collapsible details for the long statement + rationale.
    """
    if not subject:
        return html.Div(
            [html.H2("Weekly Strategy", className="center-text"),
             html.Div("Select a subject to view the weekly strategy.")],
            className="dashboard-card",
        )

    # Fetch latest strategy for this subject (latest window per subject)
    params = {"subjects": subject, "latest": "1"}
    df = fetch_df_with_params(WEEKLY_STRATEGY_URL, params, timeout=6)

    # Ensure expected columns
    needed = [
        "Subject", "StrategySummary", "StrategyStatement", "Rationale",
        "SupportCount", "Confidence", "ActionabilityScore",
        "WeekStartUTC", "WeekEndUTC"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    if df.empty:
        body = html.Div("No strategy generated for this subject in the latest window.")
        return html.Div([html.H2("Weekly Strategy", className="center-text"), body], className="dashboard-card")

    # Filter defensively to this subject & take the first row
    sdf = df[df["Subject"].astype(str) == str(subject)].copy()
    if sdf.empty:
        body = html.Div("No strategy generated for this subject in the latest window.")
        return html.Div([html.H2("Weekly Strategy", className="center-text"), body], className="dashboard-card")

    r = sdf.iloc[0]
    summary   = (r.get("StrategySummary")   or "").strip()
    statement = (r.get("StrategyStatement") or "").strip()
    rationale = (r.get("Rationale")         or "").strip()
    support   = r.get("SupportCount")
    conf      = r.get("Confidence")
    action    = r.get("ActionabilityScore")
    win_start = r.get("WeekStartUTC")
    win_end   = r.get("WeekEndUTC")

    # Small meta row
    meta_bits = []
    if pd.notna(support): meta_bits.append(f"support={int(support):,}")
    if pd.notna(conf):    meta_bits.append(f"conf={float(conf):.2f}")
    if pd.notna(action):  meta_bits.append(f"actionable={float(action):.2f}")
    if pd.notna(win_start) and pd.notna(win_end): meta_bits.append(f"window: {win_start} → {win_end}")
    meta_line = " • ".join(meta_bits) if meta_bits else ""

    # Use HTML <details> for collapsible content (Dash supports native tags)
    details = html.Details([
        html.Summary("Read full strategy"),
        html.Div([
            html.H4("Strategy Statement"),
            dcc.Markdown(statement or "_(no statement)_"),
            html.H4("Rationale", style={"marginTop": "14px"}),
            dcc.Markdown(rationale or "_(no rationale)_"),
        ], style={"marginTop": "10px"})
    ])

    return html.Div(
        [
            html.H2("Weekly Strategy", className="center-text"),
            html.H3(summary or "(no strategy available)", style={"marginTop": "4px"}),
            html.Div(meta_line, style={"color": "#666", "fontSize": "12px", "marginTop": "6px"}),
            html.Div(details, style={"marginTop": "10px"}),
        ],
        className="dashboard-card",
    )


# -----------------------------
# Dashboard builder: dynamic cards + fixed chart cards
# -----------------------------
# -----------------------------
# Dashboard builder: dynamic cards + fixed chart cards
# -----------------------------
@app.callback(
    Output("dashboard-grid", "children"),
    Input("subject-dropdown", "value"),
)
def render_dashboard(subject):
    dynamic_cards = []

    # Empty state
    if not subject:
        dynamic_cards.append(html.Div("Choose a subject to begin.", className="dashboard-card"))
        return dynamic_cards + chart_cards() + [latest_comments_card(None)]

    # ---------- 1) One-call subject bundle (last 30 days) ----------
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)
    bundle = fetch_subject_bundle(subject, str(start), str(end), timeout=4)

    # ---------- 2) Shared datasets in parallel ----------
    scorecard = fetch_df(SCORECARD_URL, timeout=4)
    traits_df = fetch_df(TRAITS_URL, timeout=4)       # SubjectTraitSummary
    bills_df  = fetch_df(BILL_SENTIMENT_URL, timeout=4)  # NationalBillSentimentMentions (rolled up)
    common_df = fetch_df(COMMON_GROUND_URL, timeout=4)   # CommonGroundIssues (latest per subject)
    issues    = fetch_json(TOP_ISSUES_URL, timeout=4) or {}  # WeeklyIdeologyTopics (Top Issues JSON)

    # ---------- 3) Scorecard tile ----------
    score, office, party, state, photo_url = 5000, "", "", "", None
    if not scorecard.empty and "Subject" in scorecard.columns:
        row = scorecard[scorecard["Subject"].astype(str) == str(subject)]
        if not row.empty and "NormalizedSentimentScore" in row.columns:
            score = coerce_int(safe_first(row["NormalizedSentimentScore"], None), default=5000)

    photos = bundle.get("photos") or []
    if photos:
        p0 = photos[0]
        photo_url = p0.get("PhotoURL")
        office    = p0.get("OfficeTitle") or ""
        party     = p0.get("Party") or ""
        state     = p0.get("State") or ""

    color = "green" if score > 6000 else ("crimson" if score < 4000 else "orange")
    dynamic_cards.append(
        html.Div(
            [
                html.Div([
                    html.Img(src=photo_url, className="scorecard-img")
                    if photo_url else html.Div(style={"width": "100px", "height": "120px", "background": "#eee"})
                ]),
                html.Div([
                    html.H1(subject, style={"fontSize": "30px", "marginBottom": "5px"}),
                    html.Div(f"{office} • {party} • {state}", style={"fontSize": "16px", "color": "#666"}),
                    html.Div("Sentiment Score", style={"marginTop": "10px", "fontSize": "14px", "color": "#999"}),
                    html.Div(f"{score:,}", style={"fontSize": "60px", "color": color, "fontWeight": "bold"}),
                ])
            ],
            className="dashboard-card scorecard-container",
        )
    )

    # ---------- 3b) Election Outlook (lazy-loaded shell) ----------
    dynamic_cards.append(election_outlook_card(None))

    # ---------- 4) Weekly Strategy (bundle) ----------
    strat = bundle.get("strategy") or {}
    meta_parts = []
    if strat.get("SupportCount") is not None:     meta_parts.append(f"support={format(int(strat['SupportCount']), ',')}")
    if strat.get("Confidence") is not None:       meta_parts.append(f"conf={float(strat['Confidence']):.2f}")
    if strat.get("ActionabilityScore") is not None: meta_parts.append(f"actionable={float(strat['ActionabilityScore']):.2f}")
    if strat.get("WeekStartUTC") and strat.get("WeekEndUTC"):
        meta_parts.append(f"window: {strat['WeekStartUTC']} → {strat['WeekEndUTC']}")
    dynamic_cards.append(
        html.Div(
            [
                html.H2("Weekly Strategy", className="center-text"),
                html.H3((strat.get("StrategySummary") or "(no strategy available)"), style={"marginTop": "4px"}),
                html.Div(" • ".join(meta_parts), style={"color": "#666", "fontSize": "12px", "marginTop": "6px"}),
                html.Details([
                    html.Summary("Read full strategy"),
                    html.Div([
                        html.H4("Strategy Statement"),
                        dcc.Markdown((strat.get("StrategyStatement") or "_(no statement)_")),
                        html.H4("Rationale", style={"marginTop": "14px"}),
                        dcc.Markdown((strat.get("Rationale") or "_(no rationale)_")),
                    ], style={"marginTop": "10px"})
                ])
            ],
            className="dashboard-card",
        )
    )

    # ---------- 5) Constituent Asks (bundle) ----------
    asks = bundle.get("asks") or []
    dynamic_cards.append(
        html.Div(
            [
                html.H2("Voice of the People", className="center-text"),
                html.Ul([
                    html.Li(
                        f"{i+1}. {a.get('Ask','')}"
                        + (f" (support={a.get('SupportCount','—')}, conf={a.get('Confidence','—')})" if a else "")
                    )
                    for i, a in enumerate(asks[:5])
                ]) if asks else html.Div("No asks available for the latest window.")
            ],
            className="dashboard-card",
        )
    )


    # ---------- 7) Latest Comments (combined) ----------
    # Combine /api/latest-comments with Truth Social comments (dbo.CombinedTruthSocialComments)
    ldf = fetch_latest_comments_combined(subject, limit=10, timeout=4)

    # Ensure columns
    for col in ["Source", "Comment", "CreatedUTC", "URL"]:
        if col not in ldf.columns:
            ldf[col] = None

    rows = []
    for _, r in ldf.iterrows():
        src = (r.get("Source") or "")
        cmt = (r.get("Comment") or "")
        ts  = (r.get("CreatedUTC") or "")
        url = (r.get("URL") or "")
        rows.append(
            html.Tr([
                html.Td(src, style={"width": "8%", "verticalAlign": "top"}),
                html.Td(cmt, style={"verticalAlign": "top", "whiteSpace": "normal"}),
                html.Td(ts,  style={"width": "14%", "verticalAlign": "top"}),
                html.Td(html.A("link", href=url, target="_blank") if isinstance(url, str) and url.strip() else "",
                        style={"width": "8%", "verticalAlign": "top"}),
            ])
        )

    if not rows:
        rows = [html.Tr([html.Td("No recent comments available.", colSpan=4)])]

    dynamic_cards.append(
        html.Div(
            [
                html.H2("Latest Comments", className="center-text"),
                html.Div(
                    html.Table(
                        [html.Thead(html.Tr([html.Th("Source"), html.Th("Comment"), html.Th("Timestamp"), html.Th("URL")])) ,
                         html.Tbody(rows)],
                        style={"width":"100%"}
                    ),
                    style={"maxHeight":"460px","overflowY":"auto","border":"1px solid #ddd","padding":"6px"},
                ),
            ],
            className="dashboard-card",
            style={"gridColumn":"1 / -1","fontSize":"16px"},
        )
    )

# ---------- 8) Behavioral Traits ----------
    pos_list, neg_list = [], []
    if not traits_df.empty and "Subject" in traits_df.columns:
        subj_norm = str(subject).strip().casefold()
        tsub = traits_df[traits_df["Subject"].astype(str).str.strip().str.casefold().eq(subj_norm)].copy()
        if not tsub.empty:
            # parse CreatedUTC robustly
            s = tsub.get("CreatedUTC", pd.Series([], dtype="object")).astype(str).str.replace("T"," ",regex=False).str.replace("Z","",regex=False)
            s = s.str.replace(r"(\.\d{6})\d+", r"\1", regex=True)
            tsub["CreatedUTC_parsed"] = pd.to_datetime(s, errors="coerce")
            def newest5(df, kind):
                kk = df[df["TraitType"].astype(str).str.strip().str.casefold().eq(kind.casefold())].copy()
                kk = kk[kk["CreatedUTC_parsed"].notna()].sort_values("CreatedUTC_parsed", ascending=False).head(5)
                return kk["TraitDescription"].dropna().astype(str).tolist()
            pos_list = newest5(tsub, "Positive")
            neg_list = newest5(tsub, "Negative")

    dynamic_cards.append(
        html.Div(
            [
                html.H2("Behavioral Traits", className="center-text"),
                html.Div([
                    html.H3("People like it when I...", style={"color": "green"}),
                    html.Ul([html.Li(p) for p in pos_list]) if pos_list else html.Div("No recent positive traits."),
                ], style={"marginBottom": "20px"}),
                html.Div([
                    html.H3("People don't like it when I...", style={"color": "crimson"}),
                    html.Ul([html.Li(n) for n in neg_list]) if neg_list else html.Div("No recent negative traits."),
                ]),
            ],
            className="dashboard-card",
        )
    )

    # ---------- 9) Public Sentiment toward National Bills ----------
    if bills_df.empty:
        dynamic_cards.append(html.Div("No bill sentiment data available.", className="dashboard-card"))
    else:
        for col in ["BillName","AverageSentimentScore","SentimentLabel"]:
            if col not in bills_df.columns: bills_df[col] = None
        dynamic_cards.append(
            html.Div(
                [
                    html.H2("Public Sentiment Toward National Bills", className="center-text"),
                    html.Table(
                        [
                            html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(str(r["BillName"])),
                                    html.Td("—" if pd.isna(r["AverageSentimentScore"]) else round(float(r["AverageSentimentScore"]),2)),
                                    html.Td(str(r["SentimentLabel"])),
                                ]) for _, r in bills_df.iterrows()
                            ]),
                        ], style={"width":"100%"}
                    ),
                ],
                className="dashboard-card",
            )
        )

    # ---------- 10) Ideology Topics (Top Issues) ----------
    if issues and all(k in issues for k in ("Liberal","Conservative","WeekStartDate")):
        dynamic_cards.append(
            html.Div(
                [
                    html.H2(f"Top Issues (Week of {issues['WeekStartDate']})", className="center-text"),
                    html.Div([
                        html.H3("Conservative Topics", style={'color':'crimson'}),
                        html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize":"20px"}) for t in issues["Conservative"]]),
                        html.H3("Liberal Topics", style={'color':'blue','marginTop':'20px'}),
                        html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize":"20px"}) for t in issues["Liberal"]]),
                    ]),
                ],
                className="dashboard-card",
            )
        )

    # ---------- 11) Common Ground Issues (filter to subject) ----------
    filtered = pd.DataFrame()

    if not common_df.empty and "Subject" in common_df.columns:
        # 1) Filter to the selected subject (case-insensitive, trimmed)
        filtered = common_df[
            common_df["Subject"].astype(str).str.strip().str.casefold()
            == str(subject).strip().casefold()
        ]

        if not filtered.empty:
            # 2) Try to identify a "week" column (e.g., WeekStart, WeekOf, WeekEnding, etc.)
            week_cols = [c for c in filtered.columns if "week" in c.lower()]
            if week_cols:
                week_col = week_cols[0]
                tmp = filtered.copy()
                tmp[week_col] = pd.to_datetime(tmp[week_col], errors="coerce")

                latest_week = tmp[week_col].max()
                if pd.notna(latest_week):
                    filtered = tmp[tmp[week_col] == latest_week]

            # 3) Limit to the latest five issues
            if "IssueRank" in filtered.columns:
                filtered = (
                    filtered.sort_values("IssueRank", ascending=True)
                            .head(5)
                )
            else:
                filtered = filtered.head(5)

    dynamic_cards.append(
        html.Div(
            [
                html.H2("Common Ground Issues", className="center-text"),
                (
                    html.Ul(
                        [
                            html.Li(
                                [
                                    html.Span(f"{r.get('IssueRank','')}. "),
                                    html.Span(f"{r.get('Issue','')}: "),
                                    html.Span(r.get("Explanation","")),
                                ],
                                className="common-ground-item",
                            )
                            for _, r in filtered.iterrows()
                        ],
                        className="common-ground-list",
                    )
                    if not filtered.empty
                    else html.Div("No common ground items for this subject (yet).")
                ),
            ],
            className="dashboard-card",
        )
    )

    # ---------- 12) Return cards + charts ----------
    return dynamic_cards + chart_cards()




@app.callback(
    Output("election-outlook-store", "data"),
    Input("subject-dropdown", "value"),
)
def load_election_outlook_store(subject):
    if not subject:
        return {}
    try:
        payload = fetch_election_outlook(subject, timeout=4)
        return payload if isinstance(payload, dict) else {}
    except Exception as e:
        print(f"[election_outlook] load error for {subject}: {e}")
        return {}


@app.callback(
    Output("election-outlook-gauge", "figure"),
    Output("election-outlook-primary", "children"),
    Output("election-outlook-competitors", "children"),
    Output("election-outlook-path", "children"),
    Output("election-outlook-card", "style"),
    Input("election-outlook-store", "data"),
    Input("election-outlook-toggle", "value"),
)
def update_election_outlook_card(payload, mode):
    payload = payload or {}
    hidden_style = {"gridColumn": "1 / -1", "display": "none"}
    shown_style = {"gridColumn": "1 / -1", "display": "block"}
    if not payload or payload.get("error") or not payload.get("RaceKey"):
        fig = make_election_gauge(0.0, "Unavailable", "Election Outlook")
        primary = html.Div()
        competitors = []
        path = ""
        return fig, primary, competitors, path, hidden_style

    subject_info = payload.get("Subject") or {}
    mode = (mode or "general").lower()
    if mode == "nomination":
        mode_key = "NominationOutlook"
        title = "Nomination Outlook"
        prob = float(((payload.get(mode_key) or {}).get("Probability") or 0.0))
        label = (payload.get(mode_key) or {}).get("Label") or "Long Shot"
        competitors = (payload.get(mode_key) or {}).get("Competitors") or []
    else:
        mode_key = "GeneralElectionOutlook"
        title = "General Election Outlook"
        prob = float(((payload.get(mode_key) or {}).get("Probability") or 0.0))
        label = (payload.get(mode_key) or {}).get("Label") or "Long Shot"
        competitors = (payload.get(mode_key) or {}).get("Competitors") or []

    fig = make_election_gauge(prob, label, title)

    def _money(v):
        try:
            return f"${float(v):,.0f}"
        except Exception:
            return "—"

    primary = html.Div([
        html.Div(title, style={"fontSize": "14px", "color": "#666"}),
        html.Div(f"{prob * 100:.1f}%", style={"fontSize": "42px", "fontWeight": "700", "color": probability_color(prob), "lineHeight": "1.0"}),
        html.Div(label, style={"fontSize": "20px", "fontWeight": "600", "marginBottom": "10px"}),
        html.Div([
            html.Span(subject_info.get("Subject") or "", style={"fontWeight": "700", "fontSize": "20px"}),
            html.Span(f"  •  {(subject_info.get('Party') or '')}", style={"color": party_color(subject_info.get("Party")), "fontWeight": "600"}),
        ]),
        html.Div(subject_info.get("OfficeTitle") or "", style={"color": "#666", "fontSize": "14px", "marginTop": "4px"}),
        html.Div(
            [
                html.Span(f"Nomination: {float(subject_info.get('NominationProbability') or 0.0) * 100:.1f}%"),
                html.Span("  •  "),
                html.Span(f"General: {float(subject_info.get('WinProbability') or 0.0) * 100:.1f}%"),
            ],
            style={"marginTop": "10px", "fontSize": "15px"},
        ),
        html.Div(
            [
                html.Span(f"Fundraising: {_money(subject_info.get('TTL_DISB'))}"),
                html.Span("  •  "),
                html.Span(f"Tier Score: {float(subject_info.get('CandidateTierScore') or 0.0):.2f}"),
            ],
            style={"marginTop": "6px", "fontSize": "14px", "color": "#666"},
        ),
    ])

    comp_children = []
    for row in competitors:
        p = float(row.get("Probability") or 0.0)
        comp_children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(row.get("Subject") or "", style={"fontWeight": "600"}),
                            html.Span(
                                row.get("Party") or "",
                                style={
                                    "marginLeft": "8px",
                                    "padding": "2px 8px",
                                    "borderRadius": "12px",
                                    "backgroundColor": party_color(row.get("Party")),
                                    "color": "white",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "6px", "flexWrap": "wrap"},
                    ),
                    html.Div(
                        [
                            html.Span(f"{p * 100:.1f}%", style={"fontWeight": "700", "color": probability_color(p)}),
                            html.Span(f"  •  {row.get('Outlook') or ''}", style={"color": "#666"}),
                        ],
                        style={"marginTop": "2px"},
                    ),
                ],
                style={"padding": "8px 0", "borderBottom": "1px solid #eee"},
            )
        )

    if not comp_children:
        comp_children = [html.Div("No competitor data available.")]

    path = payload.get("PathToVictoryInsight") or subject_info.get("PathToVictoryInsight") or "No path-to-victory insight available."
    return fig, primary, comp_children, path, shown_style


# -----------------------------
# Enable/disable custom date pickers based on mode
# -----------------------------
@app.callback(
    Output("sentiment-custom-range", "disabled"),
    Input("sentiment-range-mode", "value"),
)
def toggle_sentiment_custom(mode):
    return mode != "Custom"


@app.callback(
    Output("mentions-custom-range", "disabled"),
    Input("mentions-range-mode", "value"),
)
def toggle_mentions_custom(mode):
    return mode != "Custom"


@app.callback(
    Output("momentum-custom-range", "disabled"),
    Input("momentum-range-mode", "value"),
)
def toggle_momentum_custom(mode):
    return mode != "Custom"


# -----------------------------
# Chart callbacks (subject + timeframe aware, using cached base datasets)
# -----------------------------
# --- SENTIMENT OVER TIME CALLBACK ---
# --- SENTIMENT OVER TIME CALLBACK ---
@app.callback(
    Output("sentiment-graph", "figure"),
    Input("subject-dropdown", "value"),
    Input("sentiment-range-mode", "value"),
    Input("sentiment-custom-range", "start_date"),
    Input("sentiment-custom-range", "end_date"),
)
def update_sentiment_chart(subject, mode, custom_start, custom_end):
    import pandas as pd
    import plotly.graph_objects as go
    def empty(title): 
        f=go.Figure(); f.update_layout(title=title, template="plotly_white"); return f
    if not subject: return empty("Select a subject to view sentiment.")
    try:
        start, end = date_range_from_mode(mode, custom_start, custom_end)
    except Exception:
        end = pd.Timestamp.utcnow().normalize(); start = end - pd.Timedelta(days=30)

    df = fetch_df_with_params(TIMESERIES_URL, {
        "subject": subject,
        "start_date": str(pd.to_datetime(start).date()),
        "end_date":   str(pd.to_datetime(end).date()),
    }, timeout=5)
    if df.empty: return empty("Sentiment over time")

    date_col = "SentimentDate" if "SentimentDate" in df.columns else ("Date" if "Date" in df.columns else None)
    if not date_col: return empty("Sentiment over time")
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    ycol = "NormalizedSentimentScore" if "NormalizedSentimentScore" in df.columns else ("Score" if "Score" in df.columns else None)
    if not ycol: return empty("Sentiment over time")

    fig = go.Figure()
    fig.add_scatter(x=df["Date"].astype(str), y=df[ycol].astype(int), mode="lines", name="Score")
    fig.update_layout(title="Sentiment over time", template="plotly_white",
                      xaxis_title="Date", yaxis_title="Score", margin=dict(l=40,r=10,t=40,b=40))
    return fig




@app.callback(
    Output("mentions-graph", "figure"),
    Input("subject-dropdown", "value"),
    Input("mentions-range-mode", "value"),
    Input("mentions-custom-range", "start_date"),
    Input("mentions-custom-range", "end_date"),
)
def update_mentions_chart(subject, mode, custom_start, custom_end):
    import pandas as pd
    import plotly.graph_objects as go
    def empty(title): 
        f=go.Figure(); f.update_layout(title=title, template="plotly_white"); return f
    if not subject: return empty("Select a subject to view mentions.")

    try:
        start, end = date_range_from_mode(mode, custom_start, custom_end)
    except Exception:
        end = pd.Timestamp.utcnow().normalize(); start = end - pd.Timedelta(days=30)

    df = fetch_df_with_params(MENTION_COUNTS_DAILY_URL, {
        "subject": subject,
        "start_date": str(pd.to_datetime(start).date()),
        "end_date":   str(pd.to_datetime(end).date()),
    }, timeout=5)
    if df.empty: return empty("Mentions by subject")

    date_col = "ActivityDate" if "ActivityDate" in df.columns else ("Date" if "Date" in df.columns else None)
    ycol   = "MentionCount" if "MentionCount" in df.columns else ("Count" if "Count" in df.columns else None)
    if not date_col or not ycol: return empty("Mentions by subject")

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    fig = go.Figure()
    fig.add_bar(x=df["Date"].astype(str), y=df[ycol].fillna(0).astype(int), name="Mentions")
    fig.update_layout(title="Mentions by subject", template="plotly_white",
                      xaxis_title="Date", yaxis_title="Mentions", margin=dict(l=40,r=10,t=40,b=40))
    return fig



# --- MOMENTUM CALLBACK (z-scored EMA of log-volume-weighted sentiment) ---
@app.callback(
    Output("momentum-graph", "figure"),
    Input("subject-dropdown", "value"),
    Input("momentum-range-mode", "value"),
    Input("momentum-custom-range", "start_date"),
    Input("momentum-custom-range", "end_date"),
)
def update_momentum_chart(subject, mode, custom_start, custom_end):
    import pandas as pd, numpy as np
    import plotly.graph_objects as go
    def empty(title): 
        f=go.Figure(); f.update_layout(title=title, template="plotly_white"); f.add_hline(y=0,line_width=1,line_dash="dot"); return f
    if not subject: return empty("Select a subject to view momentum.")

    try:
        start, end = date_range_from_mode(mode, custom_start, custom_end)
    except Exception:
        end = pd.Timestamp.utcnow().normalize(); start = end - pd.Timedelta(days=30)

    df = fetch_df_with_params(MOMENTUM_URL, {
        "subject": subject,
        "start_date": str(pd.to_datetime(start).date()),
        "end_date":   str(pd.to_datetime(end).date()),
    }, timeout=5)
    if df.empty: return empty("Momentum")

    # normalize columns
    date_col = "ActivityDate" if "ActivityDate" in df.columns else ("Date" if "Date" in df.columns else None)
    if not date_col: return empty("Momentum")
    df["ActivityDate"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    if "MentionCount" not in df.columns and "Mentions" in df.columns: df["MentionCount"] = df["Mentions"]
    if "MentionCount" not in df.columns: df["MentionCount"] = 0
    if "AvgSentiment" not in df.columns and "AverageSentiment" in df.columns: df["AvgSentiment"] = df["AverageSentiment"]
    if "AvgSentiment" not in df.columns: df["AvgSentiment"] = np.nan
    df = df.sort_values("ActivityDate").reset_index(drop=True)

    # lightweight momentum calc (fallback if your compute_momentum isn't present)
    try:
        df["Subject"] = str(subject)
        tmp = compute_momentum(df)  # use your existing if available
    except Exception:
        tmp = df.copy()
        tmp["logV"] = np.log1p(tmp["MentionCount"].fillna(0).astype(float))
        tmp["lw_sent"] = tmp["AvgSentiment"].fillna(0).astype(float) * tmp["logV"]
        tmp["ema_fast"] = tmp["lw_sent"].ewm(span=7, adjust=False, min_periods=3).mean()
        tmp["ema_slow"] = tmp["lw_sent"].ewm(span=21, adjust=False, min_periods=5).mean()
        tmp["accel"] = (tmp["ema_fast"] - tmp["ema_slow"]).fillna(0.0)
        roll = tmp["ema_fast"].rolling(90, min_periods=10)
        mu, sigma = roll.mean(), roll.std().replace(0, np.nan)
        tmp["base"] = mu
        tmp["z_momentum"] = ((tmp["ema_fast"] - mu) / sigma).fillna(0.0)

    need = {"ActivityDate","z_momentum","MentionCount","AvgSentiment","base","ema_fast"}
    if not need.issubset(set(tmp.columns)): return empty("Momentum")

    x = pd.to_datetime(tmp["ActivityDate"]).astype(str).tolist()
    z = tmp["z_momentum"].astype(float).tolist()
    mentions = tmp["MentionCount"].fillna(0).astype(float).tolist()
    avgsent  = tmp["AvgSentiment"].fillna(0).astype(float).tolist()
    base     = tmp["base"].fillna(0).astype(float).tolist()
    ema_fast = tmp["ema_fast"].fillna(0).astype(float).tolist()
    accel    = tmp.get("accel", pd.Series([0]*len(tmp))).astype(float).tolist()

    fig = go.Figure()
    fig.add_bar(
        name=f"{subject} — z-momentum",
        x=x, y=z,
        hovertemplate=("<b>%{customdata[0]}</b><br>Date: %{x}<br>"
                       "Z-Momentum: %{y:.2f}<br>Mentions: %{customdata[1]:.0f}<br>"
                       "Avg Sentiment: %{customdata[2]:.2f}<br>Base: %{customdata[3]:.3f}<br>"
                       "EMA7: %{customdata[4]:.3f}<extra></extra>"),
        customdata=np.column_stack([np.full(len(x), str(subject)), mentions, avgsent, base, ema_fast]),
    )
    fig.add_trace(go.Scatter(name=f"{subject} — accel (EMA7−EMA21)", x=x, y=accel, mode="lines", yaxis="y2",
                             hovertemplate="Acceleration: %{y:.3f}<extra></extra>"))
    fig.update_layout(
        title="Momentum (z-scored EMA of log-volume-weighted sentiment)",
        template="plotly_white",
        xaxis_title="Date", yaxis_title="Z-Score vs 90-day baseline",
        yaxis2=dict(overlaying="y", side="right", showgrid=False),
        barmode="overlay", bargap=0.15,
        legend_orientation="h", legend_y=-0.2,
        hovermode="x unified", margin=dict(l=40,r=40,t=40,b=40),
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot")
    return fig


@app.callback(
    Output("local-default-subject", "data"),
    Input("subject-dropdown", "value"),
    prevent_initial_call=True
)
def persist_default_subject(value):
    if not value:
        return dash.no_update
    uid = current_user_id()
    try:
        if uid:
            with engine.begin() as conn:
                # Upsert pattern for SQL Server (MERGE for full correctness; here’s a simple pattern)
                conn.execute(text("""
                    IF EXISTS (SELECT 1 FROM dbo.UserPreferences WHERE user_id = :u)
                        UPDATE dbo.UserPreferences SET default_subject = :s WHERE user_id = :u;
                    ELSE
                        INSERT INTO dbo.UserPreferences (user_id, default_subject) VALUES (:u, :s);
                """), {"u": uid, "s": value})
            # For logged-in users we still return local data to keep UX consistent across tabs
        # For guests, this is the primary persistence
        return {"default_subject": value}
    except Exception:
        # Fallback: at least keep it locally
        return {"default_subject": value}


@app.callback(
    Output("subject-dropdown", "options"),
    Output("subject-dropdown", "value"),
    Input("page-load-once", "n_intervals"),
    State("local-default-subject", "data"),
    State("subject-dropdown", "value"),
    prevent_initial_call=True,
)
def load_subject_options(_, saved_value, current_value):
    print("[page-load] load_subject_options fired")
    try:
        df = fetch_df(SUBJECTS_URL, timeout=8)
        if df.empty or "Subject" not in df.columns:
            df = fetch_df(PHOTOS_URL, timeout=8)  # fallback
            if not df.empty and "Subject" in df.columns:
                df = df[["Subject"]].dropna().drop_duplicates()
            else:
                print("[subjects] empty from both /subjects and /subject-photos")
                return [], None

        subs = sorted(df["Subject"].astype(str).str.strip().unique())
        options = [{"label": s, "value": s} for s in subs if s]

        # pull the actual value out of the store dict, if present
        saved_str = None
        if isinstance(saved_value, dict):
            saved_str = saved_value.get("default_subject")
        elif isinstance(saved_value, str):
            saved_str = saved_value

        # choose the default: current -> saved -> first
        if current_value in subs:
            value = current_value
        elif saved_str in subs:
            value = saved_str
        else:
            value = subs[0] if subs else None

        print(f"[subjects] loaded {len(options)} options; default={value}")
        return options, value
    except Exception as e:
        print("[subjects] error:", e)
        return [], None


if __name__ == "__main__":
    app.run(debug=False)















































































