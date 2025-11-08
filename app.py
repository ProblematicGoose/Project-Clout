import time
import dash
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
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


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
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mssql+pyodbc://sa:Is2*2set@3.tcp.ngrok.io,27671/Clout?driver=ODBC+Driver+17+for+SQL+Server"
)

# Create the engine
engine = create_engine(DATABASE_URL)



from flask_sql_api import app as flask_app
app = dash.Dash(__name__, server=flask_app)
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
    # Limit cache to 20 entries max
    if len(_CACHE) > 20:
        oldest = sorted(_CACHE.items(), key=lambda kv: kv[1][0])[:5]
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

    # helpful log so you can copy/paste into a browser
    print(f"[bundle] GET {url}")

    cache_key = f"BUNDLE::{subject_norm}::{params.get('start_date','')}::{params.get('end_date','')}"
    cached = _cache_get(cache_key)
    if isinstance(cached, dict):
        return cached

    # Try the bundle route
    data = fetch_json(url, timeout=timeout)  # your existing helper
    if data:
        _cache_set(cache_key, data)   # ✅ cache what you received
        return data

    # ---- Fallback: assemble bundle from existing, working endpoints ----
    try:
        ts_df = fetch_timeseries_df(subject_norm, params.get("start_date"), params.get("end_date"))
        mo_df = fetch_momentum_df(subject_norm, params.get("start_date"), params.get("end_date"))
        ws_df = fetch_df_with_params(WEEKLY_STRATEGY_URL, {"subjects": subject_norm, "latest": "1"}, timeout=timeout)
        asks_df = fetch_df_with_params(CONSTITUENT_ASKS_URL, {"subjects": subject_norm, "top_n": 5, "latest": "1"}, timeout=timeout)
        photos_df = fetch_df(PHOTOS_URL)
        comments_df = fetch_df_with_params(LATEST_COMMENTS_URL, {"subject": subject_norm, "limit": 5}, timeout=timeout)

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
        _cache_set(cache_key, bundle)  # ✅ cache what you built
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
        except Exception:
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
        except Exception:
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
import pandas as pd

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
import os
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
        df = fetch_df_with_params(TIMESERIES_URL, params, timeout=10)  # your app helper
    except NameError:
        r = requests.get(TIMESERIES_URL, params=params, timeout=10)
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
    
 # Radio options for date ranges used by all charts
TIME_MODES = [
    {"label": "Today",      "value": "Today"},
    {"label": "This Week",  "value": "This Week"},
    {"label": "This Month", "value": "This Month"},
    {"label": "This Year",  "value": "This Year"},
    {"label": "Custom",     "value": "Custom"},
    # If you want quick-shortcuts too, uncomment any of these:
    # {"label": "Last 7d",  "value": "7d"},
    # {"label": "Last 30d", "value": "30d"},
    # {"label": "Last 90d", "value": "90d"},
]   


# -----------------------------
# Layout (single grid)
# -----------------------------
app.layout = html.Div([
    html.H1("Sentiment Dashboard", style={"textAlign": "center", "paddingTop": "20px"}),
    # Persist per-browser as a fallback when user is not authenticated
    dcc.Store(id="local-default-subject", storage_type="local"),

    # One-time page-load trigger
    dcc.Interval(id="page-load-once", max_intervals=1, interval=250),
    

    html.Div([
        dcc.Dropdown(
            id="subject-dropdown",
            options=[],         # start empty; we’ll fill via callback
            value=None,
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
                    value="This Month",
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
                    value="This Month",
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
                    value="This Month",
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
    Big table at the bottom showing newest 10 total comments across Reddit + Bluesky + YouTube.
    This version is non-blocking: on error/timeout, it shows cached or a friendly placeholder.
    """
    params = {"limit": 10}
    if subject:
        params["subject"] = subject

    # Use a shorter timeout just for this hit so it can't stall the whole dashboard render.
    df = fetch_df_with_params(LATEST_COMMENTS_URL, params, timeout=4)

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
                    "border": "1px solid #ddd",
                    "padding": "6px",
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
            [html.H2("Constituent Asks", className="center-text"),
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
    Output("subject-dropdown", "options"),
    Output("subject-dropdown", "value"),
    Input("page-load-once", "n_intervals"),
    State("subject-dropdown", "value"),
    prevent_initial_call=True,
)
def load_subject_options(_, current_value):
    # Fetch canonical subject list from the API route you already have
    df = fetch_df(SUBJECTS_URL, timeout=8)
    if df.empty or "Subject" not in df.columns:
        # Fallback: use photos if /api/subjects returns nothing
        photo_df = fetch_df(PHOTOS_URL, timeout=8)
        if not photo_df.empty and "Subject" in photo_df.columns:
            df = photo_df[["Subject"]].dropna().drop_duplicates()
        else:
            return [], None

    subs = sorted(df["Subject"].astype(str).str.strip().unique())
    options = [{"label": s, "value": s} for s in subs if s]

    # keep current if still valid; otherwise choose the first
    value = current_value if current_value in subs else (subs[0] if subs else None)
    return options, value



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

    # 1) One-call subject bundle (last 30 days)
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)
    bundle = fetch_subject_bundle(subject, str(start), str(end)) or {}

    # 2) Shared/cached datasets (scorecard, traits, bills, common)
    urls = {
        "scorecard": SCORECARD_URL,
        "traits": TRAITS_URL,
        "bills": BILL_SENTIMENT_URL,
        "common": COMMON_GROUND_URL,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {k: ex.submit(fetch_df, v) for k, v in urls.items()}
        results = {k: f.result() for k, f in futures.items()}

    scorecard = results["scorecard"]
    traits_df = results["traits"]
    bills_df = results["bills"]
    common_df = results["common"]

    # 3) Scorecard tile (score + photo/meta)
    score, office, party, state, photo_url = 10000, "", "", "", None
    if not scorecard.empty and "Subject" in scorecard.columns:
        row = scorecard[scorecard["Subject"].astype(str) == str(subject)]
        if not row.empty and "NormalizedSentimentScore" in row.columns:
            val = safe_first(row["NormalizedSentimentScore"], None)
            score = coerce_int(val, default=5000)

    photos = bundle.get("photos") or []
    if photos:
        p0 = photos[0]
        photo_url = p0.get("PhotoURL")
        office = p0.get("OfficeTitle") or ""
        party = p0.get("Party") or ""
        state = p0.get("State") or ""
    else:
        try:
            photo_df = fetch_df(PHOTOS_URL)
            if not photo_df.empty and "Subject" in photo_df.columns:
                meta = photo_df[photo_df["Subject"].astype(str) == str(subject)]
                if not meta.empty:
                    photo_url = safe_first(meta.get("PhotoURL", pd.Series([None])), None)
                    office = (safe_first(meta.get("OfficeTitle", pd.Series([""])), "") or "")
                    party  = (safe_first(meta.get("Party", pd.Series([""])), "") or "")
                    state  = (safe_first(meta.get("State", pd.Series([""])), "") or "")
        except Exception:
            pass

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

    # 4) Weekly Strategy (bundle)
    strat = bundle.get("strategy") or {}
    meta_parts = []
    try:
        if strat.get("SupportCount") is not None:
            meta_parts.append(f"support={format(int(strat['SupportCount']), ',')}")
    except Exception:
        pass
    try:
        if strat.get("Confidence") is not None:
            meta_parts.append(f"conf={float(strat['Confidence']):.2f}")
    except Exception:
        pass
    try:
        if strat.get("ActionabilityScore") is not None:
            meta_parts.append(f"actionable={float(strat['ActionabilityScore']):.2f}")
    except Exception:
        pass
    if strat.get("WeekStartUTC") and strat.get("WeekEndUTC"):
        meta_parts.append(f"window: {strat['WeekStartUTC']} → {strat['WeekEndUTC']}")
    subtitle = " • ".join(meta_parts)

    dynamic_cards.append(
        html.Div(
            [
                html.H2("Weekly Strategy", className="center-text"),
                html.H3((strat.get("StrategySummary") or "(no strategy available)"), style={"marginTop": "4px"}),
                html.Div(subtitle, style={"color": "#666", "fontSize": "12px", "marginTop": "6px"}),
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

    # 5) Constituent Asks (bundle)
    asks = bundle.get("asks") or []
    dynamic_cards.append(
        html.Div(
            [
                html.H2("Constituent Asks", className="center-text"),
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

    # 6) Subject Photos (bundle; up to 3)
    if photos:
        dynamic_cards.append(
            html.Div(
                [
                    html.H2("Subject Photos", className="center-text"),
                    html.Div([
                        html.Img(src=p.get("PhotoURL"), style={"maxHeight": "160px", "marginRight": "10px"})
                        for p in photos[:3] if p.get("PhotoURL")
                    ])
                ],
                className="dashboard-card",
            )
        )

    # 7) Latest Comments (bundle)
    comments = bundle.get("latest_comments") or []
    rows = [
        html.Tr([
            html.Td(c.get("Source", ""), style={"width": "8%", "verticalAlign": "top"}),
            html.Td(c.get("Comment", ""), style={"verticalAlign": "top", "whiteSpace": "normal"}),
            html.Td(c.get("CreatedUTC", ""), style={"width": "14%", "verticalAlign": "top"}),
            html.Td(html.A("link", href=c.get("URL"), target="_blank") if c.get("URL") else "",
                    style={"width": "8%", "verticalAlign": "top"}),
        ])
        for c in comments
    ] or [html.Tr([html.Td("No recent comments available.", colSpan=4)])]
    dynamic_cards.append(
        html.Div(
            [
                html.H2("Latest Comments", className="center-text"),
                html.Div(
                    html.Table(
                        [
                            html.Thead(html.Tr([html.Th("Source"), html.Th("Comment"), html.Th("Timestamp"), html.Th("URL")])),
                            html.Tbody(rows),
                        ],
                        style={"width": "100%"},
                    ),
                    style={"maxHeight": "460px", "overflowY": "auto", "border": "1px solid #ddd", "padding": "6px"},
                ),
            ],
            className="dashboard-card",
            style={"gridColumn": "1 / -1", "fontSize": "16px"},
        )
    )

    # 8) Behavioral Traits (Positive/Negative – newest 5 each)
    pos_list, neg_list = [], []
    if not traits_df.empty and "Subject" in traits_df.columns and "CreatedUTC" in traits_df.columns:
        subj_norm = str(subject).strip().casefold()
        subj_col = traits_df["Subject"].astype(str).str.strip().str.casefold()
        tsub = traits_df[subj_col == subj_norm].copy()
        if not tsub.empty:
            s = tsub["CreatedUTC"].astype(str).str.strip()
            s = s.str.replace("T", " ", regex=False).str.replace("Z", "", regex=False)
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

    # 9) Public Sentiment toward National Bills
    if bills_df.empty:
        dynamic_cards.append(html.Div("No bill sentiment data available.", className="dashboard-card"))
    else:
        for col in ["BillName", "AverageSentimentScore", "SentimentLabel"]:
            if col not in bills_df.columns:
                bills_df[col] = None
        dynamic_cards.append(
            html.Div(
                [
                    html.H2("Public Sentiment Toward National Bills", className="center-text"),
                    html.Table(
                        [
                            html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(r["BillName"]),
                                    html.Td((round(r["AverageSentimentScore"], 2) if pd.notna(r["AverageSentimentScore"]) else "—")),
                                    html.Td(r["SentimentLabel"]),
                                ]) for _, r in bills_df.iterrows()
                            ]),
                        ],
                        style={"width": "100%"},
                    ),
                ],
                className="dashboard-card",
            )
        )

    # 10) Ideology Topics (Top Issues)
    issues = fetch_json(TOP_ISSUES_URL)
    if issues and all(k in issues for k in ("Liberal", "Conservative", "WeekStartDate")):
        week = issues["WeekStartDate"]
        liberal = issues["Liberal"]
        conservative = issues["Conservative"]
        dynamic_cards.append(
            html.Div(
                [
                    html.H2(f"Top Issues (Week of {week})", className="center-text"),
                    html.Div([
                        html.Div([
                            html.H3("Conservative Topics", style={'color': 'crimson'}),
                            html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize": "20px"}) for t in conservative]),
                            html.H3("Liberal Topics", style={'color': 'blue', 'marginTop': '20px'}),
                            html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize": "20px"}) for t in liberal])
                        ])
                    ]),
                ],
                className="dashboard-card",
            )
        )

    # 11) Common Ground Issues (filtered for this subject)
    if not common_df.empty and "Subject" in common_df.columns:
        subj_norm = str(subject).strip().casefold()
        subj_col = common_df["Subject"].astype(str).str.strip().str.casefold()
        filtered = common_df[subj_col.eq(subj_norm)]
    else:
        filtered = pd.DataFrame()

    dynamic_cards.append(
        html.Div(
            [
                html.H2("Common Ground Issues", className="center-text"),
                (html.Ul([
                    html.Li([
                        html.Span(f"{r.get('IssueRank', '')}. "),
                        html.Span(f"{r.get('Issue', '')}: "),
                        html.Span(r.get("Explanation", "")),
                    ], className="common-ground-item")
                    for _, r in filtered.iterrows()
                ], className="common-ground-list")
                 if not filtered.empty else
                 html.Div("No common ground items for this subject (yet)."))
            ],
            className="dashboard-card",
        )
    )

    # 12) Done: return subject panels + charts
    return dynamic_cards + chart_cards()


    # Traits (cached) — newest 5 Positive and 5 Negative
    traits = fetch_df(TRAITS_URL)
    pos_list, neg_list = [], []
    if not traits.empty and "Subject" in traits.columns and "CreatedUTC" in traits.columns:
        subj_norm = str(subject).strip().casefold()
        subj_col = traits["Subject"].astype(str).str.strip().str.casefold()
        tsub = traits[subj_col == subj_norm].copy()
        if not tsub.empty:
            raw = tsub["CreatedUTC"].astype(str).str.strip()
            s = raw.str.replace("T", " ", regex=False).str.replace("Z", "", regex=False)
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

    # Bill Sentiment (cached)
    bills = fetch_df(BILL_SENTIMENT_URL)
    if bills.empty:
        dynamic_cards.append(html.Div("No bill sentiment data available.", className="dashboard-card"))
    else:
        for col in ["BillName", "AverageSentimentScore", "SentimentLabel"]:
            if col not in bills.columns:
                bills[col] = None
        dynamic_cards.append(
            html.Div(
                [
                    html.H2("Public Sentiment Toward National Bills", className="center-text"),
                    html.Table(
                        [
                            html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(r.get("BillName", "") if isinstance(r, dict) else r["BillName"]),
                                    html.Td((lambda v: round(v, 2) if pd.notna(v) else "—")(r.get("AverageSentimentScore", None) if isinstance(r, dict) else r["AverageSentimentScore"])),
                                    html.Td(r.get("SentimentLabel", "") if isinstance(r, dict) else r["SentimentLabel"]),
                                ]) for _, r in bills.iterrows()
                            ]),
                        ],
                        style={"width": "100%"},
                    ),
                ],
                className="dashboard-card",
            )
        )

    # Top Issues (cached)
    issues = fetch_json(TOP_ISSUES_URL)
    if issues and all(k in issues for k in ("Liberal", "Conservative", "WeekStartDate")):
        week = issues["WeekStartDate"]
        liberal = issues["Liberal"]
        conservative = issues["Conservative"]
        dynamic_cards.append(
            html.Div(
                [
                    html.H2(f"Top Issues (Week of {week})", className="center-text"),
                    html.Div([
                        html.Div([
                            html.H3("Conservative Topics", style={'color': 'crimson'}),
                            html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize": "20px"}) for t in conservative]),
                            html.H3("Liberal Topics", style={'color': 'blue', 'marginTop': '20px'}),
                            html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize": "20px"}) for t in liberal])
                        ])
                    ]),
                ],
                className="dashboard-card",
            )
        )

        # Common Ground (cached)
    common_df = fetch_df(COMMON_GROUND_URL)
    if not common_df.empty and "Subject" in common_df.columns:
        subj_norm = str(subject).strip().casefold()
        subj_col = common_df["Subject"].astype(str).str.strip().str.casefold()
        filtered = common_df[subj_col.eq(subj_norm)]
    else:
        filtered = pd.DataFrame()

    dynamic_cards.append(
        html.Div(
            [
                html.H2("Common Ground Issues", className="center-text"),
                (html.Ul([
                    html.Li([
                        html.Span(f"{r.get('IssueRank', '')}. "),
                        html.Span(f"{r.get('Issue', '')}: "),
                        html.Span(r.get("Explanation", "")),
                    ], className="common-ground-item")
                    for _, r in filtered.iterrows()
                ], className="common-ground-list")
                 if not filtered.empty else
                 html.Div("No common ground items for this subject (yet)."))
            ],
            className="dashboard-card",
        )
    )  

    # Cache the full layout per subject to speed up repeat visits
    subject_cache_key = f"DASH::{subject}"
    cached_layout = _cache_get(subject_cache_key)
    if cached_layout:
        return cached_layout

    # FINAL: charts + latest comments table appended at the bottom
    output = dynamic_cards + chart_cards() + [latest_comments_card(subject)]
    _cache_set(subject_cache_key, output)
    return output



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
    [
        Input("subject-dropdown", "value"),
        Input("sentiment-range-mode", "value"),            # radio: Today / This Week / ...
        Input("sentiment-custom-range", "start_date"),     # datepicker
        Input("sentiment-custom-range", "end_date"),
    ],
)
def update_sentiment_over_time(subject, mode, custom_start, custom_end):
    fig = go.Figure()

    if not subject:
        fig.update_layout(
            title="Select a subject to view sentiment over time.",
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Normalized Sentiment (0–10,000)",
        )
        return fig

    # unified helper you already have (supports Today/This Week/etc and 7d/30d/90d/custom)
    start, end = date_range_from_mode(mode, custom_start, custom_end)

    df = fetch_timeseries_df(subject=subject, start_ts=start, end_ts=end)
    if df.empty:
        fig.update_layout(
            title="No time series data in the selected range.",
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Normalized Sentiment (0–10,000)",
        )
        return fig

    fig.add_trace(go.Scatter(
        name=subject,
        x=df["SentimentDate"],
        y=df["NormalizedSentimentScore"],
        mode="lines+markers",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Sentiment Over Time",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Normalized Sentiment (0–10,000)",
        hovermode="x unified",
    )
    return fig




@app.callback(
    Output("mentions-graph", "figure"),
    [
        Input("subject-dropdown", "value"),
        Input("mentions-range-mode", "value"),
        Input("mentions-custom-range", "start_date"),
        Input("mentions-custom-range", "end_date"),
    ],
)
def update_mentions_chart(subject, mode, custom_start, custom_end):
    fig = go.Figure()

    if not subject:
        fig.update_layout(title="Select a subject to view data.", template="plotly_white")
        return fig

    start, end = date_range_from_mode(mode, custom_start, custom_end)

    start, end = date_range_from_mode(mode, custom_start, custom_end)
    df = fetch_df_with_params(
        MENTION_COUNTS_DAILY_URL,
        {
            "subject": subject,
            "start_date": str(start.date()),
            "end_date": str(end.date()),
        },
        timeout=10,
    )

    if df.empty or "MentionCount" not in df.columns:
        fig.update_layout(title="No mention data available for the selected range.", template="plotly_white")
        return fig

    # Try to find the right date column
    date_col = None
    for c in ["Date", "MentionDate", "CreatedUTC", "Day"]:
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        fig.update_layout(title="No date column in mention count data.", template="plotly_white")
        return fig

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    if df.empty:
        fig.update_layout(title="No mention data available for the selected range.", template="plotly_white")
        return fig

    daily_counts = df.groupby(date_col)["MentionCount"].sum().reset_index(name="Mentions")
    total_mentions = daily_counts["Mentions"].sum()

    fig.add_trace(
        go.Bar(
            x=daily_counts[date_col],
            y=daily_counts["Mentions"],
            name="Mentions per Day",
            marker_color="dodgerblue",
        )
    )

    fig.update_layout(
        title=f"Mentions ({mode}) — Total: {total_mentions:,}",
        xaxis_title="Date",
        yaxis_title="Mentions",
        template="plotly_white",
    )

    return fig



# --- MOMENTUM CALLBACK (z-scored EMA of log-volume-weighted sentiment) ---
@app.callback(
    Output("momentum-graph", "figure"),
    [
        Input("subject-dropdown", "value"),
        Input("momentum-range-mode", "value"),
        Input("momentum-custom-range", "start_date"),
        Input("momentum-custom-range", "end_date"),
    ],
)
def update_momentum_chart(subject, mode, custom_start, custom_end):
    fig = go.Figure()

    # Guardrails: need a subject selected
    if not subject:
        fig.update_layout(
            title="Select a subject to view momentum.",
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Z-Score vs 90-day baseline",
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot")
        return fig

    # Determine date range using your existing helper
    # (Assumes you already have a function date_range_from_mode(mode, custom_start, custom_end) elsewhere in your file.)
    try:
        start, end = date_range_from_mode(mode, custom_start, custom_end)
    except NameError:
        # Fallback: if helper not present, default to last 30 days
        import pandas as pd
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=30)

    # Fetch raw daily aggregates for this subject
    df_raw = fetch_momentum_df(subject=subject, start_date=str(start.date()), end_date=str(end.date()))

    # Normalize + sanity checks
    if df_raw.empty or not {"Subject", "ActivityDate", "MentionCount", "AvgSentiment"}.issubset(df_raw.columns):
        fig.update_layout(
            title="No momentum data available for the selected range.",
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Z-Score vs 90-day baseline",
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot")
        return fig

    # Compute momentum features
    df = compute_momentum(df_raw)
    df = df.sort_values("ActivityDate")

    # Diverging bars of z-score momentum
    fig.add_bar(
        name=f"{subject} — z-momentum",
        x=df["ActivityDate"],
        y=df["z_momentum"],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Z-Momentum: %{y:.2f}<br>"
            "Mentions: %{customdata[1]:.0f}<br>"
            "Avg Sentiment: %{customdata[2]:.2f}<br>"
            "Base: %{customdata[3]:.3f}<br>"
            "EMA7: %{customdata[4]:.3f}<br>"
            "<extra></extra>"
        ),
        customdata=np.stack([
            df["Subject"].astype(str).values,
            df["MentionCount"].fillna(0).values,
            df["AvgSentiment"].fillna(0).values,
            df["base"].fillna(0).values,
            df["ema_fast"].fillna(0).values,
        ], axis=-1),
    )

    # Acceleration overlay (EMA7 − EMA21)
    fig.add_trace(go.Scatter(
        name=f"{subject} — accel (EMA7−EMA21)",
        x=df["ActivityDate"],
        y=df["accel"],
        mode="lines",
        hovertemplate="Acceleration: %{y:.3f}<extra></extra>",
    ))

    # Layout polish
    fig.update_layout(
        title="Momentum (z-scored EMA of log-volume-weighted sentiment)",
        xaxis_title="Date",
        yaxis_title="Z-Score vs 90-day baseline",
        barmode="overlay",
        bargap=0.15,
        legend_orientation="h",
        legend_y=-0.2,
        hovermode="x unified",
        template="plotly_white",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot")
    return fig


@app.callback(
    Output("subject-dropdown", "value"),
    [Input("page-load-once", "n_intervals")],
    prevent_initial_call=True
)
def load_default_subject(_):
    if _ is None:
        raise PreventUpdate

    uid = current_user_id()
    try:
        if uid:
            with engine.begin() as conn:
                row = conn.execute(
                    text("SELECT default_subject FROM dbo.UserPreferences WHERE user_id = :u"),
                    {"u": uid}
                ).fetchone()
            if row and row[0]:
                return row[0]
        # no user or no row: let the next callback try localStorage
        raise PreventUpdate
    except Exception:
        # Fail safe: don't block page
        raise PreventUpdate
    
@app.callback(
    Output("subject-dropdown", "value", allow_duplicate=True),
    [Input("page-load-once", "n_intervals"),
     Input("local-default-subject", "data")],
    prevent_initial_call=True
)
def load_from_local(_, local_data):
    # If we've already set a value (e.g., from DB), don't overwrite.
    # Dash doesn't give us the current 'value' here cleanly, so we just
    # return None unless we have a local value and the initial render likely hasn't set one yet.
    if not local_data or not isinstance(local_data, dict):
        raise PreventUpdate
    return local_data.get("default_subject") or dash.no_update

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


if __name__ == "__main__":
    app.run(debug=False)















































































