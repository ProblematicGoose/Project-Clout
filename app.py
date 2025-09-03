import time
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.parse
import urllib.request
import json
from datetime import datetime, timedelta, date

# -----------------------------
# Config / Endpoints
# -----------------------------
BASE_URL = "hhttps://58d3a9f93a0e.ngrok-free.app"
SCORECARD_URL = f"{BASE_URL}/api/scorecard"
PHOTOS_URL = f"{BASE_URL}/api/subject-photos"
TIMESERIES_URL = f"{BASE_URL}/api/timeseries"
TRAITS_URL = f"{BASE_URL}/api/traits"
BILL_SENTIMENT_URL = f"{BASE_URL}/api/bill-sentiment"
TOP_ISSUES_URL = f"{BASE_URL}/api/top-issues"
COMMON_GROUND_URL = f"{BASE_URL}/api/common-ground-issues"
MENTION_COUNT_URL = f"{BASE_URL}/api/mention-counts"
MOMENTUM_URL = f"{BASE_URL}/api/momentum"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# -----------------------------
# Lightweight in-memory TTL cache (speeds up subject switches)
# -----------------------------
_TTL_SECONDS = 90  # adjust if you want longer/shorter freshness
_CACHE: dict[str, tuple[float, object]] = {}

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

# -----------------------------
# Helpers
# -----------------------------

def fetch_df(url: str, timeout: int = 15) -> pd.DataFrame:
    cached = _cache_get(f"DF::{url}")
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            df = pd.DataFrame(json.load(r))
            _cache_set(f"DF::{url}", df.copy())
            return df
    except Exception:
        return pd.DataFrame()


def fetch_json(url: str, timeout: int = 15) -> dict:
    cached = _cache_get(f"JSON::{url}")
    if isinstance(cached, dict):
        # return a shallow copy
        return json.loads(json.dumps(cached))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
            _cache_set(f"JSON::{url}", json.loads(json.dumps(data)))
            return data
    except Exception:
        return {}


def fetch_df_with_params(base_url: str, params: dict) -> pd.DataFrame:
    """Cached GET with query parameters."""
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{base_url}?{query}" if query else base_url
    return fetch_df(url)


def start_of_today() -> datetime:
    now = datetime.now()
    return datetime(year=now.year, month=now.month, day=now.day)


def date_range_from_mode(mode: str, custom_start: str | None, custom_end: str | None) -> tuple[datetime, datetime]:
    now = datetime.now()
    if mode == "Today":
        start = start_of_today()
        end = now
    elif mode == "This Week":
        monday = start_of_today() - timedelta(days=start_of_today().weekday())
        start = monday
        end = now
    elif mode == "This Month":
        first = datetime(now.year, now.month, 1)
        start = first
        end = now
    elif mode == "This Year":
        first = datetime(now.year, 1, 1)
        start = first
        end = now
    elif mode == "Custom":
        try:
            start_date = datetime.fromisoformat(custom_start) if custom_start else (now - timedelta(days=30))
        except Exception:
            start_date = now - timedelta(days=30)
        try:
            end_date = datetime.fromisoformat(custom_end) if custom_end else now
        except Exception:
            end_date = now
        if end_date < start_date:
            start_date, end_date = end_date, start_date
        end = end_date + timedelta(days=1) - timedelta(seconds=1)
        start = start_date
    else:
        start = now - timedelta(days=30)
        end = now
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


TIME_MODES = [
    {"label": "Today", "value": "Today"},
    {"label": "This Week", "value": "This Week"},
    {"label": "This Month", "value": "This Month"},
    {"label": "This Year", "value": "This Year"},
    {"label": "Custom", "value": "Custom"},
]

# Preload subjects for the subject dropdown
try:
    _scorecard_df = fetch_df(SCORECARD_URL)
    subjects = sorted(_scorecard_df["Subject"].dropna().unique()) if "Subject" in _scorecard_df.columns else []
except Exception:
    subjects = []

# -----------------------------
# Layout (single grid)
# -----------------------------
app.layout = html.Div([
    html.H1("Sentiment Dashboard", style={"textAlign": "center", "paddingTop": "20px"}),

    html.Div([
        dcc.Dropdown(
            id="subject-dropdown",
            options=[{"label": s, "value": s} for s in subjects],
            value=subjects[0] if subjects else None,
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


# -----------------------------
# Dashboard builder: dynamic cards + fixed chart cards
# -----------------------------
@app.callback(
    Output("dashboard-grid", "children"),
    Input("subject-dropdown", "value"),
)
def render_dashboard(subject):
    dynamic_cards = []

    if not subject:
        dynamic_cards.append(html.Div("Choose a subject to begin.", className="dashboard-card"))
        return dynamic_cards + chart_cards()

    # Scorecard (cached)
    scorecard = fetch_df(SCORECARD_URL)
    photo = fetch_df(PHOTOS_URL)
    score, office, party, state, photo_url = 10000, "", "", "", None
    if not scorecard.empty and "Subject" in scorecard.columns:
        row = scorecard[scorecard["Subject"].astype(str) == str(subject)]
        if not row.empty and "NormalizedSentimentScore" in row.columns:
            val = safe_first(row["NormalizedSentimentScore"], None)
            score = coerce_int(val, default=5000)
    if not photo.empty and "Subject" in photo.columns:
        meta = photo[photo["Subject"].astype(str) == str(subject)]
        if not meta.empty:
            photo_url = safe_first(meta.get("PhotoURL", pd.Series([None])), None)
            office = safe_first(meta.get("OfficeTitle", pd.Series([""])), "") or ""
            party = safe_first(meta.get("Party", pd.Series([""])), "") or ""
            state = safe_first(meta.get("State", pd.Series([""])), "") or ""

    color = "green" if score > 6000 else "crimson" if score < 4000 else "orange"
    dynamic_cards.append(
        html.Div(
            [
                html.Div([
                    html.Img(src=photo_url, className="scorecard-img")
                    if photo_url
                    else html.Div(style={"width": "100px", "height": "120px", "background": "#eee"})
                ]),
               html.Div([
        html.H1(subject, style={"fontSize": "30px", "marginBottom": "5px"}),
        html.Div(f"{office} • {party} • {state}", style={"fontSize": "16px", "color": "#666"}),
        html.Div("Sentiment Score", style={"marginTop": "10px", "fontSize": "14px", "color": "#999"}),
        html.Div(f"{score:,}", style={"fontSize": "60px", "color": color, "fontWeight": "bold"})
    ])
            ],
            className="dashboard-card scorecard-container",
        )
    )

    # Traits (cached)
            # Traits (cached) — newest 5 Positive and 5 Negative by CreatedUTC
    traits = fetch_df(TRAITS_URL)
    pos_list, neg_list = [], []
    if not traits.empty and "Subject" in traits.columns and "CreatedUTC" in traits.columns:
        subj_norm = str(subject).strip().casefold()
        subj_col = traits["Subject"].astype(str).str.strip().str.casefold()
        tsub = traits[subj_col == subj_norm].copy()

        if not tsub.empty:
            # Parse timestamps
            raw = tsub["CreatedUTC"].astype(str).str.strip()
            s = raw.str.replace("T", " ", regex=False).str.replace("Z", "", regex=False)
            s = s.str.replace(r"(\.\d{6})\d+", r"\1", regex=True)  # trim >6 microseconds
            tsub["CreatedUTC_parsed"] = pd.to_datetime(s, errors="coerce")

            def newest5(df: pd.DataFrame, kind: str) -> list[str]:
                kk = df[df["TraitType"].astype(str).str.strip().str.casefold().eq(kind.casefold())].copy()
                kk = kk[kk["CreatedUTC_parsed"].notna()]
                if kk.empty:
                    return []
                kk = kk.sort_values("CreatedUTC_parsed", ascending=False).head(5)
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
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td(r.get("BillName", "") if isinstance(r, dict) else r["BillName"]),
                                            html.Td(
                                                (lambda v: round(v, 2) if pd.notna(v) else "—")(
                                                    r.get("AverageSentimentScore", None) if isinstance(r, dict) else r["AverageSentimentScore"]
                                                )
                                            ),
                                            html.Td(r.get("SentimentLabel", "") if isinstance(r, dict) else r["SentimentLabel"]),
                                        ]
                                    )
                                    for _, r in bills.iterrows()
                                ]
                            ),
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
                    html.Div(
                        [
                            html.Div([
    html.H3("Conservative Topics", style={'color': 'crimson'}),
    html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize": "20px"}) for t in conservative]),
    html.H3("Liberal Topics", style={'color': 'blue', 'marginTop': '20px'}),
    html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}", style={"fontSize": "20px"}) for t in liberal])
])
                        ]
                    ),
                ],
                className="dashboard-card",
            )
        )

    # Common Ground (cached)
    common_df = fetch_df(COMMON_GROUND_URL)
    if not common_df.empty and "Subject" in common_df.columns:
        filtered = common_df[common_df["Subject"].astype(str) == str(subject)]
        if not filtered.empty:
            dynamic_cards.append(
                html.Div(
    [
        html.H2("Common Ground Issues", className="center-text"),
        html.Ul(
            [
                html.Li(
                    [
                        html.Span(f"{r.get('IssueRank', '')}. ", style={"fontWeight": ""}),
                        html.Span(f"{r.get('Issue', '')}: ", style={"fontWeight": ""}),
                        html.Span(r.get("Explanation", ""))
                    ],
                    className="common-ground-item"
                )
                for _, r in filtered.iterrows()
            ],
            className="common-ground-list"
        ),
    ],
    className="dashboard-card"
)

            )

    return dynamic_cards + chart_cards()


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
@app.callback(
    Output("sentiment-graph", "figure"),
    [
        Input("subject-dropdown", "value"),
        Input("sentiment-range-mode", "value"),
        Input("sentiment-custom-range", "start_date"),
        Input("sentiment-custom-range", "end_date"),
    ],
)
def update_sentiment_chart(subject, mode, custom_start, custom_end):
    fig = go.Figure()
    if not subject:
        fig.update_layout(title="Select a subject to view data.", template="plotly_white")
        return fig

    start, end = date_range_from_mode(mode, custom_start, custom_end)

    # Fetch subject-wide data once; filter locally for speed
    base_df = fetch_df_with_params(TIMESERIES_URL, {"subject": subject})
    df = base_df.copy() if not base_df.empty else fetch_df(TIMESERIES_URL)

    if not df.empty:
        if "SentimentDate" in df.columns:
            df["SentimentDate"] = pd.to_datetime(df["SentimentDate"], errors="coerce")
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "SentimentDate"})
            df["SentimentDate"] = pd.to_datetime(df["SentimentDate"], errors="coerce")

        if "Subject" in df.columns:
            df = df[df["Subject"].astype(str) == str(subject)]

        if "SentimentDate" in df.columns:
            mask = (df["SentimentDate"] >= start) & (df["SentimentDate"] <= end)
            df = df[mask]

    if not df.empty and {"SentimentDate", "NormalizedSentimentScore"}.issubset(df.columns):
        df = df.sort_values("SentimentDate")
        fig.add_trace(
            go.Scatter(
                x=df["SentimentDate"],
                y=df["NormalizedSentimentScore"],
                mode="lines+markers",
                name=str(subject),
            )
        )
        fig.update_layout(
            title=f"Sentiment Over Time ({mode})",
            xaxis_title="Date",
            yaxis_title="Score",
            yaxis=dict(range=[0, 10000]),
            template="plotly_white",
        )
    else:
        fig.update_layout(
            title="No sentiment data available for the selected range.",
            template="plotly_white",
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

    # Try mention counts endpoint (cached); if not present, derive from subject timeseries
    df = fetch_df_with_params(MENTION_COUNT_URL, {"subject": subject})
    if df.empty or "MentionCount" not in df.columns:
        ts = fetch_df_with_params(TIMESERIES_URL, {"subject": subject})
        if not ts.empty:
            date_col = None
            for c in ["CreatedUTC", "MentionDate", "SentimentDate", "Date"]:
                if c in ts.columns:
                    date_col = c
                    break
            if date_col is not None:
                ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
                sub = ts[(ts.get("Subject", pd.Series()).astype(str).eq(str(subject))) & (ts[date_col].between(start, end))]
                mention_count = int(len(sub))
                df = pd.DataFrame({"Subject": [subject], "MentionCount": [mention_count]})

    if not df.empty and "MentionCount" in df.columns:
        if "Subject" in df.columns:
            row = df[df["Subject"].astype(str) == str(subject)]
            count = coerce_int(safe_first(row["MentionCount"], 0), default=0) if not row.empty else coerce_int(safe_first(df["MentionCount"], 0), default=0)
        else:
            count = coerce_int(safe_first(df["MentionCount"], 0), default=0)
        fig.add_trace(go.Bar(x=[str(subject)], y=[count], name="Mentions"))
        fig.update_layout(
            title=f"Mentions ({mode})",
            yaxis_title="Mentions",
            xaxis_title="Subject",
            template="plotly_white",
        )
    else:
        fig.update_layout(title="No mention data available for the selected range.", template="plotly_white")
    return fig


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
    if not subject:
        fig.update_layout(title="Select a subject to view data.", template="plotly_white")
        return fig

    start, end = date_range_from_mode(mode, custom_start, custom_end)

    # Fetch per-subject momentum once; filter locally for speed
    df = fetch_df_with_params(MOMENTUM_URL, {"subject": subject})
    if df.empty:
        df = fetch_df(MOMENTUM_URL)

    if not df.empty:
        if "Subject" in df.columns:
            df = df[df["Subject"].astype(str) == str(subject)]
        if "MomentumDate" not in df.columns:
            for c in ["ActivityDate", "Date"]:
                if c in df.columns:
                    df = df.rename(columns={c: "MomentumDate"})
                    break
        if "Momentum" not in df.columns:
            for c in ["MomentumScore", "Value", "Score"]:
                if c in df.columns:
                    df = df.rename(columns={c: "Momentum"})
                    break
        if "MomentumDate" in df.columns:
            df["MomentumDate"] = pd.to_datetime(df["MomentumDate"], errors="coerce")
            df = df.dropna(subset=["MomentumDate"])
            df = df[(df["MomentumDate"] >= start) & (df["MomentumDate"] <= end)]
            df = (
                df.groupby(df["MomentumDate"].dt.date)
                .agg({"Momentum": "mean"})
                .reset_index()
                .rename(columns={"MomentumDate": "Date"})
            )
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

    if not df.empty and {"Date", "Momentum"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Momentum"], mode="lines+markers", name=str(subject)))
        fig.update_layout(
            title=f"Momentum = Mentions × Avg Sentiment ({mode})",
            xaxis_title="Date",
            yaxis_title="Momentum",
            template="plotly_white",
        )
    else:
        fig.update_layout(title="No momentum data available for the selected range.", template="plotly_white")
    return fig


if __name__ == "__main__":
    app.run(debug=True)
















































































