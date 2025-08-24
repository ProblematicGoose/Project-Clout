import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import urllib.parse
import urllib.request
import json
from datetime import datetime, timedelta, date

# -----------------------------
# Config / Endpoints
# -----------------------------
BASE_URL = "https://58dc8b387f7b.ngrok-free.app"
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
# Helpers
# -----------------------------

def fetch_df(url: str, timeout: int = 15) -> pd.DataFrame:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return pd.DataFrame(json.load(r))
    except Exception:
        return pd.DataFrame()


def fetch_json(url: str, timeout: int = 15) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return {}


def fetch_df_with_params(base_url: str, params: dict) -> pd.DataFrame:
    """Attempt to call API with query parameters; falls back to plain fetch if needed."""
    try:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{base_url}?{query}"
        return fetch_df(url)
    except Exception:
        return fetch_df(base_url)


def start_of_today() -> datetime:
    now = datetime.now()
    return datetime(year=now.year, month=now.month, day=now.day)


def date_range_from_mode(mode: str, custom_start: str | None, custom_end: str | None) -> tuple[datetime, datetime]:
    """Return (start_datetime, end_datetime) based on mode.
    custom_* are ISO date strings from DatePickerRange like '2025-08-23'.
    """
    now = datetime.now()
    if mode == "Today":
        start = start_of_today()
        end = now
    elif mode == "This Week":
        # start of week = Monday
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
        # If user hasn't set both yet, default to last 30 days
        try:
            start_date = datetime.fromisoformat(custom_start) if custom_start else (now - timedelta(days=30))
        except Exception:
            start_date = now - timedelta(days=30)
        try:
            # DatePickerRange gives a date with no time; set end to end-of-day
            end_date = datetime.fromisoformat(custom_end) if custom_end else now
        except Exception:
            end_date = now
        # ensure ordering
        if end_date < start_date:
            start_date, end_date = end_date, start_date
        # include the whole end day
        end = end_date + timedelta(days=1) - timedelta(seconds=1)
        start = start_date
    else:
        # Safe default: last 30 days
        start = now - timedelta(days=30)
        end = now
    return start, end


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
# Layout
# -----------------------------
app.layout = html.Div([
    html.H1("Sentiment Dashboard", style={"textAlign": "center", "paddingTop": "20px"}),
    # Subject selector
    html.Div([
        dcc.Dropdown(
            id="subject-dropdown",
            options=[{"label": s, "value": s} for s in subjects],
            value=subjects[0] if subjects else None,
            style={"width": "100%"},
            placeholder="Select a subject...",
        )
    ], className="dropdown-wrapper"),

    # Grid holds the cards
    html.Div(id="dashboard-grid", className="dashboard-grid"),
])

# -----------------------------
# Dashboard (subject-dependent) structure
# -----------------------------
@app.callback(
    Output("dashboard-grid", "children"),
    Input("subject-dropdown", "value"),
)
def render_dashboard(subject):
    if subject is None:
        return [html.Div("Choose a subject to begin.", className="dashboard-card")]

    cards = []

    # Scorecard
    scorecard = fetch_df(SCORECARD_URL)
    photo = fetch_df(PHOTOS_URL)
    score, office, party, state, photo_url = 5000, "", "", "", None
    if not scorecard.empty and "Subject" in scorecard.columns:
        row = scorecard[scorecard["Subject"] == subject]
        if not row.empty and "NormalizedSentimentScore" in row.columns:
            score = int(row["NormalizedSentimentScore"].iloc[0])
    if not photo.empty and "Subject" in photo.columns:
        meta = photo[photo["Subject"] == subject]
        if not meta.empty:
            photo_url = meta.get("PhotoURL", pd.Series([None])).iloc[0]
            office = meta.get("OfficeTitle", pd.Series([""])).iloc[0]
            party = meta.get("Party", pd.Series([""])).iloc[0]
            state = meta.get("State", pd.Series([""])).iloc[0]

    color = "green" if score > 6000 else "crimson" if score < 4000 else "orange"
    cards.append(
        html.Div(
            [
                html.Div([
                    html.Img(src=photo_url, className="scorecard-img")
                    if photo_url
                    else html.Div(style={"width": "100px", "height": "120px", "background": "#eee"})
                ]),
                html.Div([
                    html.H1(subject),
                    html.Div(f"{office} • {party} • {state}"),
                    html.Div("Sentiment Score", className="section-header"),
                    html.Div(f"{score:,}", style={"fontSize": "40px", "color": color, "fontWeight": "bold"}),
                ], className="scorecard-metadata"),
            ],
            className="dashboard-card scorecard-container",
        )
    )

    # Traits
    traits = fetch_df(TRAITS_URL)
    pos_list, neg_list = [], []
    if not traits.empty and "Subject" in traits.columns:
        tsub = traits[traits["Subject"] == subject]
        if not tsub.empty:
            if {"TraitType", "TraitRank", "TraitDescription"}.issubset(tsub.columns):
                pos_list = (
                    tsub[tsub["TraitType"] == "Positive"].sort_values("TraitRank")["TraitDescription"].tolist()
                )
                neg_list = (
                    tsub[tsub["TraitType"] == "Negative"].sort_values("TraitRank")["TraitDescription"].tolist()
                )
    cards.append(
        html.Div(
            [
                html.H2("Behavioral Traits", className="center-text"),
                html.Div([
                    html.H3("People like it when I...", style={"color": "green"}),
                    html.Ul([html.Li(p) for p in pos_list]) if pos_list else html.Div("No positive traits found."),
                ], style={"marginBottom": "20px"}),
                html.Div([
                    html.H3("People don't like it when I...", style={"color": "crimson"}),
                    html.Ul([html.Li(n) for n in neg_list]) if neg_list else html.Div("No negative traits found."),
                ]),
            ],
            className="dashboard-card",
        )
    )

    # Bill Sentiment
    bills = fetch_df(BILL_SENTIMENT_URL)
    if bills.empty:
        cards.append(html.Div("No bill sentiment data available.", className="dashboard-card"))
    else:
        cards.append(
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
                                            html.Td(row.get("BillName", "")),
                                            html.Td(round(row.get("AverageSentimentScore", 0), 2)),
                                            html.Td(row.get("SentimentLabel", "")),
                                        ]
                                    )
                                    for _, row in bills.iterrows()
                                ]
                            ),
                        ],
                        style={"width": "100%"},
                    ),
                ],
                className="dashboard-card",
            )
        )

    # -----------------------------
    # Sentiment Over Time (with timeframe controls)
    # -----------------------------
    cards.append(
        html.Div(
            [
                html.H2("Sentiment Over Time", className="center-text"),
                html.Div(
                    [
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
                    ],
                    className="control-row",
                ),
                dcc.Loading(dcc.Graph(id="sentiment-graph", style={"height": "400px"})),
            ],
            className="dashboard-card",
        )
    )

    # -----------------------------
    # Top Issues (static)
    # -----------------------------
    issues = fetch_json(TOP_ISSUES_URL)
    if issues and all(k in issues for k in ("Liberal", "Conservative", "WeekStartDate")):
        week = issues["WeekStartDate"]
        liberal = issues["Liberal"]
        conservative = issues["Conservative"]
        cards.append(
            html.Div(
                [
                    html.H2(f"Top Issues (Week of {week})", className="center-text"),
                    html.Div(
                        [
                            html.Div([
                                html.H3("Conservative Topics", style={"color": "crimson"}),
                                html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for t in conservative]),
                            ], style={"width": "45%", "display": "inline-block"}),
                            html.Div(
                                [
                                    html.H3("Liberal Topics", style={"color": "blue"}),
                                    html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for t in liberal]),
                                ],
                                style={"width": "45%", "display": "inline-block", "marginLeft": "5%"},
                            ),
                        ]
                    ),
                ],
                className="dashboard-card",
            )
        )

    # -----------------------------
    # Common Ground (static)
    # -----------------------------
    common_df = fetch_df(COMMON_GROUND_URL)
    if not common_df.empty and "Subject" in common_df.columns:
        filtered = common_df[common_df["Subject"] == subject]
        if not filtered.empty:
            cards.append(
                html.Div(
                    [
                        html.H2("Common Ground Issues", className="center-text"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Span(f"{r.get('IssueRank', '')}. ", style={"fontWeight": "bold"}),
                                        html.Span(f"{r.get('Issue', '')}: ", style={"fontWeight": "bold"}),
                                        html.Span(r.get("Explanation", "")),
                                    ]
                                )
                                for _, r in filtered.iterrows()
                            ]
                        ),
                    ],
                    className="dashboard-card",
                )
            )

    # -----------------------------
    # Mentions (with timeframe controls)
    # -----------------------------
    cards.append(
        html.Div(
            [
                html.H2("Mentions by Subject", className="center-text"),
                html.Div(
                    [
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
                    ],
                    className="control-row",
                ),
                dcc.Loading(dcc.Graph(id="mentions-graph", style={"height": "400px"})),
            ],
            className="dashboard-card",
        )
    )

    # -----------------------------
    # Momentum (with timeframe controls)
    # -----------------------------
    cards.append(
        html.Div(
            [
                html.H2("Momentum by Subject", className="center-text"),
                html.Div(
                    [
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
                    ],
                    className="control-row",
                ),
                dcc.Loading(dcc.Graph(id="momentum-graph", style={"height": "400px"})),
            ],
            className="dashboard-card",
        )
    )

    return cards


# -----------------------------
# Interactivity: Enable/disable custom date pickers based on mode
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
# Chart callbacks
# -----------------------------
# Sentiment Over Time
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
    start, end = date_range_from_mode(mode, custom_start, custom_end)

    df = fetch_df_with_params(TIMESERIES_URL, {
        "subject": subject,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
    })

    if df.empty:
        # fall back to plain endpoint and client-side filter if necessary
        df = fetch_df(TIMESERIES_URL)

    # Normalize columns
    if not df.empty:
        # expected columns: Subject, SentimentDate, NormalizedSentimentScore
        if "SentimentDate" in df.columns:
            df["SentimentDate"] = pd.to_datetime(df["SentimentDate"], errors="coerce")
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "SentimentDate"})
            df["SentimentDate"] = pd.to_datetime(df["SentimentDate"], errors="coerce")

        if "Subject" in df.columns:
            df = df[df["Subject"] == subject]

        # filter by date window if we have a date column
        if "SentimentDate" in df.columns:
            mask = (df["SentimentDate"] >= start) & (df["SentimentDate"] <= end)
            df = df[mask]

    fig = go.Figure()
    if not df.empty and {"SentimentDate", "NormalizedSentimentScore"}.issubset(df.columns):
        df = df.sort_values("SentimentDate")
        fig.add_trace(
            go.Scatter(
                x=df["SentimentDate"],
                y=df["NormalizedSentimentScore"],
                mode="lines+markers",
                name=subject,
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


# Mentions chart
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
    start, end = date_range_from_mode(mode, custom_start, custom_end)

    # Try hitting API with params first
    df = fetch_df_with_params(MENTION_COUNT_URL, {
        "subject": subject,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
    })

    # If the API doesn't return dated data, try deriving from time series as a fallback
    if df.empty or "MentionCount" not in df.columns:
        ts = fetch_df(TIMESERIES_URL)
        if not ts.empty:
            # attempt to compute counts from any available date column
            date_col = None
            for c in ["CreatedUTC", "MentionDate", "SentimentDate", "Date"]:
                if c in ts.columns:
                    date_col = c
                    break
            if date_col is not None:
                ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
                sub = ts[(ts.get("Subject", pd.Series()).eq(subject)) & (ts[date_col].between(start, end))]
                mention_count = len(sub)
                df = pd.DataFrame({"Subject": [subject], "MentionCount": [mention_count]})

    # Build bar chart (single bar for the selected subject)
    fig = go.Figure()
    if not df.empty and "MentionCount" in df.columns:
        # If multiple subjects returned, filter to selected one
        if "Subject" in df.columns:
            row = df[df["Subject"] == subject]
            if not row.empty:
                count = int(row.iloc[0]["MentionCount"])
            else:
                count = int(df["MentionCount"].iloc[0])
        else:
            count = int(df["MentionCount"].iloc[0])
        fig.add_trace(go.Bar(x=[subject], y=[count], name="Mentions"))
        fig.update_layout(
            title=f"Mentions ({mode})",
            yaxis_title="Mentions",
            xaxis_title="Subject",
            template="plotly_white",
        )
    else:
        fig.update_layout(title="No mention data available for the selected range.", template="plotly_white")
    return fig


# Momentum chart
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
    start, end = date_range_from_mode(mode, custom_start, custom_end)

    df = fetch_df_with_params(MOMENTUM_URL, {
        "subject": subject,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
    })

    if df.empty:
        df = fetch_df(MOMENTUM_URL)

    # Normalize likely column names
    if not df.empty:
        # Ensure subject filter
        if "Subject" in df.columns:
            df = df[df["Subject"] == subject]
        # Standardize date and value columns
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
            # date range filter
            df = df[(df["MomentumDate"] >= start) & (df["MomentumDate"] <= end)]
            # daily aggregate
            df = (
                df.groupby(df["MomentumDate"].dt.date)
                .agg({"Momentum": "mean"})
                .reset_index()
                .rename(columns={"MomentumDate": "Date"})
            )
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

    fig = go.Figure()
    if not df.empty and {"Date", "Momentum"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Momentum"], mode="lines+markers", name=subject))
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

















