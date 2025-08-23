import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json
from datetime import datetime, timedelta

# --------------------------- CONFIG ---------------------------
BASE_URL = "https://e8eb17633693.ngrok-free.app"
ENDPOINTS = {
    "scorecard": f"{BASE_URL}/api/scorecard",
    "timeseries": f"{BASE_URL}/api/timeseries",
    "traits": f"{BASE_URL}/api/traits",
    "bills": f"{BASE_URL}/api/bill-sentiment",
    "issues": f"{BASE_URL}/api/top-issues",
    "common": f"{BASE_URL}/api/common-ground-issues",
    "photos": f"{BASE_URL}/api/subject-photos",
    "mentions": f"{BASE_URL}/api/mention-counts",
    "momentum": f"{BASE_URL}/api/momentum"
}

TIME_RANGES = {
    "Today": (datetime.now(), datetime.now()),
    "This Week": (datetime.now() - timedelta(days=datetime.now().weekday()), datetime.now()),
    "This Month": (datetime(datetime.now().year, datetime.now().month, 1), datetime.now()),
    "This Year": (datetime(datetime.now().year, 1, 1), datetime.now())
}

# --------------------------- INIT APP ---------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --------------------------- HELPERS ---------------------------
def fetch_df(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return pd.DataFrame(json.load(r))
    except Exception as e:
        print(f"Fetch failed for {url}: {e}")
        return pd.DataFrame()

def load_subjects():
    try:
        df = fetch_df(ENDPOINTS["scorecard"])
        return sorted(df["Subject"].dropna().unique())
    except:
        return []

# --------------------------- LAYOUT ---------------------------
subjects = load_subjects()

app.layout = html.Div([
    html.Div([
        html.H1("Sentiment Dashboard", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{"label": s, "value": s} for s in subjects],
            value=subjects[0] if subjects else None,
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '40px'}),

    html.Div(id='scorecard-div', className='card'),
    html.Div([dcc.Graph(id='timeseries-graph')], className='card'),
    html.Div(id='traits-div', className='card'),
    html.Div(id='bill-sentiment-table', className='card'),
    html.Div(id='top-issues-div', className='card'),
    html.Div(id='common-ground-div', className='card')
])

# --------------------------- CALLBACK ---------------------------
@app.callback(
    Output('scorecard-div', 'children'),
    Output('timeseries-graph', 'figure'),
    Output('traits-div', 'children'),
    Output('bill-sentiment-table', 'children'),
    Output('top-issues-div', 'children'),
    Output('common-ground-div', 'children'),
    Input('subject-dropdown', 'value')
)
def update_dashboard(subject):
    # --- Scorecard ---
    score = 5000
    scorecard = fetch_df(ENDPOINTS["scorecard"])
    row = scorecard[scorecard.Subject == subject]
    if not row.empty:
        score = int(row.NormalizedSentimentScore.iloc[0])

    photo = fetch_df(ENDPOINTS["photos"])
    meta = photo[photo.Subject == subject]
    photo_url, office, party, state = None, '', '', ''
    if not meta.empty:
        photo_url = meta.PhotoURL.iloc[0]
        office, party, state = meta.OfficeTitle.iloc[0], meta.Party.iloc[0], meta.State.iloc[0]

    color = 'green' if score > 6000 else 'crimson' if score < 4000 else 'orange'
    scorecard_div = html.Div([
        html.Img(src=photo_url, style={'width': '140px', 'height': '170px'}) if photo_url else html.Div(style={'width': '140px', 'height': '170px', 'background': '#eee'}),
        html.Div([
            html.H1(subject),
            html.Div(f"{office} • {party} • {state}"),
            html.Div("Sentiment Score", style={'marginTop': '14px'}),
            html.Div(f"{score:,}", style={'fontSize': '56px', 'color': color, 'fontWeight': 'bold'})
        ])
    ], style={'display': 'flex', 'gap': '24px'})

    # --- Timeseries ---
    df = fetch_df(ENDPOINTS["timeseries"])
    df['SentimentDate'] = pd.to_datetime(df['SentimentDate'])
    ts = df[df.Subject == subject]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts['SentimentDate'], y=ts['NormalizedSentimentScore'], mode='lines+markers'))
    fig.update_layout(title="Sentiment Over Time", yaxis_range=[0, 10000], template='plotly_white')

    # --- Traits ---
    tr = fetch_df(ENDPOINTS["traits"])
    tr = tr[tr.Subject == subject]
    pos = tr[tr.TraitType == 'Positive'].sort_values('TraitRank').TraitDescription.tolist()
    neg = tr[tr.TraitType == 'Negative'].sort_values('TraitRank').TraitDescription.tolist()
    traits_div = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(p) for p in pos]),
        html.H2("People don't like it when I...", style={'color': 'crimson'}),
        html.Ul([html.Li(n) for n in neg])
    ])

    # --- Bills ---
    bills = fetch_df(ENDPOINTS["bills"])
    bill_table = html.Div([
        html.H2("Public Sentiment Toward National Bills", style={'textAlign': 'center'}),
        html.Table([
            html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
            html.Tbody([
                html.Tr([html.Td(r.BillName), html.Td(round(r.AverageSentimentScore, 2)), html.Td(r.SentimentLabel)]) for _, r in bills.iterrows()
            ])
        ], style={'width': '80%', 'margin': '0 auto'})
    ])

    # --- Top Issues ---
    issues = fetch_df(ENDPOINTS["issues"])
    week = issues.WeekStartDate.iloc[0] if not issues.empty else ""
    top_issues_div = html.Div([
        html.H2(f"Top Issues for the Week of {week}"),
        html.Div([
            html.Div([
                html.H3("Conservative Topics", style={'color': 'crimson'}),
                html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for _, t in issues[issues.IdeologyLabel=='Conservative'].iterrows()])
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.H3("Liberal Topics", style={'color': 'blue'}),
                html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for _, t in issues[issues.IdeologyLabel=='Liberal'].iterrows()])
            ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'})
        ])
    ])

    # --- Common Ground ---
    cg = fetch_df(ENDPOINTS["common"])
    cg = cg[cg.Subject.str.lower() == subject.lower()].sort_values('IssueRank')
    if cg.empty:
        common_ground_div = html.Div([html.H3("No common ground issues found.", style={'textAlign': 'center'})])
    else:
        common_ground_div = html.Div([
            html.H2("Issues to focus on to win over moderates"),
            html.Ul([
                html.Li([
                    html.Span(f"{r.IssueRank}. ", style={'fontWeight': 'bold'}),
                    html.Span(f"{r.Issue}: ", style={'fontWeight': 'bold'}),
                    html.Span(r.Explanation)
                ]) for _, r in cg.iterrows()
            ])
        ])

    return scorecard_div, fig, traits_div, bill_table, top_issues_div, common_ground_div

if __name__ == '__main__':
    app.run(debug=False)









