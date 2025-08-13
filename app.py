# app.py
import os
import requests
import pandas as pd
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output, State

# ============ CONFIG ============
# Point this to your Flask API. Example:
#   set BASE_API=https://<your-ngrok>.ngrok-free.app
# or leave unset to use localhost:5057
BASE_API = os.getenv("https://b686a735fbe9.ngrok-free.app", "http://localhost:5057")

SCORECARD_URL     = f"{BASE_API}/api/scorecard"
TIMESERIES_URL    = f"{BASE_API}/api/timeseries"
TRAITS_URL        = f"{BASE_API}/api/traits"
BILL_SENTIMENT_URL= f"{BASE_API}/api/bill-sentiment"
TOP_ISSUES_URL    = f"{BASE_API}/api/top-issues"

REQ_TIMEOUT = 12  # seconds

# ============ APP INIT ============
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

def get_json(url, default=None):
    try:
        r = requests.get(url, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] GET {url}: {e}")
        return default if default is not None else {}

# Load subjects for dropdown (tolerant of API hiccups)
scorecard_boot = get_json(SCORECARD_URL, default=[])
subjects = sorted({(row.get("Subject") or "").strip()
                   for row in scorecard_boot if row.get("Subject")}) or ["(no data)"]
default_subject = subjects[0]

# ============ LAYOUT ============
app.layout = html.Div([
    html.Div([
        html.H1("Sentiment Dashboard", style={'textAlign': 'center', 'marginBottom': '8px'}),
        html.Div([
            dcc.Dropdown(
                id='subject-dropdown',
                options=[{'label': s, 'value': s} for s in subjects],
                value=default_subject,
                style={'width': '60%'}
            ),
            dcc.Interval(id="refresh-interval", interval=60*1000, n_intervals=0)  # optional 60s refresh
        ], style={'display': 'flex', 'gap': '16px', 'justifyContent': 'center', 'alignItems': 'center'})
    ], style={'marginBottom': '24px'}),

    # Scorecard
    html.Div(id='scorecard-div', className='card',
             style={'padding': '16px', 'marginBottom': '24px', 'textAlign': 'center',
                    'border': '1px solid #eee', 'borderRadius': '12px'}),

    # Time series
    html.Div([
        dcc.Graph(id='timeseries-graph')
    ], className='card', style={'padding': '8px', 'marginBottom': '24px',
                                'border': '1px solid #eee', 'borderRadius': '12px'}),

    # Traits
    html.Div(id='traits-div', className='card',
             style={'padding': '16px', 'marginBottom': '24px',
                    'border': '1px solid #eee', 'borderRadius': '12px'}),

    # Top issues this week (new)
    html.Div([
        html.H2("Top Issues This Week", style={'textAlign': 'center', 'marginBottom': '16px'}),
        html.Div([
            html.Div([
                html.H3("Conservative", style={'textAlign': 'center', 'marginBottom': '8px'}),
                dcc.Graph(id='top-issues-conservative')
            ], style={'flex': 1, 'padding': '8px', 'minWidth': '300px'}),
            html.Div([
                html.H3("Liberal", style={'textAlign': 'center', 'marginBottom': '8px'}),
                dcc.Graph(id='top-issues-liberal')
            ], style={'flex': 1, 'padding': '8px', 'minWidth': '300px'}),
        ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}),
        html.Div(id='top-issues-week-label', style={'textAlign': 'center', 'color': '#666', 'marginTop': '-4px'})
    ], className='card', style={'padding': '16px', 'marginBottom': '24px',
                                'border': '1px solid #eee', 'borderRadius': '12px'}),

    # Bill sentiment table
    html.Div(id='bill-sentiment-table', className='card',
             style={'padding': '16px', 'marginBottom': '24px',
                    'border': '1px solid #eee', 'borderRadius': '12px'})
])


# ============ CALLBACK ============
@app.callback(
    Output('scorecard-div', 'children'),
    Output('timeseries-graph', 'figure'),
    Output('traits-div', 'children'),
    Output('bill-sentiment-table', 'children'),
    Output('top-issues-conservative', 'figure'),
    Output('top-issues-liberal', 'figure'),
    Output('top-issues-week-label', 'children'),
    Input('subject-dropdown', 'value'),
    Input('refresh-interval', 'n_intervals')
)
def update_dashboard(selected_subject, _tick):
    # -------- SCORECARD --------
    scorecard_data = get_json(SCORECARD_URL, default=[])
    df_scorecard = pd.DataFrame(scorecard_data) if scorecard_data else pd.DataFrame()
    score = 5000
    if not df_scorecard.empty and 'Subject' in df_scorecard.columns:
        row = df_scorecard[df_scorecard['Subject'] == selected_subject]
        if not row.empty:
            try:
                score = int(row['NormalizedSentimentScore'].iloc[0])
            except Exception:
                pass

    if score < 4000:
        score_color = 'crimson'
    elif score > 6000:
        score_color = 'green'
    else:
        score_color = 'orange'

    scorecard_display = html.Div([
        html.H1(selected_subject, style={'fontSize': '46px', 'fontWeight': 'bold', 'margin': 0}),
        html.Div("Sentiment Score", style={'fontSize': '22px', 'color': 'gray'}),
        html.Div(f"{score:,}", style={'fontSize': '64px', 'color': score_color, 'fontWeight': 'bold'})
    ])

    # -------- TIME SERIES --------
    timeseries_data = get_json(TIMESERIES_URL, default=[])
    df_ts = pd.DataFrame(timeseries_data) if timeseries_data else pd.DataFrame(columns=['SentimentDate','Subject','NormalizedSentimentScore'])
    if not df_ts.empty and 'SentimentDate' in df_ts.columns:
        df_ts['SentimentDate'] = pd.to_datetime(df_ts['SentimentDate'], errors='coerce')
        df_ts = df_ts.dropna(subset=['SentimentDate'])
    df_sel = df_ts[df_ts['Subject'] == selected_subject] if not df_ts.empty else pd.DataFrame(columns=df_ts.columns)

    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(
        x=df_sel['SentimentDate'],
        y=df_sel['NormalizedSentimentScore'],
        mode='lines+markers',
        name=selected_subject
    ))
    ts_fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Normalized Sentiment Score (0–10,000)",
        yaxis=dict(range=[0, 10000]),
        template='plotly_white',
        margin=dict(l=40, r=20, t=50, b=40)
    )

    # -------- TRAITS --------
    traits_data = get_json(TRAITS_URL, default=[])
    df_traits = pd.DataFrame(traits_data) if traits_data else pd.DataFrame(columns=['Subject','TraitType','TraitRank','TraitDescription'])
    df_traits = df_traits[df_traits['Subject'] == selected_subject] if not df_traits.empty else df_traits

    pos = (df_traits[df_traits['TraitType'] == 'Positive']
           .sort_values('TraitRank')['TraitDescription'].tolist()) if not df_traits.empty else []
    neg = (df_traits[df_traits['TraitType'] == 'Negative']
           .sort_values('TraitRank')['TraitDescription'].tolist()) if not df_traits.empty else []

    traits_section = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(t, style={'textAlign': 'left'}) for t in pos], style={'listStyleType': 'none', 'paddingLeft': 0}),
        html.H2("People don't like it when I...", style={'color': 'crimson', 'marginTop': '20px'}),
        html.Ul([html.Li(t, style={'textAlign': 'left'}) for t in neg], style={'listStyleType': 'none', 'paddingLeft': 0})
    ])

    # -------- BILL SENTIMENT TABLE --------
    bill_data = get_json(BILL_SENTIMENT_URL, default=[])
    df_bills = pd.DataFrame(bill_data) if bill_data else pd.DataFrame(columns=['BillName','AverageSentimentScore','SentimentLabel'])

    bill_rows = []
    for _, r in df_bills.iterrows():
        bill_rows.append(html.Tr([
            html.Td(r.get('BillName', '')),
            html.Td(f"{float(r.get('AverageSentimentScore', 0)):.2f}", style={'paddingRight': '24px'}),
            html.Td(r.get('SentimentLabel', ''))
        ]))

    bill_table = html.Div([
        html.H2("Public Sentiment Toward National Bills", style={'textAlign': 'center', 'marginBottom': '12px'}),
        html.Table([
            html.Thead(html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])),
            html.Tbody(bill_rows or [html.Tr([html.Td("No data"), html.Td(""), html.Td("")])])
        ], style={'width': '90%', 'margin': '0 auto', 'borderCollapse': 'collapse'})
    ])

    # -------- TOP ISSUES (NEW) --------
    top_issues_json = get_json(TOP_ISSUES_URL, default={"weekStartDate": None, "data": {"Conservative": [], "Liberal": []}})
    week_label = top_issues_json.get("weekStartDate") or None
    data = top_issues_json.get("data") or {}
    cons_list = data.get("Conservative", [])
    lib_list  = data.get("Liberal", [])

    # Convert to DataFrames (ranked)
    df_cons = pd.DataFrame(cons_list)
    df_lib  = pd.DataFrame(lib_list)
    if not df_cons.empty:
        df_cons = df_cons.sort_values("rank")
    if not df_lib.empty:
        df_lib = df_lib.sort_values("rank")

    # Horizontal bar charts with rank ascending (1 at top)
    cons_fig = go.Figure()
    if not df_cons.empty:
        cons_fig.add_trace(go.Bar(
            x=[1]*len(df_cons),  # dummy width to show equal bars if you prefer rank-based no-weight
            y=df_cons['topic'],
            orientation='h',
            text=[f"#{r}" for r in df_cons['rank']],
            textposition='outside',
            hovertemplate="Rank %{text}<br>%{y}<extra></extra>"
        ))
    cons_fig.update_layout(
        title=f"Conservative — Top 5 Issues" + (f" (week of {week_label})" if week_label else ""),
        xaxis=dict(visible=False),
        yaxis={'autorange': 'reversed'},
        template='plotly_white',
        margin=dict(l=140, r=30, t=50, b=30),
        height=360
    )

    lib_fig = go.Figure()
    if not df_lib.empty:
        lib_fig.add_trace(go.Bar(
            x=[1]*len(df_lib),
            y=df_lib['topic'],
            orientation='h',
            text=[f"#{r}" for r in df_lib['rank']],
            textposition='outside',
            hovertemplate="Rank %{text}<br>%{y}<extra></extra>"
        ))
    lib_fig.update_layout(
        title=f"Liberal — Top 5 Issues" + (f" (week of {week_label})" if week_label else ""),
        xaxis=dict(visible=False),
        yaxis={'autorange': 'reversed'},
        template='plotly_white',
        margin=dict(l=140, r=30, t=50, b=30),
        height=360
    )

    week_caption = f"Showing latest week: {week_label}" if week_label else "No week available."

    return (
        scorecard_display,
        ts_fig,
        traits_section,
        bill_table,
        cons_fig,
        lib_fig,
        week_caption
    )


# ============ RUN ============
if __name__ == '__main__':
    # For Render or other hosts, you might bind to 0.0.0.0 and set a PORT env var.
    app.run_server(host='0.0.0.0', port=int(os.getenv("PORT", "8060")), debug=False)

