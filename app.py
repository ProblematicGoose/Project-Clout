import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json
from datetime import datetime, timedelta

BASE_URL = "https://e8eb17633693.ngrok-free.app"
SCORECARD_URL = f"{BASE_URL}/api/scorecard"
PHOTOS_URL = f"{BASE_URL}/api/subject-photos"
TIMESERIES_URL = f"{BASE_URL}/api/timeseries"
TRAITS_URL = f"{BASE_URL}/api/traits"
BILL_SENTIMENT_URL = f"{BASE_URL}/api/bill-sentiment"
TOP_ISSUES_URL = f"{BASE_URL}/api/top-issues"
COMMON_GROUND_URL = f"{BASE_URL}/api/common-ground-issues"
MENTION_COUNT_URL = f"{BASE_URL}/api/mention-counts"
MOMENTUM_URL = f"{BASE_URL}/api/momentum"

app = dash.Dash(__name__)
server = app.server

def fetch_df(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return pd.DataFrame(json.load(r))
    except Exception as e:
        print(f"Fetch failed for {url}: {e}")
        return pd.DataFrame()

def fetch_json(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.load(r)
    except Exception as e:
        print(f"Fetch failed for {url}: {e}")
        return {}

TIME_OPTIONS = {
    "Today": (datetime.now(), datetime.now()),
    "This Week": (datetime.now() - timedelta(days=datetime.now().weekday()), datetime.now()),
    "This Month": (datetime(datetime.now().year, datetime.now().month, 1), datetime.now()),
    "This Year": (datetime(datetime.now().year, 1, 1), datetime.now())
}

try:
    scorecard_df = fetch_df(SCORECARD_URL)
    subjects = sorted(scorecard_df["Subject"].dropna().unique())
except:
    subjects = []

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

   # Scorecard
html.Div(id='scorecard-div'),

# Sentiment Over Time Card
html.Div([
    html.H2("Sentiment Over Time", style={'textAlign': 'center'}),
    dcc.Graph(id='timeseries-graph')
], style={
    'marginTop': '40px',
    'padding': '20px',
    'border': '1px solid #ccc',
    'borderRadius': '10px',
    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)',
    'background': '#f9f9f9'
}),

# Traits Card
html.Div(id='traits-div', style={
    'marginTop': '40px',
    'padding': '20px',
    'border': '1px solid #ccc',
    'borderRadius': '10px',
    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)',
    'background': '#f9f9f9'
}),

# Bill Sentiment Card
html.Div(id='bill-sentiment-table', style={
    'marginTop': '40px',
    'padding': '20px',
    'border': '1px solid #ccc',
    'borderRadius': '10px',
    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)',
    'background': '#f9f9f9'
}),

# Top Issues Card
html.Div(id='top-issues-div', style={
    'marginTop': '40px',
    'padding': '20px',
    'border': '1px solid #ccc',
    'borderRadius': '10px',
    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)',
    'background': '#f9f9f9'
}),

# Common Ground Card
html.Div(id='common-ground-div', style={
    'marginTop': '40px',
    'padding': '20px',
    'border': '1px solid #ccc',
    'borderRadius': '10px',
    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)',
    'background': '#f9f9f9'
}),


    html.Div([
        html.H2("Mentions by Subject", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='mention-time-range-dropdown',
            options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
            value='This Week',
            style={'width': '50%', 'margin': '0 auto'}
        ),
        html.Div([
            dcc.DatePickerRange(
                id='mention-custom-date-picker',
                min_date_allowed=datetime(2022, 1, 1),
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now()
            )
        ], id='mention-custom-date-container', style={'textAlign': 'center', 'marginTop': '20px', 'display': 'none'}),
        dcc.Graph(id='mention-count-graph')
    ], style={'marginTop': '40px'}),

    html.Div([
        html.H2("Momentum by Subject", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='momentum-time-range-dropdown',
            options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
            value='This Week',
            style={'width': '50%', 'margin': '0 auto'}
        ),
        html.Div([
            dcc.DatePickerRange(
                id='momentum-custom-date-picker',
                min_date_allowed=datetime(2022, 1, 1),
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now()
            )
        ], id='momentum-custom-date-container', style={'textAlign': 'center', 'marginTop': '20px', 'display': 'none'}),
        dcc.Graph(id='momentum-graph')
    ], style={'marginTop': '40px'})
])

@app.callback(
    Output('mention-custom-date-container', 'style'),
    Input('mention-time-range-dropdown', 'value')
)
def toggle_custom_datepicker_mention(value):
    return {'textAlign': 'center', 'marginTop': '20px', 'display': 'block'} if value == 'Custom' else {'display': 'none'}

@app.callback(
    Output('momentum-custom-date-container', 'style'),
    Input('momentum-time-range-dropdown', 'value')
)
def toggle_custom_datepicker_momentum(value):
    return {'textAlign': 'center', 'marginTop': '20px', 'display': 'block'} if value == 'Custom' else {'display': 'none'}

@app.callback(
    Output('mention-count-graph', 'figure'),
    Input('subject-dropdown', 'value'),
    Input('mention-time-range-dropdown', 'value'),
    Input('mention-custom-date-picker', 'start_date'),
    Input('mention-custom-date-picker', 'end_date')
)
def update_mentions_graph(subject, range_choice, custom_start, custom_end):
    if range_choice == 'Custom' and custom_start and custom_end:
        start, end = custom_start[:10], custom_end[:10]
    elif range_choice in TIME_OPTIONS:
        start, end = TIME_OPTIONS[range_choice]
        start, end = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    else:
        return go.Figure()

    query = f"{MENTION_COUNT_URL}?start_date={start}&end_date={end}"
    df = fetch_df(query)
    df = df.rename(columns={"ActivityDate": "MomentumDate", "MomentumScore": "Momentum"})

    df = df[df['Subject'].str.strip().str.lower() == subject.strip().lower()] if not df.empty else pd.DataFrame(columns=['Subject', 'MentionCount'])
    count = df['MentionCount'].values[0] if not df.empty else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[subject], y=[count], name=subject))
    fig.update_layout(title="Mentions Across Platforms", yaxis_title="Mentions", xaxis_title="Subject")
    return fig

@app.callback(
    Output('momentum-graph', 'figure'),
    Input('subject-dropdown', 'value'),
    Input('momentum-time-range-dropdown', 'value'),
    Input('momentum-custom-date-picker', 'start_date'),
    Input('momentum-custom-date-picker', 'end_date')
)
def update_momentum_graph(subject, range_choice, custom_start, custom_end):
    # Resolve date range
    if range_choice == 'Custom' and custom_start and custom_end:
        start, end = custom_start[:10], custom_end[:10]
    elif range_choice in TIME_OPTIONS:
        start_dt, end_dt = TIME_OPTIONS[range_choice]
        start, end = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
    else:
        return go.Figure()

    # Fetch
    query = f"{MOMENTUM_URL}?start_date={start}&end_date={end}"
    df = fetch_df(query)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No momentum data available for this timeframe.")
        return fig

    # Normalize Subject filtering
    if 'Subject' in df.columns and subject:
        df['Subject'] = df['Subject'].astype(str)
        df = df[df['Subject'].str.strip().str.lower() == subject.strip().lower()]

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No momentum data available for this timeframe.")
        return fig

    # Normalize columns
    date_candidates = [c for c in ['MomentumDate', 'ActivityDate', 'Date', 'CreatedDate', 'EventDate'] if c in df.columns]
    value_candidates = [c for c in ['Momentum', 'MomentumScore', 'Score', 'Value'] if c in df.columns]

    if date_candidates:
        df = df.rename(columns={date_candidates[0]: 'MomentumDate'})
    if value_candidates:
        df = df.rename(columns={value_candidates[0]: 'Momentum'})

    if 'MomentumDate' not in df.columns or 'Momentum' not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No momentum data available for this timeframe.")
        return fig

    # Clean
    df = df[['MomentumDate', 'Momentum']].copy()
    df['MomentumDate'] = pd.to_datetime(df['MomentumDate'], errors='coerce')
    df['Momentum'] = pd.to_numeric(df['Momentum'], errors='coerce')
    df = df.dropna(subset=['MomentumDate', 'Momentum'])

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No momentum data available for this timeframe.")
        return fig

    # Aggregate daily momentum values
    df = df.groupby(df['MomentumDate'].dt.date).agg({'Momentum': 'mean'}).reset_index()
    df['MomentumDate'] = pd.to_datetime(df['MomentumDate'])
    df = df.sort_values('MomentumDate')

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['MomentumDate'],
        y=df['Momentum'],
        mode='lines+markers',
        name=subject
    ))
    fig.update_layout(
        title="Momentum = Mentions × Avg Sentiment",
        xaxis_title="Date",
        yaxis_title="Momentum",
        template='plotly_white'
    )
    return fig

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
    # Scorecard
    scorecard = fetch_df(SCORECARD_URL)
    row = scorecard[scorecard.Subject == subject]
    score = int(row.NormalizedSentimentScore.iloc[0]) if not row.empty else 5000

    photo = fetch_df(PHOTOS_URL)
    meta = photo[photo.Subject == subject]
    photo_url, office, party, state = None, '', '', ''
    if not meta.empty:
        photo_url = meta.PhotoURL.iloc[0]
        office, party, state = meta.OfficeTitle.iloc[0], meta.Party.iloc[0], meta.State.iloc[0]

    color = 'green' if score > 6000 else 'crimson' if score < 4000 else 'orange'
    scorecard_div = html.Div([
    html.Div([
        html.Img(src=photo_url, style={'width': '140px', 'height': '170px'}) if photo_url else html.Div(style={'width': '140px', 'height': '170px', 'background': '#eee'}),
    ]),
    html.Div([
        html.H1(subject, style={'marginBottom': '4px'}),
        html.Div(f"{office} • {party} • {state}", style={'color': '#666', 'marginBottom': '12px'}),
        html.Div("Sentiment Score", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
        html.Div(f"{score:,}", style={'fontSize': '56px', 'color': color, 'fontWeight': 'bold'})
    ])
], style={
    'display': 'flex',
    'alignItems': 'center',
    'gap': '24px',
    'padding': '20px',
    'border': '1px solid #ccc',
    'borderRadius': '10px',
    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)',
    'background': '#f9f9f9',
    'width': 'fit-content',
    'margin': '0 auto'
})


    # Sentiment over time
    df = fetch_df(TIMESERIES_URL)
    df['SentimentDate'] = pd.to_datetime(df['SentimentDate'])
    filtered = df[df['Subject'] == subject]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered['SentimentDate'],
        y=filtered['NormalizedSentimentScore'],
        mode='lines+markers',
        name=subject,
        line=dict(color='royalblue')
    ))
    fig.update_layout(title="Sentiment Over Time", xaxis_title="Date", yaxis_title="Normalized Sentiment Score", yaxis=dict(range=[0, 10000]), template='plotly_white')

    # Traits
    traits = fetch_df(TRAITS_URL)
    traits = traits[traits.Subject == subject]
    pos = traits[traits.TraitType == 'Positive'].sort_values('TraitRank').TraitDescription.tolist()
    neg = traits[traits.TraitType == 'Negative'].sort_values('TraitRank').TraitDescription.tolist()
    traits_div = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(p) for p in pos]),
        html.H2("People don't like it when I...", style={'color': 'crimson', 'marginTop': '30px'}),
        html.Ul([html.Li(n) for n in neg])
    ])

    # Bill Sentiment
    bills = fetch_df(BILL_SENTIMENT_URL)
    if bills.empty:
        bill_table = html.Div([html.H3("No bill sentiment data available.")])
    else:
        bill_table = html.Div([
            html.H2("Public Sentiment Toward National Bills", style={'textAlign': 'center'}),
            html.Table([
                html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
                html.Tbody([
                    html.Tr([
                        html.Td(row['BillName']),
                        html.Td(round(row['AverageSentimentScore'], 2)),
                        html.Td(row['SentimentLabel'])
                    ]) for _, row in bills.iterrows()
                ])
            ], style={'width': '80%', 'margin': '0 auto'})
        ])

    # Top Issues
    issues = fetch_json(TOP_ISSUES_URL)
    if not issues or not all(k in issues for k in ('Liberal', 'Conservative', 'WeekStartDate')):
        issues_div = html.Div([html.H3("Top issues data unavailable.")])
    else:
        week = issues['WeekStartDate']
        liberal = issues['Liberal']
        conservative = issues['Conservative']
        issues_div = html.Div([
            html.H2(f"Top Issues for the Week of {week}", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H3("Conservative Topics", style={'color': 'crimson', 'fontSize': '12pt'}),
                    html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for t in conservative])
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Liberal Topics", style={'color': 'blue', 'fontSize': '12pt'}),
                    html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for t in liberal])
                ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'})
            ])
        ])

    # Common Ground
    common_df = fetch_df(COMMON_GROUND_URL)
    if common_df.empty or 'Subject' not in common_df.columns:
        common_ground_div = html.Div([
            html.H3("No common ground issues found for this subject.", style={'textAlign': 'center', 'color': 'gray'})
        ])
    else:
        common_df['Subject'] = common_df['Subject'].fillna('').str.lower()
        filtered = common_df[common_df['Subject'] == subject.lower()].sort_values('IssueRank')
        if filtered.empty:
            common_ground_div = html.Div([
                html.H3("No common ground issues found for this subject.", style={'textAlign': 'center', 'color': 'gray'})
            ])
        else:
            common_ground_div = html.Div([
                html.H2("Issues to focus on to win over moderates", style={'textAlign': 'center'}),
                html.Ul([
                    html.Li([
                        html.Span(f"{r['IssueRank']}. ", style={'fontWeight': 'bold'}),
                        html.Span(f"{r['Issue']}: ", style={'fontWeight': 'bold'}),
                        html.Span(r['Explanation'])
                    ]) for _, r in filtered.iterrows()
                ])
            ])

    return scorecard_div, fig, traits_div, bill_table, issues_div, common_ground_div


if __name__ == '__main__':
    app.run(debug=True)














