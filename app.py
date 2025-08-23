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
    html.H1("Sentiment Dashboard", style={'textAlign': 'center', 'paddingTop': '20px'}),
    html.Div([
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{"label": s, "value": s} for s in subjects],
            value=subjects[0] if subjects else None,
            style={'width': '100%'}
        )
    ], className='dropdown-wrapper'),

    html.Div([
        html.Div(id='scorecard-div', className='dashboard-card'),
        html.Div([
            html.H2("Sentiment Over Time", className='center-text'),
            dcc.Dropdown(
                id='timeseries-time-range-dropdown',
                options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
                value='This Month',
                className='dcc-control'
            ),
            html.Div([
                dcc.DatePickerRange(
                    id='timeseries-custom-date-picker',
                    min_date_allowed=datetime(2022, 1, 1),
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
            ], id='timeseries-custom-date-container', style={'textAlign': 'center', 'marginTop': '10px', 'display': 'none'}),
            dcc.Graph(id='timeseries-graph', style={'height': '400px', 'overflow': 'hidden'})
        ], className='dashboard-card'),
        html.Div(id='dashboard-cards', className='dashboard-grid'),
        html.Div([
            html.H2("Mentions by Subject", className='center-text'),
            dcc.Dropdown(
                id='mention-time-range-dropdown',
                options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
                value='This Week',
                className='dcc-control'
            ),
            html.Div([
                dcc.DatePickerRange(
                    id='mention-custom-date-picker',
                    min_date_allowed=datetime(2022, 1, 1),
                    start_date=datetime.now() - timedelta(days=7),
                    end_date=datetime.now()
                )
            ], id='mention-custom-date-container', style={'textAlign': 'center', 'marginTop': '10px', 'display': 'none'}),
            dcc.Graph(id='mention-count-graph', style={'height': '400px', 'overflow': 'hidden'})
        ], className='dashboard-card'),
        html.Div([
            html.H2("Momentum by Subject", className='center-text'),
            dcc.Dropdown(
                id='momentum-time-range-dropdown',
                options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
                value='This Week',
                className='dcc-control'
            ),
            html.Div([
                dcc.DatePickerRange(
                    id='momentum-custom-date-picker',
                    min_date_allowed=datetime(2022, 1, 1),
                    start_date=datetime.now() - timedelta(days=7),
                    end_date=datetime.now()
                )
            ], id='momentum-custom-date-container', style={'textAlign': 'center', 'marginTop': '10px', 'display': 'none'}),
            dcc.Graph(id='momentum-graph', style={'height': '400px', 'overflow': 'hidden'})
        ], className='dashboard-card')
    ], className='dashboard-grid')
])

    html.H1("Sentiment Dashboard", style={'textAlign': 'center', 'paddingTop': '20px'}),
    html.Div([
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{"label": s, "value": s} for s in subjects],
            value=subjects[0] if subjects else None,
            style={'width': '100%'}
        )
    ], className='dropdown-wrapper'),

    html.Div([
        html.Div([
            html.Div([
                html.H2("Sentiment Over Time", className='center-text'),
                dcc.Dropdown(
                    id='timeseries-time-range-dropdown',
                    options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
                    value='This Month',
                    className='dcc-control'
                ),
                html.Div([
                    dcc.DatePickerRange(
                        id='timeseries-custom-date-picker',
                        min_date_allowed=datetime(2022, 1, 1),
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now()
                    )
                ], id='timeseries-custom-date-container', style={'textAlign': 'center', 'marginTop': '10px', 'display': 'none'}),
                dcc.Graph(id='timeseries-graph', style={'height': '400px', 'overflow': 'hidden'})
            ], className='dashboard-card'),

            html.Div(id='dashboard-cards', className='dashboard-grid'),

            html.Div([
                html.H2("Mentions by Subject", className='center-text'),
                dcc.Dropdown(
                    id='mention-time-range-dropdown',
                    options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
                    value='This Week',
                    className='dcc-control'
                ),
                html.Div([
                    dcc.DatePickerRange(
                        id='mention-custom-date-picker',
                        min_date_allowed=datetime(2022, 1, 1),
                        start_date=datetime.now() - timedelta(days=7),
                        end_date=datetime.now()
                    )
                ], id='mention-custom-date-container', style={'textAlign': 'center', 'marginTop': '10px', 'display': 'none'}),
                dcc.Graph(id='mention-count-graph', style={'height': '400px', 'overflow': 'hidden'})
            ], className='dashboard-card'),

            html.Div([
                html.H2("Momentum by Subject", className='center-text'),
                dcc.Dropdown(
                    id='momentum-time-range-dropdown',
                    options=[{'label': k, 'value': k} for k in TIME_OPTIONS] + [{'label': 'Custom Range', 'value': 'Custom'}],
                    value='This Week',
                    className='dcc-control'
                ),
                html.Div([
                    dcc.DatePickerRange(
                        id='momentum-custom-date-picker',
                        min_date_allowed=datetime(2022, 1, 1),
                        start_date=datetime.now() - timedelta(days=7),
                        end_date=datetime.now()
                    )
                ], id='momentum-custom-date-container', style={'textAlign': 'center', 'marginTop': '10px', 'display': 'none'}),
                dcc.Graph(id='momentum-graph', style={'height': '400px', 'overflow': 'hidden'})
            ], className='dashboard-card')
        ], className='dashboard-grid')
    ])
])

# (Callbacks will be filled in next)

@app.callback(
    Output('timeseries-custom-date-container', 'style'),
    Input('timeseries-time-range-dropdown', 'value')
)
def toggle_custom_datepicker_timeseries(value):
    return {'textAlign': 'center', 'marginTop': '10px', 'display': 'block'} if value == 'Custom' else {'display': 'none'}

@app.callback(
    Output('mention-custom-date-container', 'style'),
    Input('mention-time-range-dropdown', 'value')
)
def toggle_custom_datepicker_mention(value):
    return {'textAlign': 'center', 'marginTop': '10px', 'display': 'block'} if value == 'Custom' else {'display': 'none'}

@app.callback(
    Output('momentum-custom-date-container', 'style'),
    Input('momentum-time-range-dropdown', 'value')
)
def toggle_custom_datepicker_momentum(value):
    return {'textAlign': 'center', 'marginTop': '10px', 'display': 'block'} if value == 'Custom' else {'display': 'none'}

@app.callback(
    Output('timeseries-graph', 'figure'),
    Input('subject-dropdown', 'value'),
    Input('timeseries-time-range-dropdown', 'value'),
    Input('timeseries-custom-date-picker', 'start_date'),
    Input('timeseries-custom-date-picker', 'end_date')
)
def update_timeseries_graph(subject, range_choice, custom_start, custom_end):
    if range_choice == 'Custom' and custom_start and custom_end:
        start, end = custom_start[:10], custom_end[:10]
    elif range_choice in TIME_OPTIONS:
        start_dt, end_dt = TIME_OPTIONS[range_choice]
        start, end = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
    else:
        return go.Figure()

    df = fetch_df(TIMESERIES_URL)
    if df.empty: return go.Figure()

    df['SentimentDate'] = pd.to_datetime(df['SentimentDate'], errors='coerce')
    df = df[df['Subject'].str.lower() == subject.lower()]
    df = df[(df['SentimentDate'] >= start) & (df['SentimentDate'] <= end)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['SentimentDate'],
        y=df['NormalizedSentimentScore'],
        mode='lines+markers',
        name=subject,
        line=dict(color='royalblue')
    ))
    fig.update_layout(title="Sentiment Over Time", xaxis_title="Date", yaxis_title="Normalized Sentiment Score", yaxis=dict(range=[0, 10000]), template='plotly_white')
    return fig

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
        start_dt, end_dt = TIME_OPTIONS[range_choice]
        start, end = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
    else:
        return go.Figure()

    query = f"{MENTION_COUNT_URL}?start_date={start}&end_date={end}"
    df = fetch_df(query)
    df = df[df['Subject'].str.lower() == subject.lower()] if not df.empty else pd.DataFrame(columns=['Subject', 'MentionCount'])
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
    if range_choice == 'Custom' and custom_start and custom_end:
        start, end = custom_start[:10], custom_end[:10]
    elif range_choice in TIME_OPTIONS:
        start_dt, end_dt = TIME_OPTIONS[range_choice]
        start, end = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
    else:
        return go.Figure()

    query = f"{MOMENTUM_URL}?start_date={start}&end_date={end}"
    df = fetch_df(query)
    if 'Subject' in df.columns and subject:
        df['Subject'] = df['Subject'].astype(str)
        df = df[df['Subject'].str.lower() == subject.lower()]

    df = df.rename(columns={c: 'MomentumDate' for c in ['MomentumDate', 'ActivityDate', 'Date'] if c in df.columns})
    df = df.rename(columns={c: 'Momentum' for c in ['Momentum', 'MomentumScore'] if c in df.columns})
    if 'MomentumDate' not in df.columns or 'Momentum' not in df.columns: return go.Figure()

    df['MomentumDate'] = pd.to_datetime(df['MomentumDate'], errors='coerce')
    df['Momentum'] = pd.to_numeric(df['Momentum'], errors='coerce')
    df = df.dropna()
    df = df.groupby(df['MomentumDate'].dt.date).agg({'Momentum': 'mean'}).reset_index()
    df['MomentumDate'] = pd.to_datetime(df['MomentumDate'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['MomentumDate'], y=df['Momentum'], mode='lines+markers', name=subject))
    fig.update_layout(title="Momentum = Mentions × Avg Sentiment", xaxis_title="Date", yaxis_title="Momentum", template='plotly_white')
    return fig

@app.callback(
    Output('dashboard-cards', 'children'),
    Input('subject-dropdown', 'value')
)
def update_dashboard(subject):
    scorecard = fetch_df(SCORECARD_URL)
    if 'Subject' not in scorecard.columns or scorecard.empty:
        return [html.Div(html.H3("No scorecard data available."), className='dashboard-card')]
    row = scorecard[scorecard['Subject'] == subject]
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
            html.Img(src=photo_url, className='scorecard-img') if photo_url else html.Div(style={'width': '100px', 'height': '120px', 'background': '#eee'})
        ]),
        html.Div([
            html.H1(subject),
            html.Div(f"{office} • {party} • {state}"),
            html.Div("Sentiment Score", className='section-header'),
            html.Div(f"{score:,}", style={'fontSize': '40px', 'color': color, 'fontWeight': 'bold'})
        ], className='scorecard-metadata')
    ], className='scorecard-container')

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
            ], style={'width': '100%'})
        ])

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
                    html.H3("Conservative Topics", style={'color': 'crimson'}),
                    html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for t in conservative])
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Liberal Topics", style={'color': 'blue'}),
                    html.Ul([html.Li(f"{t['Rank']}. {t['Topic']}") for t in liberal])
                ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'})
            ])
        ])

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

    return [
        html.Div(scorecard_div, className='dashboard-card'),
        html.Div(traits_div, className='dashboard-card'),
        html.Div(bill_table, className='dashboard-card'),
        html.Div(issues_div, className='dashboard-card'),
        html.Div(common_ground_div, className='dashboard-card')
    ]

if __name__ == '__main__':
    app.run(debug=True)















