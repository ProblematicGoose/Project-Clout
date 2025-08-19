import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json

# -----------------------
# URLs for live data
# -----------------------
API_BASE = "https://efc7f5a4c6c7.ngrok-free.app"
SCORECARD_URL = f"{API_BASE}/scorecard"
TIMESERIES_URL = f"{API_BASE}/timeseries"
TRAITS_URL = f"{API_BASE}/traits"
BILL_SENTIMENT_URL = f"{API_BASE}/bill-sentiment"
TOP_ISSUES_URL = f"{API_BASE}/top-issues"
COMMON_GROUND_URL = f"{API_BASE}/common-ground-issues"
PHOTOS_URL = f"{API_BASE}/photos"  # NEW

app = dash.Dash(__name__)
server = app.server

# -----------------------
# Load initial Subject list
# -----------------------
def safe_json_get(url, timeout=6):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except Exception as e:
        print(f"Fetch failed for {url}: {e}")
        return None

try:
    subjects_data = safe_json_get(SCORECARD_URL)
    subjects = sorted({entry["Subject"].strip() for entry in subjects_data if entry.get("Subject")} ) if subjects_data else []
except Exception as e:
    print("Failed to load initial subjects:", e)
    subjects = []

# -----------------------
# Layout
# -----------------------
app.layout = html.Div([
    html.Div([
        html.H1("Sentiment Dashboard", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{'label': subj, 'value': subj} for subj in subjects],
            value=(subjects[0] if subjects else None),
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '40px'}),

    # Scorecard with headshot
    html.Div(id='scorecard-div', className='card'),

    # Time series
    html.Div([dcc.Graph(id='timeseries-graph')], className='card'),

    # Traits
    html.Div(id='traits-div', className='card'),

    # Bill sentiment
    html.Div(id='bill-sentiment-table', className='card', style={'marginTop': '40px'}),

    # Weekly top issues
    html.Div(id='top-issues-div', className='card', style={'marginTop': '40px'}),

    # Common ground issues for selected subject
    html.Div(id='common-ground-div', className='card', style={'marginTop': '40px'}),

    # -----------------------
    # NEW: Photo Gallery controls + grid
    # -----------------------
    html.Div([
        html.H2("Elected Officials — Photo Gallery", style={'textAlign': 'center', 'marginTop': '60px'}),

        html.Div([
            dcc.Dropdown(
                id='gallery-chamber',
                options=[{'label': 'All Chambers', 'value': ''},
                         {'label': 'Senate', 'value': 'Senator'},
                         {'label': 'House', 'value': 'Representative'}],
                value='',
                clearable=False,
                style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}
            ),
            dcc.Dropdown(
                id='gallery-party',
                options=[{'label': 'All Parties', 'value': ''},
                         {'label': 'Democrat', 'value': 'Democrat'},
                         {'label': 'Republican', 'value': 'Republican'},
                         {'label': 'Independent', 'value': 'Independent'}],
                value='',
                clearable=False,
                style={'width': '30%', 'display': 'inline-block'}
            ),
            html.Button("Refresh Photos", id="refresh-photos", n_clicks=0,
                        style={'marginLeft': '2%', 'padding': '8px 16px'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div(id='photo-grid', className='card',
                 style={'padding': '10px 20px 30px 20px'})
    ])
])

# -----------------------
# Callbacks
# -----------------------

@app.callback(
    Output('scorecard-div', 'children'),
    Output('timeseries-graph', 'figure'),
    Output('traits-div', 'children'),
    Output('bill-sentiment-table', 'children'),
    Output('top-issues-div', 'children'),
    Output('common-ground-div', 'children'),
    Input('subject-dropdown', 'value')
)
def update_dashboard(selected_subject):
    print("Selected subject:", selected_subject)

    # ----- Scorecard (with headshot)
    try:
        scorecard_data = safe_json_get(SCORECARD_URL)
        df_scorecard = pd.DataFrame(scorecard_data) if scorecard_data else pd.DataFrame()
        score_row = df_scorecard[df_scorecard['Subject'] == selected_subject]
        score = int(score_row['NormalizedSentimentScore'].iloc[0]) if not score_row.empty else 5000
        photo_url = score_row['PhotoURL'].iloc[0] if (not score_row.empty and 'PhotoURL' in score_row.columns) else None
        office = score_row['OfficeTitle'].iloc[0] if (not score_row.empty and 'OfficeTitle' in score_row.columns) else ''
        state = score_row['State'].iloc[0] if (not score_row.empty and 'State' in score_row.columns) else ''
        party = score_row['Party'].iloc[0] if (not score_row.empty and 'Party' in score_row.columns) else ''
    except Exception as e:
        print("Failed to load scorecard data:", e)
        score, photo_url, office, state, party = 5000, None, '', '', ''

    color = 'green' if score > 6000 else 'crimson' if score < 4000 else 'orange'
    scorecard_display = html.Div([
        html.Div([
            html.Div([
                html.Img(
                    src=photo_url if photo_url else '',
                    style={
                        'width': '140px', 'height': '170px',
                        'objectFit': 'cover', 'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                    }
                ) if photo_url else html.Div(style={'width': '140px', 'height': '170px', 'background': '#eee', 'borderRadius': '12px'}),
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '24px'}),

            html.Div([
                html.H1(selected_subject or "—", style={'fontSize': '42px', 'fontWeight': 'bold', 'margin': 0}),
                html.Div(f"{office}{' • ' if office and (party or state) else ''}{party}{' • ' if party and state else ''}{state}",
                         style={'fontSize': '16px', 'color': '#666', 'marginTop': '4px'}),
                html.Div("Sentiment Score", style={'fontSize': '22px', 'color': 'gray', 'marginTop': '14px'}),
                html.Div(f"{score:,}", style={'fontSize': '56px', 'color': color, 'fontWeight': 'bold'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ])

    # ----- Time series
    try:
        timeseries_data = safe_json_get(TIMESERIES_URL)
        df_timeseries = pd.DataFrame(timeseries_data) if timeseries_data else pd.DataFrame()
        df_timeseries['SentimentDate'] = pd.to_datetime(df_timeseries['SentimentDate']) if not df_timeseries.empty else pd.to_datetime([])
        df_filtered = df_timeseries[df_timeseries['Subject'] == selected_subject] if not df_timeseries.empty else pd.DataFrame(columns=['SentimentDate','NormalizedSentimentScore'])
    except Exception as e:
        print("Failed to load time series data:", e)
        df_filtered = pd.DataFrame(columns=['SentimentDate', 'NormalizedSentimentScore'])

    timeseries_fig = go.Figure()
    timeseries_fig.add_trace(go.Scatter(
        x=df_filtered['SentimentDate'],
        y=df_filtered['NormalizedSentimentScore'],
        mode='lines+markers',
        name=selected_subject or "",
        line=dict(color='royalblue')
    ))
    timeseries_fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Normalized Sentiment Score (0–10,000)",
        yaxis=dict(range=[0, 10000]),
        template='plotly_white'
    )

    # ----- Traits
    try:
        traits_data = safe_json_get(TRAITS_URL)
        df_traits = pd.DataFrame(traits_data) if traits_data else pd.DataFrame()
        df_traits = df_traits[df_traits['Subject'] == selected_subject] if not df_traits.empty else df_traits
        positive = df_traits[df_traits['TraitType'] == 'Positive'].sort_values('TraitRank')['TraitDescription'].tolist() if not df_traits.empty else []
        negative = df_traits[df_traits['TraitType'] == 'Negative'].sort_values('TraitRank')['TraitDescription'].tolist() if not df_traits.empty else []
    except Exception as e:
        print("Failed to load traits:", e)
        positive, negative = [], []

    trait_display = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(trait, style={'textAlign': 'left'}) for trait in positive], style={'listStyleType': 'none'}),
        html.H2("People don't like it when I...", style={'color': 'crimson', 'marginTop': '30px'}),
        html.Ul([html.Li(trait, style={'textAlign': 'left'}) for trait in negative], style={'listStyleType': 'none'})
    ])

    # ----- Bill sentiment
    try:
        bill_data = safe_json_get(BILL_SENTIMENT_URL)
        df_bills = pd.DataFrame(bill_data) if bill_data else pd.DataFrame(columns=['BillName','AverageSentimentScore','SentimentLabel'])
    except Exception as e:
        print("Failed to load bill sentiment data:", e)
        df_bills = pd.DataFrame(columns=['BillName', 'AverageSentimentScore', 'SentimentLabel'])

    bill_table = html.Div([
        html.H2("Public Sentiment Toward National Bills", style={'textAlign': 'center'}),
        html.Table([
            html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
            html.Tbody([
                html.Tr([
                    html.Td(row['BillName']),
                    html.Td(round(row['AverageSentimentScore'], 2), style={'paddingRight': '30px'}),
                    html.Td(row['SentimentLabel'])
                ]) for _, row in df_bills.iterrows()
            ])
        ], style={'width': '80%', 'margin': '0 auto', 'borderCollapse': 'collapse', 'textAlign': 'left'})
    ])

    # ----- Weekly top issues
    try:
        issues_data = safe_json_get(TOP_ISSUES_URL)
        if issues_data:
            week = issues_data['WeekStartDate']
            conservative_issues = issues_data['Conservative']
            liberal_issues = issues_data['Liberal']
            issues_display = html.Div([
                html.H2(f"Top Issues for the Week of {week}", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.H3("Conservative Topics", style={'color': 'crimson', 'fontSize': '12pt'}),
                        html.Ul([html.Li(f"{item['Rank']}. {item['Topic']}") for item in conservative_issues])
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([
                        html.H3("Liberal Topics", style={'color': 'blue', 'fontSize': '12pt'}),
                        html.Ul([html.Li(f"{item['Rank']}. {item['Topic']}") for item in liberal_issues])
                    ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'})
                ])
            ])
        else:
            issues_display = html.Div([html.H3("Failed to load top issues")])
    except Exception as e:
        print("Failed to load top issues:", e)
        issues_display = html.Div([html.H3("Failed to load top issues")])

    # ----- Common ground issues (selected subject)
    try:
        cg_data = safe_json_get(COMMON_GROUND_URL)
        df_common = pd.DataFrame(cg_data) if cg_data else pd.DataFrame(columns=['Subject','IssueRank','Issue','Explanation'])
        if not df_common.empty and selected_subject:
            df_common['Subject'] = df_common['Subject'].fillna("").str.strip().str.lower()
            selected_subject_clean = selected_subject.strip().lower()
            df_common = df_common[df_common['Subject'] == selected_subject_clean].sort_values('IssueRank')
        if df_common.empty:
            common_issues_display = html.Div([
                html.H3("No common ground issues found for this subject.", style={'textAlign': 'center', 'color': 'gray'})
            ])
        else:
            common_issues_display = html.Div([
                html.H2("Issues to focus on to win over moderates", style={'textAlign': 'center'}),
                html.Ul([
                    html.Li([
                        html.Span(f"{row['IssueRank']}. ", style={'fontWeight': 'bold'}),
                        html.Span(f"{row['Issue']}: ", style={'fontWeight': 'bold'}),
                        html.Span(row['Explanation'])
                    ]) for _, row in df_common.iterrows()
                ])
            ])
    except Exception as e:
        print("Failed to load common ground issues:", e)
        common_issues_display = html.Div([html.H3("Failed to load common ground issues")])

    return scorecard_display, timeseries_fig, trait_display, bill_table, issues_display, common_issues_display


# -----------------------
# NEW: Photo gallery callback
# -----------------------
@app.callback(
    Output('photo-grid', 'children'),
    Input('gallery-chamber', 'value'),
    Input('gallery-party', 'value'),
    Input('refresh-photos', 'n_clicks'),
    prevent_initial_call=False
)
def update_photo_grid(chamber_value, party_value, _n):
    # Build query string
    q = []
    if chamber_value:
        q.append(f"chamber={urllib.parse.quote(chamber_value)}")
    if party_value:
        q.append(f"party={urllib.parse.quote(party_value)}")
    url = PHOTOS_URL + (("?" + "&".join(q)) if q else "")

    data = safe_json_get(url, timeout=10) or []
    df = pd.DataFrame(data)

    if df.empty:
        return html.Div("No photos available.", style={'textAlign': 'center', 'color': '#777', 'padding': '20px'})

    # Create a responsive grid
    # Each card: image + name + meta line
    cards = []
    for _, row in df.iterrows():
        cards.append(
            html.Div([
                html.Img(src=row.get('PhotoURL', ''), style={
                    'width': '100%', 'height': '240px', 'objectFit': 'cover',
                    'borderTopLeftRadius': '12px', 'borderTopRightRadius': '12px'
                }),
                html.Div([
                    html.Div(row.get('Subject', ''), style={'fontWeight': 'bold', 'fontSize': '14px'}),
                    html.Div(
                        f"{row.get('OfficeTitle','')}{' • ' if row.get('OfficeTitle') and (row.get('Party') or row.get('State')) else ''}"
                        f"{row.get('Party','')}{' • ' if row.get('Party') and row.get('State') else ''}"
                        f"{row.get('State','')}",
                        style={'fontSize': '12px', 'color': '#666', 'marginTop': '2px'}
                    )
                ], style={'padding': '8px 10px'})
            ], style={
                'boxShadow': '0 2px 10px rgba(0,0,0,0.08)',
                'borderRadius': '12px',
                'overflow': 'hidden',
                'background': 'white'
            })
        )

    grid = html.Div(cards, style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fill, minmax(180px, 1fr))',
        'gap': '16px'
    })

    return grid


if __name__ == '__main__':
    app.run(debug=False)


