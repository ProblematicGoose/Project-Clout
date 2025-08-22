import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json
from datetime import datetime, timedelta

# URLs for live data
BASE_URL = "https://43d1d5a9da0c.ngrok-free.app"
SCORECARD_URL = f"{BASE_URL}/api/scorecard"
TIMESERIES_URL = f"{BASE_URL}/api/timeseries"
TRAITS_URL = f"{BASE_URL}/api/traits"
BILL_SENTIMENT_URL = f"{BASE_URL}/api/bill-sentiment"
TOP_ISSUES_URL = f"{BASE_URL}/api/top-issues"
COMMON_GROUND_URL = f"{BASE_URL}/api/common-ground-issues"
PHOTOS_URL = f"{BASE_URL}/api/subject-photos"
MENTION_COUNT_URL = f"{BASE_URL}/api/mention-counts"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

try:
    with urllib.request.urlopen(SCORECARD_URL, timeout=5) as url:
        subjects_data = json.load(url)
    subjects = sorted({entry["Subject"].strip() for entry in subjects_data if entry["Subject"] is not None})
except Exception as e:
    print("Failed to load initial subjects:", e)
    subjects = []

# Mention count time range options
time_ranges = {
    "Today": (datetime.now(), datetime.now()),
    "This Week": (datetime.now() - timedelta(days=datetime.now().weekday()), datetime.now()),
    "This Month": (datetime(datetime.now().year, datetime.now().month, 1), datetime.now()),
    "This Year": (datetime(datetime.now().year, 1, 1), datetime.now())
}

def fetch_mention_counts(start_date, end_date):
    url = f"{MENTION_COUNT_URL}?start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return pd.DataFrame(json.load(response))
    except Exception as e:
        print("Failed to load mention counts:", e)
        return pd.DataFrame(columns=["Subject", "MentionCount"])

app.layout = html.Div([
    html.Div([
        html.H1("Sentiment Dashboard", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{'label': subj, 'value': subj} for subj in subjects],
            value=next(iter(subjects), None),
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '40px'}),

    html.Div(id='scorecard-div', className='card'),
    html.Div([dcc.Graph(id='timeseries-graph')], className='card'),
    html.Div(id='traits-div', className='card'),
    html.Div(id='bill-sentiment-table', className='card', style={'marginTop': '40px'}),
    html.Div(id='top-issues-div', className='card', style={'marginTop': '40px'}),
    html.Div(id='common-ground-div', className='card', style={'marginTop': '40px'}),

    html.Div([
        html.H2("Mentions by Subject", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='time-range-dropdown',
            options=[{'label': k, 'value': k} for k in time_ranges.keys()] + [{'label': 'Custom Range', 'value': 'Custom'}],
            value='This Week',
            style={'width': '50%', 'margin': '0 auto'}
        ),
        html.Div([
            dcc.DatePickerRange(
                id='custom-date-picker',
                min_date_allowed=datetime(2022, 1, 1),
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now()
            )
        ], id='custom-date-container', style={'textAlign': 'center', 'marginTop': '20px', 'display': 'none'}),
        dcc.Graph(id='mention-count-graph')
    ], className='card', style={'marginTop': '40px'})
])

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
    # Scorecard
    try:
        with urllib.request.urlopen(SCORECARD_URL, timeout=5) as url:
            scorecard_data = json.load(url)
        df_scorecard = pd.DataFrame(scorecard_data)
        score_row = df_scorecard[df_scorecard['Subject'] == selected_subject]
        score = int(score_row['NormalizedSentimentScore'].iloc[0]) if not score_row.empty else 5000
    except:
        score = 5000

    try:
        with urllib.request.urlopen(PHOTOS_URL, timeout=5) as url:
            photo_data = json.load(url)
        df_photos = pd.DataFrame(photo_data)
        photo_row = df_photos[df_photos['Subject'] == selected_subject]
        photo_url = photo_row['PhotoURL'].iloc[0] if not photo_row.empty else None
        office = photo_row['OfficeTitle'].iloc[0] if not photo_row.empty else ''
        party = photo_row['Party'].iloc[0] if not photo_row.empty else ''
        state = photo_row['State'].iloc[0] if not photo_row.empty else ''
    except:
        photo_url, office, party, state = None, '', '', ''

    color = 'green' if score > 6000 else 'crimson' if score < 4000 else 'orange'
    scorecard_display = html.Div([
        html.Div([
            html.Img(src=photo_url, style={'width': '140px', 'height': '170px', 'objectFit': 'cover', 'borderRadius': '12px'})
            if photo_url else html.Div(style={'width': '140px', 'height': '170px', 'background': '#eee'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '24px'}),

        html.Div([
            html.H1(selected_subject, style={'fontSize': '42px', 'fontWeight': 'bold'}),
            html.Div(f"{office}{' • ' if office and (party or state) else ''}{party}{' • ' if party and state else ''}{state}",
                     style={'fontSize': '16px', 'color': '#666', 'marginTop': '4px'}),
            html.Div("Sentiment Score", style={'fontSize': '22px', 'color': 'gray', 'marginTop': '14px'}),
            html.Div(f"{score:,}", style={'fontSize': '56px', 'color': color, 'fontWeight': 'bold'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top'})
    ])

    # Time Series
    try:
        with urllib.request.urlopen(TIMESERIES_URL, timeout=5) as url:
            timeseries_data = json.load(url)
        df_timeseries = pd.DataFrame(timeseries_data)
        df_timeseries['SentimentDate'] = pd.to_datetime(df_timeseries['SentimentDate'])
        df_filtered = df_timeseries[df_timeseries['Subject'] == selected_subject]
    except:
        df_filtered = pd.DataFrame(columns=['SentimentDate', 'NormalizedSentimentScore'])

    timeseries_fig = go.Figure()
    timeseries_fig.add_trace(go.Scatter(
        x=df_filtered['SentimentDate'],
        y=df_filtered['NormalizedSentimentScore'],
        mode='lines+markers',
        name=selected_subject,
        line=dict(color='royalblue')
    ))
    timeseries_fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Normalized Sentiment Score (0–10,000)",
        yaxis=dict(range=[0, 10000]),
        template='plotly_white'
    )

    # Traits
    try:
        with urllib.request.urlopen(TRAITS_URL, timeout=5) as url:
            traits_data = json.load(url)
        df_traits = pd.DataFrame(traits_data)
        df_traits = df_traits[df_traits['Subject'] == selected_subject]
        positive = df_traits[df_traits['TraitType'] == 'Positive'].sort_values('TraitRank')['TraitDescription'].tolist()
        negative = df_traits[df_traits['TraitType'] == 'Negative'].sort_values('TraitRank')['TraitDescription'].tolist()
    except:
        positive, negative = [], []

    trait_display = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(trait) for trait in positive]),
        html.H2("People don't like it when I...", style={'color': 'crimson', 'marginTop': '30px'}),
        html.Ul([html.Li(trait) for trait in negative])
    ])

    # Bill Sentiment
    try:
        with urllib.request.urlopen(BILL_SENTIMENT_URL, timeout=5) as url:
            bill_data = json.load(url)
        df_bills = pd.DataFrame(bill_data)
    except:
        df_bills = pd.DataFrame(columns=['BillName', 'AverageSentimentScore', 'SentimentLabel'])

    bill_table = html.Div([
        html.H2("Public Sentiment Toward National Bills", style={'textAlign': 'center'}),
        html.Table([
            html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
            html.Tbody([
                html.Tr([
                    html.Td(row['BillName']),
                    html.Td(round(row['AverageSentimentScore'], 2)),
                    html.Td(row['SentimentLabel'])
                ]) for _, row in df_bills.iterrows()
            ])
        ], style={'width': '80%', 'margin': '0 auto'})
    ])

    # Top Issues
    try:
        with urllib.request.urlopen(TOP_ISSUES_URL, timeout=5) as url:
            issues_data = json.load(url)
        week = issues_data['WeekStartDate']
        conservative_issues = issues_data['Conservative']
        liberal_issues = issues_data['Liberal']

        issues_display = html.Div([
            html.H2(f"Top Issues for the Week of {week}", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H3("Conservative Topics", style={'color': 'crimson'}),
                    html.Ul([html.Li(f"{item['Rank']}. {item['Topic']}") for item in conservative_issues])
                ], style={'width': '20%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Liberal Topics", style={'color': 'blue'}),
                    html.Ul([html.Li(f"{item['Rank']}. {item['Topic']}") for item in liberal_issues])
                ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '5%'})
            ])
        ])
    except:
        issues_display = html.Div([html.H3("Failed to load top issues")])

    # Common Ground
    try:
        with urllib.request.urlopen(COMMON_GROUND_URL, timeout=5) as url:
            issues_data = json.load(url)
        df_common = pd.DataFrame(issues_data)
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
    except:
        common_issues_display = html.Div([html.H3("Failed to load common ground issues")])

    return scorecard_display, timeseries_fig, trait_display, bill_table, issues_display, common_issues_display

@app.callback(
    Output('custom-date-container', 'style'),
    Input('time-range-dropdown', 'value')
)
def toggle_datepicker(selected):
    return {'textAlign': 'center', 'marginTop': '20px', 'display': 'block'} if selected == 'Custom' else {'display': 'none'}

@app.callback(
    Output('mention-count-graph', 'figure'),
    Input('subject-dropdown', 'value'),
    Input('time-range-dropdown', 'value'),
    Input('custom-date-picker', 'start_date'),
    Input('custom-date-picker', 'end_date')
)
def update_mention_chart(selected_subject, selected_range, start_date, end_date):
    if selected_range != 'Custom':
        start_date, end_date = time_ranges[selected_range]
    else:
        if not start_date or not end_date:
            return go.Figure()
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df_mentions = fetch_mention_counts(start_date, end_date)
    df_mentions = df_mentions[df_mentions['Subject'] == selected_subject]

    fig = go.Figure(go.Bar(
        x=df_mentions['MentionCount'],
        y=df_mentions['Subject'],
        orientation='h',
        marker=dict(color='mediumslateblue')
    ))
    fig.update_layout(
        title=f"Mentions for {selected_subject} ({selected_range})",
        xaxis_title="Number of Mentions",
        yaxis_title="Subject",
        template="plotly_white",
        margin=dict(l=100, r=40, t=60, b=40)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=False)






