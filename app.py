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

app = dash.Dash(__name__)
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
    Output('custom-date-container', 'style'),
    Input('time-range-dropdown', 'value')
)
def toggle_datepicker(selected):
    return {'textAlign': 'center', 'marginTop': '20px', 'display': 'block'} if selected == 'Custom' else {'display': 'none'}

@app.callback(
    Output('mention-count-graph', 'figure'),
    Input('time-range-dropdown', 'value'),
    Input('custom-date-picker', 'start_date'),
    Input('custom-date-picker', 'end_date')
)
def update_mention_chart(selected_range, start_date, end_date):
    if selected_range != 'Custom':
        start_date, end_date = time_ranges[selected_range]
    else:
        if not start_date or not end_date:
            return go.Figure()
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df_mentions = fetch_mention_counts(start_date, end_date)
    df_mentions.sort_values("MentionCount", ascending=True, inplace=True)

    fig = go.Figure(go.Bar(
        x=df_mentions['MentionCount'],
        y=df_mentions['Subject'],
        orientation='h',
        marker=dict(color='mediumslateblue')
    ))
    fig.update_layout(
        title=f"Mentions by Subject ({selected_range})",
        xaxis_title="Number of Mentions",
        yaxis_title="Subject",
        template="plotly_white",
        margin=dict(l=100, r=40, t=60, b=40)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=False)





