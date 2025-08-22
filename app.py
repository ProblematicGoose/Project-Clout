import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json
import datetime

# URLs for live data
BASE_URL = "https://0a12d9a96721.ngrok-free.app"
SCORECARD_URL = f"{BASE_URL}/scorecard"
TIMESERIES_URL = f"{BASE_URL}/timeseries"
TRAITS_URL = f"{BASE_URL}/traits"
BILL_SENTIMENT_URL = f"{BASE_URL}/bill-sentiment"
TOP_ISSUES_URL = f"{BASE_URL}/top-issues"
COMMON_GROUND_URL = f"{BASE_URL}/common-ground-issues"
PHOTOS_URL = f"{BASE_URL}/subject-photos"
MENTION_COUNTS_URL = f"{BASE_URL}/subject-mention-counts"

app = dash.Dash(__name__)
server = app.server

# Fetch subjects for dropdown
try:
    with urllib.request.urlopen(SCORECARD_URL, timeout=5) as url:
        subjects_data = json.load(url)
    subjects = sorted({entry["Subject"].strip() for entry in subjects_data if entry["Subject"]})
except Exception as e:
    print("Failed to load initial subjects:", e)
    subjects = []

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

    html.Div([
        html.H2("Subject Mention Counts Across Platforms", style={'textAlign': 'center'}),
        html.Div([
            dcc.Dropdown(
                id='mention-filter-dropdown',
                options=[
                    {'label': 'Today', 'value': 'today'},
                    {'label': 'This Week', 'value': 'week'},
                    {'label': 'This Month', 'value': 'month'},
                    {'label': 'This Year', 'value': 'year'},
                    {'label': 'Custom Range', 'value': 'custom'}
                ],
                value='year',
                style={'width': '300px', 'margin': '0 auto'}
            ),
            html.Div([
                dcc.DatePickerRange(
                    id='mention-date-picker',
                    start_date=datetime.date.today() - datetime.timedelta(days=30),
                    end_date=datetime.date.today(),
                    style={'marginTop': '10px'}
                )
            ], id='mention-date-picker-container', style={'textAlign': 'center', 'display': 'none'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='mention-count-graph')
    ], className='card', style={'marginTop': '40px'}),

    html.Div(id='scorecard-div', className='card'),
    html.Div([dcc.Graph(id='timeseries-graph')], className='card'),
    html.Div(id='traits-div', className='card'),
    html.Div(id='bill-sentiment-table', className='card', style={'marginTop': '40px'}),
    html.Div(id='top-issues-div', className='card', style={'marginTop': '40px'}),
    html.Div(id='common-ground-div', className='card', style={'marginTop': '40px'})
])

@app.callback(
    Output('mention-date-picker-container', 'style'),
    Input('mention-filter-dropdown', 'value')
)
def toggle_date_picker(filter_value):
    return {'textAlign': 'center', 'display': 'block'} if filter_value == 'custom' else {'display': 'none'}

@app.callback(
    Output('mention-count-graph', 'figure'),
    Input('mention-filter-dropdown', 'value'),
    State('mention-date-picker', 'start_date'),
    State('mention-date-picker', 'end_date')
)
def update_mention_count_graph(filter_value, start_date, end_date):
    try:
        if filter_value == 'custom' and start_date and end_date:
            url = f"{MENTION_COUNTS_URL}?filter=custom&start_date={start_date}&end_date={end_date}"
        else:
            url = f"{MENTION_COUNTS_URL}?filter={filter_value}"

        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.load(response)

        df = pd.DataFrame(data)
        fig = go.Figure([go.Bar(
            x=df['Subject'],
            y=df['MentionCount'],
            marker_color='indigo'
        )])
        fig.update_layout(
            title="Mentions by Subject",
            xaxis_title="Subject",
            yaxis_title="Mentions",
            template="plotly_white"
        )
        return fig
    except Exception as e:
        print("Failed to load mention count data:", e)
        return go.Figure()

# Existing subject-level callback can go below here (unchanged from original)
# ... (You can paste in the full callback body from your working script)

if __name__ == '__main__':
    app.run(debug=False)





