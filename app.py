import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json
import datetime

# URLs for live data
SCORECARD_URL = "https://810d8473570f.ngrok-free.app/api/scorecard"
TIMESERIES_URL = "https://810d8473570f.ngrok-free.app/api/timeseries"
TRAITS_URL = "https://810d8473570f.ngrok-free.app/api/traits"
BILL_SENTIMENT_URL = "https://810d8473570f.ngrok-free.app/api/bill-sentiment"
TOP_ISSUES_URL = "https://810d8473570f.ngrok-free.app/api/top-issues"
COMMON_GROUND_URL = "https://810d8473570f.ngrok-free.app/api/common-ground-issues"
PHOTOS_URL = "https://810d8473570f.ngrok-free.app/api/subject-photos"
MENTION_COUNTS_URL = "https://810d8473570f.ngrok-free.app/api/subject-mention-counts"

app = dash.Dash(__name__)
server = app.server

try:
    with urllib.request.urlopen(SCORECARD_URL, timeout=5) as url:
        subjects_data = json.load(url)
    subjects = sorted({entry["Subject"].strip() for entry in subjects_data if entry["Subject"] is not None})
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

    html.Div(id='scorecard-div', className='card'),
    html.Div([dcc.Graph(id='timeseries-graph')], className='card'),
    html.Div(id='traits-div', className='card'),
    html.Div(id='bill-sentiment-table', className='card', style={'marginTop': '40px'}),
    html.Div(id='top-issues-div', className='card', style={'marginTop': '40px'}),
    html.Div(id='common-ground-div', className='card', style={'marginTop': '40px'}),

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
                value='week',
                style={'width': '300px', 'marginBottom': '10px'}
            ),
            html.Div([
                dcc.DatePickerRange(
                    id='mention-date-picker',
                    start_date=datetime.date.today() - datetime.timedelta(days=7),
                    end_date=datetime.date.today(),
                    style={'display': 'inline-block'}
                )
            ], id='mention-date-picker-container', style={'display': 'none'})
        ], style={'textAlign': 'center'}),
        dcc.Graph(id='mention-count-graph')
    ], className='card', style={'marginTop': '40px'})
])

@app.callback(
    Output('mention-date-picker-container', 'style'),
    Input('mention-filter-dropdown', 'value')
)
def toggle_date_picker(filter_value):
    return {'display': 'block'} if filter_value == 'custom' else {'display': 'none'}

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

# Existing callback remains unchanged...
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
    # ... [unchanged existing logic] ...
    return scorecard_display, timeseries_fig, trait_display, bill_table, issues_display, common_issues_display

if __name__ == '__main__':
    app.run(debug=False)





