import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json
from datetime import datetime, timedelta

# URLs for live data
BASE_URL = "https://e8eb17633693.ngrok-free.app"
SCORECARD_URL = f"{BASE_URL}/api/scorecard"
TIMESERIES_URL = f"{BASE_URL}/api/timeseries"
TRAITS_URL = f"{BASE_URL}/api/traits"
BILL_SENTIMENT_URL = f"{BASE_URL}/api/bill-sentiment"
TOP_ISSUES_URL = f"{BASE_URL}/api/top-issues"
COMMON_GROUND_URL = f"{BASE_URL}/api/common-ground-issues"
PHOTOS_URL = f"{BASE_URL}/api/subject-photos"
MENTION_COUNT_URL = f"{BASE_URL}/api/mention-counts"
MOMENTUM_URL = f"{BASE_URL}/api/momentum"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

try:
    with urllib.request.urlopen(SCORECARD_URL, timeout=5) as url:
        subjects_data = json.load(url)
    subjects = sorted({entry["Subject"].strip() for entry in subjects_data if entry["Subject"] is not None})
except Exception as e:
    print("Failed to load initial subjects:", e)
    subjects = []

# Time range options
time_ranges = {
    "Today": (datetime.now(), datetime.now()),
    "This Week": (datetime.now() - timedelta(days=datetime.now().weekday()), datetime.now()),
    "This Month": (datetime(datetime.now().year, datetime.now().month, 1), datetime.now()),
    "This Year": (datetime(datetime.now().year, 1, 1), datetime.now())
}

def fetch_dataframe(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return pd.DataFrame(json.load(response))
    except Exception as e:
        print("Failed to fetch:", url, e)
        return pd.DataFrame()

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
            id='mention-time-range-dropdown',
            options=[{'label': k, 'value': k} for k in time_ranges.keys()] + [{'label': 'Custom Range', 'value': 'Custom'}],
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
    ], className='card', style={'marginTop': '40px'}),

    html.Div([
        html.H2("Momentum Over Time", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='momentum-time-range-dropdown',
            options=[{'label': k, 'value': k} for k in time_ranges.keys()] + [{'label': 'Custom Range', 'value': 'Custom'}],
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
    ], className='card', style={'marginTop': '40px'})
])

@app.callback(
    Output('mention-custom-date-container', 'style'),
    Input('mention-time-range-dropdown', 'value')
)
def toggle_mention_datepicker(selected):
    return {'textAlign': 'center', 'marginTop': '20px', 'display': 'block'} if selected == 'Custom' else {'display': 'none'}

@app.callback(
    Output('momentum-custom-date-container', 'style'),
    Input('momentum-time-range-dropdown', 'value')
)
def toggle_momentum_datepicker(selected):
    return {'textAlign': 'center', 'marginTop': '20px', 'display': 'block'} if selected == 'Custom' else {'display': 'none'}

@app.callback(
    Output('mention-count-graph', 'figure'),
    Input('subject-dropdown', 'value'),
    Input('mention-time-range-dropdown', 'value'),
    Input('mention-custom-date-picker', 'start_date'),
    Input('mention-custom-date-picker', 'end_date')
)
def update_mentions(selected_subject, selected_range, start_date, end_date):
    if selected_range != 'Custom':
        start, end = time_ranges[selected_range]
    else:
        if not start_date or not end_date:
            return go.Figure()
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

    url = f"{MENTION_COUNT_URL}?start_date={start.strftime('%Y-%m-%d')}&end_date={end.strftime('%Y-%m-%d')}"
    df = fetch_dataframe(url)
    df = df[df['Subject'] == selected_subject]

    fig = go.Figure(go.Bar(
        x=df['MentionCount'],
        y=df['Subject'],
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

@app.callback(
    Output('momentum-graph', 'figure'),
    Input('subject-dropdown', 'value'),
    Input('momentum-time-range-dropdown', 'value'),
    Input('momentum-custom-date-picker', 'start_date'),
    Input('momentum-custom-date-picker', 'end_date')
)
def update_momentum(selected_subject, selected_range, start_date, end_date):
    if selected_range != 'Custom':
        start, end = time_ranges[selected_range]
    else:
        if not start_date or not end_date:
            print("❌ Missing custom date values")
            return go.Figure()
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            print("❌ Invalid custom date format")
            return go.Figure()

    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    url = f"{MOMENTUM_URL}?start_date={start_str}&end_date={end_str}"
    print("Momentum URL:", url)

    df = fetch_dataframe(url)
    df = df[df['Subject'] == selected_subject]

    fig = go.Figure()
    if not df.empty:
        df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
        df = df.sort_values('ActivityDate')
        fig.add_trace(go.Scatter(
            x=df['ActivityDate'],
            y=df['MomentumScore'],
            mode='lines+markers',
            name=selected_subject,
            line=dict(color='orange')
        ))
    fig.update_layout(
        title=f"Momentum Over Time for {selected_subject}",
        xaxis_title="Date",
        yaxis_title="Momentum Score",
        template="plotly_white"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=False)









