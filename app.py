import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json

# URLs for live data
SCORECARD_URL = "https://b47121c63ba4.ngrok-free.app/api/scorecard"
TIMESERIES_URL = "https://b47121c63ba4.ngrok-free.app/api/timeseries"
TRAITS_URL = "https://b47121c63ba4.ngrok-free.app/api/traits"
BILL_SENTIMENT_URL = "https://b47121c63ba4.ngrok-free.app/api/bill-sentiment"
TOP_ISSUES_URL = "https://b47121c63ba4.ngrok-free.app/api/top-issues"
COMMON_GROUND_URL = "https://b47121c63ba4.ngrok-free.app/api/common-ground-issues"
PHOTOS_URL = "https://b47121c63ba4.ngrok-free.app/api/subject-photos"

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

    html.Div(id='photo-grid', className='card', style={'marginBottom': '60px'}),

    html.Div(id='scorecard-div', className='card'),
    html.Div([dcc.Graph(id='timeseries-graph')], className='card'),
    html.Div(id='traits-div', className='card'),
    html.Div(id='bill-sentiment-table', className='card', style={'marginTop': '40px'}),
    html.Div(id='top-issues-div', className='card', style={'marginTop': '40px'}),
    html.Div(id='common-ground-div', className='card', style={'marginTop': '40px'})
])

@app.callback(
    Output('photo-grid', 'children'),
    Input('subject-dropdown', 'value')
)
def display_photos(_):
    try:
        with urllib.request.urlopen(PHOTOS_URL, timeout=5) as url:
            photo_data = json.load(url)
        df_photos = pd.DataFrame(photo_data)
    except Exception as e:
        print("Failed to load photo data:", e)
        return html.Div("Failed to load photos.", style={'textAlign': 'center', 'color': 'gray'})

    if df_photos.empty:
        return html.Div("No photos available.", style={'textAlign': 'center', 'color': 'gray'})

    cards = []
    for _, row in df_photos.iterrows():
        cards.append(html.Div([
            html.Img(src=row['PhotoURL'], style={'width': '100%', 'height': '220px', 'objectFit': 'cover', 'borderRadius': '8px'}),
            html.Div([
                html.Div(row['Subject'], style={'fontWeight': 'bold', 'fontSize': '16px'}),
                html.Div(f"{row['OfficeTitle']} • {row['Party']} • {row['State']}", style={'fontSize': '12px', 'color': '#666'})
            ], style={'marginTop': '8px'})
        ], style={
            'width': '180px',
            'margin': '10px',
            'padding': '10px',
            'border': '1px solid #ddd',
            'borderRadius': '10px',
            'textAlign': 'center',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)'
        }))

    return html.Div(cards, style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'center'
    })

# The rest of your existing update_dashboard() callback stays the same.
# You may paste or reuse your previous scorecard, time series, trait, bill, and issue logic.

if __name__ == '__main__':
    app.run(debug=False)



