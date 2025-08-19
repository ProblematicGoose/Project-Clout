import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json

# URLs for live data
SCORECARD_URL = "https://efc7f5a4c6c7.ngrok-free.app/api/scorecard"
TIMESERIES_URL = "https://7a4c2efc8dcb.ngrok-free.app/api/timeseries"
TRAITS_URL = "https://7a4c2efc8dcb.ngrok-free.app/api/traits"
BILL_SENTIMENT_URL = "https://7a4c2efc8dcb.ngrok-free.app/api/bill-sentiment"
TOP_ISSUES_URL = "https://7a4c2efc8dcb.ngrok-free.app/api/top-issues"
COMMON_GROUND_URL = "https://7a4c2efc8dcb.ngrok-free.app/api/common-ground-issues"

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
    html.Div(id='common-ground-div', className='card', style={'marginTop': '40px'})
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
    print("Selected subject:", selected_subject)

    try:
        with urllib.request.urlopen(SCORECARD_URL, timeout=5) as url:
            scorecard_data = json.load(url)
        df_scorecard = pd.DataFrame(scorecard_data)
        score_row = df_scorecard[df_scorecard['Subject'] == selected_subject]
        score = int(score_row['NormalizedSentimentScore'].iloc[0]) if not score_row.empty else 5000
        photo_url = score_row['PhotoURL'].iloc[0] if 'PhotoURL' in score_row.columns and not score_row.empty else None
        office = score_row['OfficeTitle'].iloc[0] if 'OfficeTitle' in score_row.columns and not score_row.empty else ''
        party = score_row['Party'].iloc[0] if 'Party' in score_row.columns and not score_row.empty else ''
        state = score_row['State'].iloc[0] if 'State' in score_row.columns and not score_row.empty else ''
    except Exception as e:
        print("Failed to load scorecard data:", e)
        score, photo_url, office, party, state = 5000, None, '', '', ''

    color = 'green' if score > 6000 else 'crimson' if score < 4000 else 'orange'
    scorecard_display = html.Div([
        html.Div([
            html.Img(
                src=photo_url,
                style={
                    'width': '140px', 'height': '170px',
                    'objectFit': 'cover', 'borderRadius': '12px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                }
            ) if photo_url else html.Div(style={'width': '140px', 'height': '170px', 'background': '#eee'}),
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '24px'}),

        html.Div([
            html.H1(selected_subject, style={'fontSize': '42px', 'fontWeight': 'bold'}),
            html.Div(f"{office}{' • ' if office and (party or state) else ''}{party}{' • ' if party and state else ''}{state}",
                     style={'fontSize': '16px', 'color': '#666', 'marginTop': '4px'}),
            html.Div("Sentiment Score", style={'fontSize': '22px', 'color': 'gray', 'marginTop': '14px'}),
            html.Div(f"{score:,}", style={'fontSize': '56px', 'color': color, 'fontWeight': 'bold'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top'})
    ])

    # The rest of the function remains unchanged...
    # Add your existing timeseries, traits, bill sentiment, top issues, and common ground logic here

if __name__ == '__main__':
    app.run(debug=False)


