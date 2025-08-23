iimport dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json
from datetime import datetime

# Minimal working version with dropdown + scorecard only
BASE_URL = "https://e8eb17633693.ngrok-free.app"
SCORECARD_URL = f"{BASE_URL}/api/scorecard"
PHOTOS_URL = f"{BASE_URL}/api/subject-photos"

app = dash.Dash(__name__)
server = app.server

def fetch_df(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return pd.DataFrame(json.load(r))
    except Exception as e:
        print(f"Fetch failed for {url}: {e}")
        return pd.DataFrame()

# Preload subject list
try:
    scorecard_df = fetch_df(SCORECARD_URL)
    subjects = sorted(scorecard_df["Subject"].dropna().unique())
except:
    subjects = []

app.layout = html.Div([
    html.Div([
        html.H1("Test Scorecard Only", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{"label": s, "value": s} for s in subjects],
            value=subjects[0] if subjects else None,
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '40px'}),

    html.Div(id='scorecard-div')
])

@app.callback(
    Output('scorecard-div', 'children'),
    Input('subject-dropdown', 'value')
)
def update_scorecard(subject):
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
    return html.Div([
        html.Img(src=photo_url, style={'width': '140px', 'height': '170px'}) if photo_url else html.Div(style={'width': '140px', 'height': '170px', 'background': '#eee'}),
        html.Div([
            html.H1(subject),
            html.Div(f"{office} • {party} • {state}"),
            html.Div("Sentiment Score", style={'marginTop': '14px'}),
            html.Div(f"{score:,}", style={'fontSize': '56px', 'color': color, 'fontWeight': 'bold'})
        ])
    ], style={'display': 'flex', 'gap': '24px'})

if __name__ == '__main__':
    app.run(debug=True)










