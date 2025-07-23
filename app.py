import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json

# URLs for live data
SCORECARD_URL = "https://8b0a3ec14958.ngrok-free.app/api/scorecard"
TIMESERIES_URL = "https://8b0a3ec14958.ngrok-free.app/api/timeseries"
TRAITS_URL = "https://8b0a3ec14958.ngrok-free.app/api/traits"

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load subjects dynamically from API
with urllib.request.urlopen(SCORECARD_URL) as url:
    subjects_data = json.load(url)
subjects = sorted({entry["Subject"].strip() for entry in subjects_data})

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("Sentiment Dashboard", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{'label': subj, 'value': subj} for subj in subjects],
            value=next(iter(subjects)),
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '40px'}),

    html.Div(id='scorecard-div', className='card'),

    html.Div([
        dcc.Graph(id='timeseries-graph')
    ], className='card'),

    html.Div(id='traits-div', className='card')
])


# Callbacks
@app.callback(
    Output('scorecard-div', 'children'),
    Output('timeseries-graph', 'figure'),
    Output('traits-div', 'children'),
    Input('subject-dropdown', 'value')
)
def update_dashboard(selected_subject):
    # Fetch scorecard data
    with urllib.request.urlopen(SCORECARD_URL) as url:
        scorecard_data = json.load(url)
    df_scorecard = pd.DataFrame(scorecard_data)

    score_row = df_scorecard[df_scorecard['Subject'] == selected_subject]
    if score_row.empty:
        score = 5000  # fallback neutral score
    else:
        score = int(score_row['NormalizedSentimentScore'].iloc[0])

    # Determine color
    if score < 4000:
        color = 'crimson'
    elif score > 6000:
        color = 'green'
    else:
        color = 'orange'

    scorecard_display = html.Div([
        html.H1(selected_subject, style={'fontSize': '50px', 'fontWeight': 'bold'}),
        html.Div("Sentiment Score", style={'fontSize': '30px', 'color': 'gray'}),
        html.Div(f"{score:,}", style={'fontSize': '80px', 'color': color})
    ])

    # Fetch time series data
    with urllib.request.urlopen(TIMESERIES_URL) as url:
        timeseries_data = json.load(url)
    df_timeseries = pd.DataFrame(timeseries_data)
    df_timeseries['SentimentDate'] = pd.to_datetime(df_timeseries['SentimentDate'])
    df_filtered = df_timeseries[df_timeseries['Subject'] == selected_subject]

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

    # Fetch traits data
    with urllib.request.urlopen(TRAITS_URL) as url:
        traits_data = json.load(url)
    df_traits = pd.DataFrame(traits_data)
    df_traits = df_traits[df_traits['Subject'] == selected_subject]

    positive = df_traits[df_traits['TraitType'] == 'Positive'].sort_values('TraitRank')['TraitDescription'].tolist()
    negative = df_traits[df_traits['TraitType'] == 'Negative'].sort_values('TraitRank')['TraitDescription'].tolist()

    trait_display = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(trait, style={'textAlign': 'left'}) for trait in positive], style={'listStyleType': 'none'}),
        html.H2("People don't like it when I...", style={'color': 'crimson', 'marginTop': '30px'}),
        html.Ul([html.Li(trait, style={'textAlign': 'left'}) for trait in negative], style={'listStyleType': 'none'})
    ])

    return scorecard_display, timeseries_fig, trait_display


if __name__ == '__main__':
    app.run_server(debug=False)


