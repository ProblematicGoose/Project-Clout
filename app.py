import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json

# URLs for live data
SCORECARD_URL = "https://dd7694e9231a.ngrok-free.app/api/top-issues/api/scorecard"
TIMESERIES_URL = "https://dd7694e9231a.ngrok-free.app/api/top-issues/api/timeseries"
TRAITS_URL = "https://dd7694e9231a.ngrok-free.app/api/top-issues/api/traits"
BILL_SENTIMENT_URL = "https://dd7694e9231a.ngrok-free.app/api/top-issues/api/bill-sentiment"
TOP_ISSUES_URL = "https://dd7694e9231a.ngrok-free.app/api/top-issues"

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load subjects dynamically from API
with urllib.request.urlopen(SCORECARD_URL) as url:
    subjects_data = json.load(url)
subjects = sorted({entry["Subject"].strip() for entry in subjects_data if entry["Subject"] is not None})

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

    html.Div(id='traits-div', className='card'),

    html.Div(id='bill-sentiment-table', className='card', style={'marginTop': '40px'}),

    html.Div([
        html.H2("Top 5 Issues by Ideology", style={'textAlign': 'center'}),
        dcc.Graph(id='ideology-issues-graph'),
        html.Div(id='ideology-week-label', style={'textAlign': 'center', 'color': 'gray'})
    ], className='card', style={'marginTop': '40px'})
])

# Callback for main dashboard
@app.callback(
    Output('scorecard-div', 'children'),
    Output('timeseries-graph', 'figure'),
    Output('traits-div', 'children'),
    Output('bill-sentiment-table', 'children'),
    Input('subject-dropdown', 'value')
)
def update_dashboard(selected_subject):
    # Scorecard
    with urllib.request.urlopen(SCORECARD_URL) as url:
        scorecard_data = json.load(url)
    df_scorecard = pd.DataFrame(scorecard_data)
    score_row = df_scorecard[df_scorecard['Subject'] == selected_subject]
    score = int(score_row['NormalizedSentimentScore'].iloc[0]) if not score_row.empty else 5000
    color = 'crimson' if score < 4000 else 'green' if score > 6000 else 'orange'
    scorecard_display = html.Div([
        html.H1(selected_subject, style={'fontSize': '50px', 'fontWeight': 'bold'}),
        html.Div("Sentiment Score", style={'fontSize': '30px', 'color': 'gray'}),
        html.Div(f"{score:,}", style={'fontSize': '80px', 'color': color})
    ])

    # Time series
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

    # Traits
    with urllib.request.urlopen(TRAITS_URL) as url:
        traits_data = json.load(url)
    df_traits = pd.DataFrame(traits_data)
    df_traits = df_traits[df_traits['Subject'] == selected_subject]
    positive = df_traits[df_traits['TraitType'] == 'Positive'].sort_values('TraitRank')['TraitDescription'].tolist()
    negative = df_traits[df_traits['TraitType'] == 'Negative'].sort_values('TraitRank')['TraitDescription'].tolist()
    trait_display = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(trait) for trait in positive], style={'listStyleType': 'none'}),
        html.H2("People don't like it when I...", style={'color': 'crimson', 'marginTop': '30px'}),
        html.Ul([html.Li(trait) for trait in negative], style={'listStyleType': 'none'})
    ])

    # Bill sentiment
    with urllib.request.urlopen(BILL_SENTIMENT_URL) as url:
        bill_data = json.load(url)
    df_bills = pd.DataFrame(bill_data)
    bill_table = html.Div([
        html.H2("Public Sentiment Toward National Bills", style={'textAlign': 'center'}),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(row['BillName']),
                    html.Td(round(row['AverageSentimentScore'], 2), style={'paddingRight': '30px'}),
                    html.Td(row['SentimentLabel'])
                ]) for _, row in df_bills.iterrows()
            ])
        ], style={'width': '80%', 'margin': '0 auto', 'borderCollapse': 'collapse'})
    ])

    return scorecard_display, timeseries_fig, trait_display, bill_table

# Callback for top issues by ideology
@app.callback(
    Output('ideology-issues-graph', 'figure'),
    Output('ideology-week-label', 'children'),
    Input('subject-dropdown', 'value')  # Dummy input to trigger load on page
)
def update_ideology_graph(_):
    with urllib.request.urlopen(TOP_ISSUES_URL) as url:
        issues_data = json.load(url)

    week_label = f"Issues from week starting {issues_data['WeekStartDate']}"

    liberal_issues = issues_data.get('Liberal', [])
    conservative_issues = issues_data.get('Conservative', [])

    liberal_topics = [issue['Topic'] for issue in liberal_issues]
    conservative_topics = [issue['Topic'] for issue in conservative_issues]
    liberal_ranks = [6 - issue['Rank'] for issue in liberal_issues]
    conservative_ranks = [6 - issue['Rank'] for issue in conservative_issues]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=liberal_ranks,
        y=liberal_topics,
        name='Liberal',
        orientation='h'
    ))
    fig.add_trace(go.Bar(
        x=conservative_ranks,
        y=conservative_topics,
        name='Conservative',
        orientation='h'
    ))

    fig.update_layout(
        barmode='group',
        title="Top 5 Topics by Ideology",
        xaxis_title="Importance (Higher = More Important)",
        yaxis_title="Topic",
        template='plotly_white'
    )

    return fig, week_label

if __name__ == '__main__':
    app.run_server(debug=True)

